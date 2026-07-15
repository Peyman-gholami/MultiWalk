import torch
import torch.distributed as dist
import logging
import numpy as np
from torch.multiprocessing import Process, Array, Value
import time
from base_optimizers import configure_base_optimizer
from utils.communication import (
    pack,
    unpack,
    num_bytes
)
from logger import EventLogger


class SplitRandomWalk:
    def __init__(self, parent):
        self.parent = parent

    def lazy_metropolis_hastings_step(self, current_node, neighbor_dict, rng, stay_prob=0.0):
        """
        Performs a single step in a lazy Metropolis-Hastings random walk on a graph.

        Parameters:
        - current_node (any hashable type): The current node in the graph.
        - neighbor_dict (dict): A dictionary where keys are nodes and values are lists of neighboring nodes.
        - rng (torch.Generator): The random number generator for consistency across processes.
        - stay_prob (float): Probability of staying at the current node.

        Returns:
        - any hashable type: The next node in the random walk.
        """
        # Determine whether to stay at the current node based on stay_prob
        if torch.rand(1, generator=rng).item() < stay_prob:
            return current_node

        # If not staying, proceed with the usual Metropolis-Hastings step
        neighbors = neighbor_dict[current_node]

        if not neighbors:
            return current_node  # Stay at the current node if no neighbors are available

        chosen_neighbor = neighbors[torch.randint(0, len(neighbors), (1,), generator=rng).item()]
        current_degree = len(neighbor_dict[current_node])
        neighbor_degree = len(neighbor_dict[chosen_neighbor])

        acceptance_probability = min(1, current_degree / neighbor_degree)

        if torch.rand(1, generator=rng).item() < acceptance_probability:
            return chosen_neighbor
        else:
            return current_node

    def rw_computation(self, rank, queue, queue_op_id, comm_process_started, separation_point, shared_state=None):
        torch.manual_seed(self.parent.config["seed"] + rank )
        np.random.seed(self.parent.config["seed"] + rank )

        device = torch.device(f'cuda:{self.parent.local_rank}' if torch.cuda.is_available() else 'cpu')

        logger = EventLogger(log_file_name=self.parent.log_name)

        task = self.parent.configure_task(rank, device)
        parameters, state = task.initialize(self.parent.config["seed"])
        base_optimizer = configure_base_optimizer(self.parent.config)
        base_optimizer_state = base_optimizer.init(parameters)

        batch_data_gen = task.data.iterator(
            batch_size=self.parent.config["batch_size"],
            shuffle=True,
        )


        while comm_process_started.value != len(self.parent.group_names): #wait for communication process to start
            logging.info(f"[Computation] wait at Rank {rank}")
            time.sleep(0.5)

        start_time = time.time()
        end_time = start_time + self.parent.train_time * 60  # Convert minutes to seconds
        iteration = 0



        while time.time() < end_time:
            rw = None
            while time.time() < end_time:
                for rw_id, rw_op_id in enumerate(queue_op_id):
                    if rw_op_id.value == 1:
                        rw = rw_id
                        break
                if rw is not None:
                    break
                time.sleep(0.01)

            group_name = self.parent.group_names[rw]
            logging.info(f"[{group_name}] Rank {rank} performing training in compute")

            # Assemble full parameters from two queues based on parity and separation_point
            num_params = len(parameters)
            assert 0 <= separation_point < num_params - 1, "Invalid separation_point"

            is_representation = (rw % 2 == 0)
            pair_rw = rw + 1 if is_representation else rw - 1

            # Load representation part (0..separation_point)
            rep_indices = list(range(0, separation_point + 1))
            head_indices = list(range(separation_point + 1, num_params))

            # Own queue provides own part; paired queue provides the other part
            own_part_arrays = queue[rw]
            pair_part_arrays = queue[pair_rw]

            if is_representation:
                # Load representation from own queue
                for idx, queue_param in zip(rep_indices, own_part_arrays):
                    param = parameters[idx]
                    param_data = np.frombuffer(queue_param.get_obj(), dtype=np.float32).reshape(param.shape)
                    param.data = torch.from_numpy(param_data).to(device)
                # Load head from paired queue
                for idx, queue_param in zip(head_indices, pair_part_arrays):
                    param = parameters[idx]
                    param_data = np.frombuffer(queue_param.get_obj(), dtype=np.float32).reshape(param.shape)
                    param.data = torch.from_numpy(param_data).to(device)
            else:
                # Load head from own queue
                for idx, queue_param in zip(head_indices, own_part_arrays):
                    param = parameters[idx]
                    param_data = np.frombuffer(queue_param.get_obj(), dtype=np.float32).reshape(param.shape)
                    param.data = torch.from_numpy(param_data).to(device)
                # Load representation from paired queue
                for idx, queue_param in zip(rep_indices, pair_part_arrays):
                    param = parameters[idx]
                    param_data = np.frombuffer(queue_param.get_obj(), dtype=np.float32).reshape(param.shape)
                    param.data = torch.from_numpy(param_data).to(device)

            split_random_walk_ratio = self.parent.config["split_random_walk_ratio"] if is_representation else 1
            logging.info(f"[{group_name}] Rank {rank} performing training")
            logger.log_start("local sgd")
            epoch, gradients = self.parent.local_sgd(task, parameters, state, base_optimizer, base_optimizer_state, batch_data_gen,
                                          (time.time() - start_time), self.parent.tau * split_random_walk_ratio)

            iteration += self.parent.tau * split_random_walk_ratio
            logger.log_end("local sgd", {"rank": rank, "rw": group_name, "iteration": self.parent.tau * split_random_walk_ratio, "epoch": epoch})

            if rank == 0:
                for st, shared_array in zip(state, shared_state):
                    np.copyto(np.frombuffer(shared_array.get_obj(), dtype=np.float32).reshape(st.shape),
                              st.cpu().detach().numpy())

            # Write back only the part owned by this walk
            if is_representation:
                for idx, queue_param in zip(rep_indices, own_part_arrays):
                    param = parameters[idx]
                    np.copyto(np.frombuffer(queue_param.get_obj(), dtype=np.float32).reshape(param.shape),
                              param.cpu().detach().numpy())
            else:
                for idx, queue_param in zip(head_indices, own_part_arrays):
                    param = parameters[idx]
                    np.copyto(np.frombuffer(queue_param.get_obj(), dtype=np.float32).reshape(param.shape),
                              param.cpu().detach().numpy())

            queue_op_id[rw].value = 0


    def random_walk_generator(self, current_node, neighbor_dict, rng, stay_prob=0.0):
        while True:
            current_node = self.lazy_metropolis_hastings_step(current_node, neighbor_dict, rng, stay_prob)
            yield current_node


    def rw_communication(self, rank, rw, queue, queue_op_id, comm_process_started, separation_point, shared_arrays=None, shared_array_index=None, eval_shared_arrays=None, last_part_indices=None):
        torch.manual_seed(self.parent.config["seed"] + rank + rw)
        np.random.seed(self.parent.config["seed"] + rank + rw)
        group_name = self.parent.group_names[rw]
        port = self.parent.ports[rw]
        backend = 'gloo'#nccl' if torch.cuda.is_available() else 'gloo'
        self.parent.init_process(rank, self.parent.size, backend, port, group_name)
        device = 'cpu'#torch.device(f'cuda:{self.parent.local_rank}' if torch.cuda.is_available() else 'cpu')

        logger = EventLogger(log_file_name=self.parent.log_name)

        with torch.no_grad():
            model = self.parent.create_model()
            #Move model parameters to GPU
            parameters = [param.to(device) for param in model.parameters()]
        # Determine which part this RW handles
        num_params = len(parameters)
        assert 0 <= separation_point < num_params - 1, "Invalid separation_point"
        is_representation = (rw % 2 == 0)
        rep_indices = list(range(0, separation_point + 1))
        head_indices = list(range(separation_point + 1, num_params))
        part_indices = rep_indices if is_representation else head_indices
        shapes = [parameters[idx].shape for idx in part_indices]

        global_seed = self.parent.config["seed"] + rw
        rng = torch.Generator()
        rng.manual_seed(global_seed)
        start_rank = self.parent.rw_starting_ranks[rw]

        rw_generator = self.random_walk_generator(current_node=start_rank, neighbor_dict=self.parent.neighbors, rng=rng)
        current_rank = start_rank

        with self.parent.lock:
            comm_process_started.value += 1

        # start_time = time.time()
        # end_time = start_time + self.parent.train_time * 60  # Convert minutes to seconds
        iteration = 0


        while True:
            next_rank = next(rw_generator)
            logging.info(f"[{group_name}] Iteration {iteration} at Rank {rank}, Current Active Rank: {current_rank}")
            # dist.barrier()

            if rank == current_rank:
                if True: #with self.parent.lock:
                    logging.info(f"[{group_name}] Rank {rank} performing training")

                    queue_op_id[rw].value = 1
                    while queue_op_id[rw].value != 0:
                        time.sleep(0.01)
                    iteration += self.parent.tau

                    # Load only this RW's part from its queue into parameters
                    for idx, queue_param in zip(part_indices, queue[rw]):
                        param = parameters[idx]
                        param_data = np.frombuffer(queue_param.get_obj(), dtype=np.float32).reshape(param.shape)
                        param.data = torch.from_numpy(param_data).to(device)

                    if rank == 0 and shared_arrays is not None and eval_shared_arrays is not None and last_part_indices is not None:
                        with self.parent.lock:
                            # Update this group's shared arrays (partial)
                            for idx, rw_pre_param_shared_array, global_pre_param_shared_array in zip(part_indices, shared_arrays[rw], shared_arrays[last_part_indices[rw].value]):
                                rw_pre_param = np.frombuffer(rw_pre_param_shared_array.get_obj(), dtype=np.float32).reshape(parameters[idx].shape)
                                global_pre_param = np.frombuffer(global_pre_param_shared_array.get_obj(), dtype=np.float32).reshape(parameters[idx].shape)
                                parameters[idx].data = torch.from_numpy(global_pre_param).to(device) + 1 / 2*len(self.parent.group_names) * (parameters[idx] - torch.from_numpy(rw_pre_param).to(device))

                            for idx, shared_array in zip(part_indices, shared_arrays[rw]):
                                param = parameters[idx]
                                np.copyto(np.frombuffer(shared_array.get_obj(), dtype=np.float32).reshape(param.shape), param.cpu().detach().numpy())

                            # Record last updater indices per part
                            if is_representation:
                                last_part_indices[0].value = rw
                            else:
                                last_part_indices[1].value = rw

                            # Merge parts into full eval arrays using latest of both parts
                            pair_rw = rw + 1 if is_representation else rw - 1
                            # Update representation part from last even (or current if even)
                            rep_src_rw = last_part_indices[0].value
                            head_src_rw = last_part_indices[1].value
                            # Write rep part
                            for idx, shared_array_partial in zip(rep_indices, shared_arrays[rep_src_rw]):
                                full_array = eval_shared_arrays[idx]
                                partial_np = np.frombuffer(shared_array_partial.get_obj(), dtype=np.float32).reshape(parameters[idx].shape)
                                np.copyto(np.frombuffer(full_array.get_obj(), dtype=np.float32).reshape(parameters[idx].shape), partial_np)
                            # Write head part
                            for idx, shared_array_partial in zip(head_indices, shared_arrays[head_src_rw]):
                                full_array = eval_shared_arrays[idx]
                                partial_np = np.frombuffer(shared_array_partial.get_obj(), dtype=np.float32).reshape(parameters[idx].shape)
                                np.copyto(np.frombuffer(full_array.get_obj(), dtype=np.float32).reshape(parameters[idx].shape), partial_np)

                    logging.info(f"[{group_name}] Rank {rank} sending model to Rank {next_rank}")

                if next_rank != rank:
                    with self.parent.lock_comm_send:
                        logging.info(f"[{group_name}]  Rank {rank} notifying and exchanging model with Rank {next_rank}")

                        notification = torch.tensor(rank, dtype=torch.int32).to(device)
                        dist.send(tensor=notification, dst=next_rank)

                        logger.log_start("communication")
                        # Pack only the relevant part
                        buffer = pack([parameters[idx] for idx in part_indices])
                        dist.send(tensor=buffer, dst=next_rank)
                        bytes_sent = num_bytes(buffer)
                        logger.log_end("communication", {"rw": group_name, "from": rank, "to": next_rank, "bytes_sent": bytes_sent})
                else:
                    for idx, queue_param in zip(part_indices, queue[rw]):
                        param = parameters[idx]
                        np.copyto(np.frombuffer(queue_param.get_obj(), dtype=np.float32).reshape(param.shape),
                                  param.cpu().detach().numpy())


            elif rank == next_rank:

                logging.info(f"[{group_name}]  Rank {rank} waiting for notification")

                tensor = torch.full((1,), -1, dtype=torch.int32).to(device)
                dist.recv(tensor=tensor, src=current_rank)

                with self.parent.lock_comm_recv:
                    logging.info(f"[{group_name}] Rank {rank} waiting for model from Rank {current_rank}")
                    buffer = torch.zeros_like(pack([parameters[idx] for idx in part_indices]))
                    dist.recv(tensor=buffer, src=current_rank)
                    new_params = unpack(buffer, shapes)

                for new_param, queue_param in zip(new_params, queue[rw]):
                    np.copyto(np.frombuffer(queue_param.get_obj(), dtype=np.float32).reshape(new_param.shape),
                              new_param.cpu().detach().numpy())

                # for param, new_param in zip(model.parameters(), new_params):
                #     param.data = new_param
                # logging.info(f"[{group_name}] Rank {rank} received model from Rank {current_rank.item()}")
            current_rank = next_rank

        dist.barrier()
        dist.destroy_process_group()
        logging.info(f"[{group_name}] Process {rank} finished")

    def run(self, rank):
        model = self.parent.create_model()
        shared_arrays = []
        separation_point = self.parent.get_model_separation_point(model)
        num_params = len(list(model.parameters()))
        assert 0 <= separation_point < num_params - 1, "Invalid separation_point"
        rep_indices = list(range(0, separation_point + 1))
        head_indices = list(range(separation_point + 1, num_params))

        # Per-group partial shared arrays (for tracking latest partials), and full eval arrays for evaluation
        if rank == 0:
            for gi in range(len(self.parent.group_names)):
                is_representation = (gi % 2 == 0)
                indices = rep_indices if is_representation else head_indices
                group_shared_arrays = [Array('f', list(model.parameters())[idx].numel(), lock=True) for idx in indices]
                shared_arrays.append(group_shared_arrays)
                for idx, shared_array in zip(indices, group_shared_arrays):
                    param = list(model.parameters())[idx]
                    np.copyto(np.frombuffer(shared_array.get_obj(), dtype=np.float32).reshape(param.shape), param.cpu().detach().numpy())

            # Full arrays for evaluation (assembled from latest parts)
            eval_shared_arrays = [Array('f', param.numel(), lock=True) for param in model.parameters()]
            for param, shared_array in zip(model.parameters(), eval_shared_arrays):
                np.copyto(np.frombuffer(shared_array.get_obj(), dtype=np.float32).reshape(param.shape), param.cpu().detach().numpy())

            # Track last updater indices for representation (even) and head (odd)
            last_part_indices = [Value('i', 0), Value('i', 1 if len(self.parent.group_names) > 1 else 0)]

            shared_state = [Array('f', state.numel(), lock=True) for state in model.buffers()]
            eval_process_active = Value('i', 1)
            eval_process = Process(target=self.parent.evaluation_process, args=(self.parent.eval_gpu, eval_shared_arrays, shared_state, eval_process_active, None))
            eval_process.start()

        # Per-group queues sized to the relevant part only
        queue = []
        for gi in range(len(self.parent.group_names)):
            is_representation = (gi % 2 == 0)
            indices = rep_indices if is_representation else head_indices
            group_queue = [Array('f', list(model.parameters())[idx].numel(), lock=True) for idx in indices]
            queue.append(group_queue)
            for idx, queue_param in zip(indices, group_queue):
                param = list(model.parameters())[idx]
                np.copyto(np.frombuffer(queue_param.get_obj(), dtype=np.float32).reshape(param.shape),
                          param.cpu().detach().numpy())
        queue_op_id = [Value('i', 0) for _ in
                       range(len(queue))]  # 1 means need to do local sgd on that model for spesific RW

        comm_process_started = Value('i', 0)

        comm_processes = []
        if rank == 0:
            compute_process = Process(target=self.rw_computation,
                                      args=(rank, queue, queue_op_id, comm_process_started, separation_point, shared_state))
        else:
            compute_process = Process(target=self.rw_computation, args=(rank, queue, queue_op_id, comm_process_started, separation_point))

        compute_process.start()

        for rw in range(len(self.parent.group_names)):
            if rank == 0:
                comm_process = Process(target=self.rw_communication, args=(rank, rw, queue, queue_op_id, comm_process_started, separation_point, shared_arrays, None, eval_shared_arrays, last_part_indices,))
            else:
                comm_process = Process(target=self.rw_communication,
                                       args=(rank, rw, queue, queue_op_id, comm_process_started, separation_point))
            comm_process.start()
            comm_processes.append(comm_process)

        compute_process.join()
        for p in comm_processes:
            p.terminate()


        if rank == 0:
            eval_process_active.value = 0
            eval_process.join()