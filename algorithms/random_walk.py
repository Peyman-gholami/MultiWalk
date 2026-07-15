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


class RandomWalk:
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

    def rw_computation(self, rank, queue, queue_op_id, comm_process_started, shared_state=None):
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
            logging.info(f"[Computation] wait at Rank {rank} - {comm_process_started.value} already started out of {len(self.parent.group_names)}")
            time.sleep(0.5)

        start_time = time.time()
        print(f"Rank {rank} start time: {start_time}")
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


            for param, queue_param in zip(parameters, queue[rw]):
                param_data = np.frombuffer(queue_param.get_obj(), dtype=np.float32).reshape(param.shape)
                param.data = torch.from_numpy(param_data).to(device)


            logging.info(f"[{group_name}] Rank {rank} performing training")
            logger.log_start("local sgd")
            epoch, gradients = self.parent.local_sgd(task, parameters, state, base_optimizer, base_optimizer_state, batch_data_gen,
                                          (time.time() - start_time), self.parent.tau)

            iteration += self.parent.tau
            logger.log_end("local sgd", {"rank": rank, "rw": group_name, "iteration": self.parent.tau, "epoch": epoch})

            # Update shared_state for current aggregator (not just rank 0)
            if shared_state is not None:
                for st, shared_array in zip(state, shared_state):
                    np.copyto(np.frombuffer(shared_array.get_obj(), dtype=np.float32).reshape(st.shape),
                              st.cpu().detach().numpy())

            for param, queue_param in zip(parameters, queue[rw]):
                np.copyto(np.frombuffer(queue_param.get_obj(), dtype=np.float32).reshape(param.shape),
                          param.cpu().detach().numpy())

            queue_op_id[rw].value = 0


    def random_walk_generator(self, current_node, neighbor_dict, rng, stay_prob=0.0):
        while True:
            current_node = self.lazy_metropolis_hastings_step(current_node, neighbor_dict, rng, stay_prob)
            yield current_node


    def rw_communication(self, rank, rw, queue, queue_op_id, comm_process_started, shared_arrays=None, shared_array_index=None, eval_process_active=None):
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

        shapes = [param.shape for param in parameters]

        global_seed = self.parent.config["seed"] + rw
        rng = torch.Generator()
        rng.manual_seed(global_seed)
        start_rank = self.parent.rw_starting_ranks[rw]

        rw_generator = self.random_walk_generator(current_node=start_rank, neighbor_dict=self.parent.neighbors, rng=rng)
        current_rank = start_rank
        dist.barrier()
        logging.info(f"[{group_name}] Rank {rank} barrier passed")
        with self.parent.lock:
            comm_process_started.value += 1

        start_time = time.time()
        failure_times = sorted(self.parent.failure_times)  # Sort failure times
        failure_index = 0  # Track which failure we're at
        # Use a local variable to track aggregator rank for this walk, but sync with shared state if needed
        # Note: This assumes all walks see failures at similar times, which may not be perfect but works for most cases
        current_aggregator_rank = 0
        failure_just_happened = False
        iteration = 0

        while True:
            # Check if aggregator failure should occur
            current_time_minutes = (time.time() - start_time) / 60.0
            if failure_index < len(failure_times) and current_time_minutes >= failure_times[failure_index]:
                old_aggregator = current_aggregator_rank
                # Only deactivate eval_process if we're the old aggregator
                if eval_process_active is not None and eval_process_active.value == 1 and rank == old_aggregator:
                    eval_process_active.value = 0
                    logging.info(f"[{group_name}] Deactivating eval_process for aggregator {old_aggregator}")
                current_aggregator_rank += 1
                logging.info(f"[{group_name}] Aggregator failure at {current_time_minutes:.2f} minutes. Old aggregator: {old_aggregator}, New aggregator: {current_aggregator_rank}")
                failure_index += 1
                failure_just_happened = True
                

            next_rank = next(rw_generator)
            logging.info(f"[{group_name}] Iteration {iteration} at Rank {rank}, Current Active Rank: {current_rank}")
            # dist.barrier()

            if rank == current_rank:
                if rank >= current_aggregator_rank:
                    logging.info(f"[{group_name}] Rank {rank} performing training")

                    queue_op_id[rw].value = 1
                    while queue_op_id[rw].value != 0:
                        time.sleep(0.01)
                    iteration += self.parent.tau


                    for param, queue_param in zip(parameters, queue[rw]):
                        param_data = np.frombuffer(queue_param.get_obj(), dtype=np.float32).reshape(param.shape)
                        param.data = torch.from_numpy(param_data).to(device)



                    if rank == current_aggregator_rank:
                        with self.parent.lock:
                            if failure_just_happened:
                                # First time: just update with received model
                                logging.info(f"[{group_name}] New aggregator {rank} first time receiving walk {rw}, updating directly")
                                for param, shared_array in zip(parameters, shared_arrays[rw]):
                                    np.copyto(np.frombuffer(shared_array.get_obj(), dtype=np.float32).reshape(param.shape), 
                                              param.cpu().detach().numpy())
                                # Activate eval_process for new aggregator when first walk arrives
                                if eval_process_active is not None and eval_process_active.value == 2:
                                    logging.info(f"[{group_name}] Activating eval_process for new aggregator {rank} after receiving first walk {rw}")
                                    eval_process_active.value = 1
                                failure_just_happened = False  # Reset flag after handling first reception
                            else:
                                # Not first time: perform aggregation
                                logging.info(f"[{group_name}] Aggregator {rank} aggregating walk {rw}")
                                for param, rw_pre_param_shared_array, global_pre_param_shared_array in zip(parameters, shared_arrays[rw], shared_arrays[shared_array_index.value]):
                                    rw_pre_param = np.frombuffer(rw_pre_param_shared_array.get_obj(), dtype=np.float32).reshape(param.shape)
                                    global_pre_param = np.frombuffer(global_pre_param_shared_array.get_obj(), dtype=np.float32).reshape(param.shape)
                                    param.data = torch.from_numpy(global_pre_param).to(device) + 1 / len(self.parent.group_names) * (param - torch.from_numpy(rw_pre_param).to(device))

                                for param, shared_array in zip(parameters, shared_arrays[rw]):
                                    np.copyto(np.frombuffer(shared_array.get_obj(), dtype=np.float32).reshape(param.shape), param.cpu().detach().numpy())

                            shared_array_index.value = rw


                else:
                    logging.info(f"[{group_name}] Rank {rank} acting as relay, forwarding model to Rank {next_rank}")
                    print("Problem why am I here?????")
                    for param, queue_param in zip(parameters, queue[rw]):
                        param_data = np.frombuffer(queue_param.get_obj(), dtype=np.float32).reshape(param.shape)
                        param.data = torch.from_numpy(param_data).to(device)

                logging.info(f"[{group_name}] Rank {rank} sending model to Rank {next_rank}")

                if next_rank != rank:
                    with self.parent.lock_comm_send:
                        logging.info(f"[{group_name}]  Rank {rank} notifying and exchanging model with Rank {next_rank}")

                        notification = torch.tensor(rank, dtype=torch.int32).to(device)
                        dist.send(tensor=notification, dst=next_rank)

                        if rank >= current_aggregator_rank:
                            logger.log_start("communication")
                        buffer = pack(parameters)
                        dist.send(tensor=buffer, dst=next_rank)
                        bytes_sent = num_bytes(buffer)
                        if rank >= current_aggregator_rank:
                            logger.log_end("communication", {"rw": group_name, "from": rank, "to": next_rank, "bytes_sent": bytes_sent})
                else:
                    for param, queue_param in zip(parameters, queue[rw]):
                        np.copyto(np.frombuffer(queue_param.get_obj(), dtype=np.float32).reshape(param.shape),
                                  param.cpu().detach().numpy())


            elif rank == next_rank:

                logging.info(f"[{group_name}]  Rank {rank} waiting for notification")

                tensor = torch.full((1,), -1, dtype=torch.int32).to(device)
                dist.recv(tensor=tensor, src=current_rank)

                with self.parent.lock_comm_recv:
                    logging.info(f"[{group_name}] Rank {rank} waiting for model from Rank {current_rank}")
                    buffer = torch.zeros_like(pack(parameters))
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
        
        # Determine potential aggregator nodes (0, 1, 2, ... up to size-1, but limit to reasonable number)
        max_potential_aggregators = min(self.parent.size, len(self.parent.failure_times) + 1)
        potential_aggregator_ranks = list(range(max_potential_aggregators))
        
        # Initialize aggregator capabilities on all potential aggregator nodes
        if rank in potential_aggregator_ranks:
            for _ in range(len(self.parent.group_names)):
                group_shared_arrays = [Array('f', param.numel(), lock=True) for param in model.parameters()]
                shared_arrays.append(group_shared_arrays)
                # Initialize as zero for all potential aggregators except rank 0
                if rank == 0:
                    for param, shared_array in zip(model.parameters(), group_shared_arrays):
                        np.copyto(np.frombuffer(shared_array.get_obj(), dtype=np.float32).reshape(param.shape),
                                  param.cpu().detach().numpy())
                else:
                    # Initialize as zero for backup aggregators
                    for shared_array in group_shared_arrays:
                        np.frombuffer(shared_array.get_obj(), dtype=np.float32).fill(0.0)

            shared_array_index = Value('i', 0)
            shared_state = [Array('f', state.numel(), lock=True) for state in model.buffers()]
            for state, shared_array in zip(model.buffers(), shared_state):
                np.copyto(np.frombuffer(shared_array.get_obj(), dtype=np.float32).reshape(state.shape),
                            state.cpu().detach().numpy())
            
            eval_process_active = Value('i', 2)  # Start as inactive for backup aggregators
            if rank == 0:
                eval_process_active.value = 1
            eval_process = Process(target=self.parent.evaluation_process, args=(
                self.parent.eval_gpu, shared_arrays, shared_state, eval_process_active, shared_array_index))
            eval_process.start()

        queue = []
        for _ in range(len(self.parent.group_names)):
            group_queue = [Array('f', param.numel(), lock=True) for param in model.parameters()]
            queue.append(group_queue)
            for param, queue_param in zip(model.parameters(), group_queue):
                np.copyto(np.frombuffer(queue_param.get_obj(), dtype=np.float32).reshape(param.shape),
                          param.cpu().detach().numpy())
        queue_op_id = [Value('i', 0) for _ in
                       range(len(queue))]  # 1 means need to do local sgd on that model for spesific RW

        comm_process_started = Value('i', 0)

        comm_processes = []
        if rank in potential_aggregator_ranks:
            compute_process = Process(target=self.rw_computation,
                                      args=(rank, queue, queue_op_id, comm_process_started, shared_state))
        else:
            compute_process = Process(target=self.rw_computation, args=(rank, queue, queue_op_id, comm_process_started))

        compute_process.start()

        for rw in range(len(self.parent.group_names)):
            if rank in potential_aggregator_ranks:
                comm_process = Process(target=self.rw_communication, args=(
                    rank, rw, queue, queue_op_id, comm_process_started, shared_arrays, shared_array_index, 
                    eval_process_active,))
            else:
                comm_process = Process(target=self.rw_communication,
                                       args=(rank, rw, queue, queue_op_id, comm_process_started))
            comm_process.start()
            comm_processes.append(comm_process)

        compute_process.join()
        for p in comm_processes:
            p.terminate()

        if rank in potential_aggregator_ranks:
            eval_process_active.value = 0  # Signal termination
            eval_process.join()

