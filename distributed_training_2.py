import torch
import torch.distributed as dist
import random
import logging
import numpy as np
from torch.multiprocessing import Process, Array, Value, Lock, set_start_method
import time
from tasks.cifar import fork_rng_with_seed
import os
from datetime import datetime
import pytz
from tasks.api import Task
from base_optimizers import configure_base_optimizer


# set_start_method('spawn', force=True)

logging.basicConfig(level=logging.CRITICAL)
# logging.basicConfig(level=logging.INFO)


class EventLogger:
    def __init__(self, log_file_name='events.log'):
        # Ensure log directory exists
        log_directory = './log/'
        os.makedirs(log_directory, exist_ok=True)

        self.log_file_path = os.path.join(log_directory, log_file_name)
        self.setup_logger()

        self.event_start_times = {}

    def setup_logger(self):
        # Create a logger and set the logging level
        self.logger = logging.getLogger(self.log_file_path)
        self.logger.setLevel(logging.INFO)

        # Check if the logger already has handlers
        if not self.logger.handlers:
            # Create a file handler that logs even debug messages
            file_handler = logging.FileHandler(self.log_file_path, mode='a')
            file_handler.setLevel(logging.INFO)

            # Create a logging format
            formatter = logging.Formatter('%(message)s')
            file_handler.setFormatter(formatter)

            # Add the handler to the logger
            self.logger.addHandler(file_handler)

    def log_start(self, event_name):
        if event_name in self.event_start_times:
            raise ValueError(f'Event "{event_name}" already exists. Cannot start the same event twice.')
        self.event_start_times[event_name] = time.time()

    def log_end(self, event_name, attribute_value):
        if not isinstance(attribute_value, dict):
            raise ValueError('attribute_value must be a dictionary')
        cst = pytz.timezone('US/Central')
        end_time = datetime.now(cst).strftime('%Y-%m-%d %H:%M:%S %Z')
        start_timestamp = self.event_start_times.pop(event_name, None)
        if start_timestamp is not None:
            duration = time.time() - start_timestamp
            start_time = datetime.fromtimestamp(start_timestamp, cst).strftime('%Y-%m-%d %H:%M:%S %Z')
            attributes_str = ', '.join(f'{key}: {value}' for key, value in attribute_value.items())
            self.logger.info(
                f'{event_name}, start_time: {start_time}, end_time: {end_time}, duration: {duration:.2f}s, attributes: {{{attributes_str}}}')
        else:
            self.logger.warning(f'End log for event {event_name} called without a corresponding start log.')



class DecentralizedTraining:
    def __init__(self, size, local_rank, tau, train_time, neighbors, top_nodes, rw_starting_ranks, master_address, ports, group_names, algorithm, config, evaluate_interval,
                 eval_gpu, train_eval_frac, log_name):
        self.size = size
        self.local_rank = local_rank
        self.tau = tau
        self.train_time = train_time
        self.neighbors = neighbors
        self.top_nodes = top_nodes
        self.rw_starting_ranks = rw_starting_ranks
        self.master_address = master_address
        self.ports = ports
        self.group_names = group_names
        self.algorithm = algorithm
        self.config = config
        self.lock = Lock()
        self.evaluate_interval = evaluate_interval
        self.eval_gpu = eval_gpu
        self.train_eval_frac = train_eval_frac
        self.log_name = log_name

    def init_process(self, rank, size, backend, port, group_name):
        init_method = f'tcp://{self.master_address}:{port}'
        dist.init_process_group(backend, rank=rank, world_size=size, init_method=init_method)
        logging.info(f"[{group_name}] Rank {rank} initialized with backend {backend} on port {port}")

    def configure_task(self, rank, device) -> Task:
        if self.config["task"] == "Cifar":
            from tasks.cifar import CifarTask

            return CifarTask(
                rank=rank,
                device=device,
                num_workers=self.size,
                weight_decay=self.config["weight_decay"],
                model_name=self.config["model_name"],
                data_split_method=self.config["data_split_method"],
                train_eval_frac=self.train_eval_frac,
                lock=self.lock,
                non_iid_alpha=self.config["non_iid_alpha"],
                seed=self.config["seed"] + 100
            )

    def create_model(self):
        if self.config["model_name"] == "ResNet20":
            from tasks.models.resnet20 import ResNet20
            with fork_rng_with_seed(self.config["seed"]):
                model = ResNet20()
        return model

    def pack(self, tensors):
        buffer = torch.cat([t.view(-1) for t in tensors])
        return buffer

    def unpack(self, buffer, shapes):
        idx = 0
        entries = []
        for tensor_shape in shapes:
            end = idx + tensor_shape.numel()
            entries.append(buffer[idx:end].view(size=tensor_shape))
            idx = end
        return entries

    def num_bytes(self, tensor):
        return tensor.nelement() * tensor.element_size()

    def learning_rate_schedule(self, time):
        lr = 1.0
        if self.size > 1 and self.config["lr_warmup_time"] > 0:
            warmup_time = self.config["lr_warmup_time"] * 60
            max_factor = 1.0
            factor = 0 + (max_factor - 0) * min(time / warmup_time, 1.0)
            lr *= factor
        for (milestone, factor) in self.config["lr_schedule_milestones"]:
            if time >= milestone:
                lr *= factor
            else:
                return lr
        return lr

    def evaluation_process(self, gpu_id, shared_arrays, shared_state, eval_process_active, shared_array_index=None):
        logger = EventLogger(log_file_name=self.log_name)
        device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        task = self.configure_task(-1, device)
        parameters, state = task.initialize(self.config["seed"])

        while eval_process_active.value == 1:
            time.sleep(self.evaluate_interval)
            if shared_array_index is not None:
                extracted_params = shared_arrays[shared_array_index.value]
            else:
                extracted_params = shared_arrays
            for param, shared_array in zip(parameters, extracted_params):
                param_data = np.frombuffer(shared_array.get_obj(), dtype=np.float32).reshape(param.shape)
                param.data = torch.from_numpy(param_data).to(device)
            for st, shared_array in zip(state, shared_state):
                st_data = np.frombuffer(shared_array.get_obj(), dtype=np.float32).reshape(st.shape)
                st.data = torch.from_numpy(st_data).to(device)

            logger.log_start("evaluation")
            test_loss = task.evaluate(task._test_data, parameters, state)
            train_loss = task.evaluate(task.data, parameters, state)
            logging.info(f"Evaluation at interval: Test loss: {test_loss}")
            logging.info(f"Evaluation at interval: Train loss: {train_loss}")
            logger.log_end("evaluation", {"test loss": test_loss, "train loss": train_loss})

    def local_sgd(self, task, parameters, state, base_optimizer, base_optimizer_state, batch_data_gen, time):
        for _ in range(self.tau):
            epoch, batch = next(batch_data_gen)
            last_loss, gradients, state = task.loss_and_gradient(parameters, state, batch)
            base_optimizer.step(
                parameters,
                gradients,
                base_optimizer_state,
                lr=self.config["learning_rate"] * self.learning_rate_schedule(time),
            )
            #print(last_loss)
        return epoch

    def run(self, rank):
        if self.algorithm == 'random_walk':
            random_walk = RandomWalk(self)
            random_walk.run(rank)
        elif self.algorithm == 'async_gossip':
            async_gossip = AsyncGossip(self)
            async_gossip.run(rank)

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

    def random_walk_generator(self, current_node, neighbor_dict, rng, stay_prob=0.0):
        while True:
            current_node = self.lazy_metropolis_hastings_step(current_node, neighbor_dict, rng, stay_prob)
            yield current_node

    def train_node(self, rank, rw, shared_arrays=None, shared_array_index=None, shared_state=None):
        torch.manual_seed(self.parent.config["seed"] + rank + rw)
        np.random.seed(self.parent.config["seed"] + rank + rw)
        group_name = self.parent.group_names[rw]
        port = self.parent.ports[rw]
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        self.parent.init_process(rank, self.parent.size, backend, port, group_name)
        device = torch.device(f'cuda:{self.parent.local_rank}' if torch.cuda.is_available() else 'cpu')

        logger = EventLogger(log_file_name=self.parent.log_name)

        task = self.parent.configure_task(rank, device)
        parameters, state = task.initialize(self.parent.config["seed"])
        base_optimizer = configure_base_optimizer(self.parent.config)
        base_optimizer_state = base_optimizer.init(parameters)

        shapes = [param.shape for param in parameters]

        batch_data_gen = task.data.iterator(
            batch_size=self.parent.config["batch_size"],
            shuffle=True,
        )

        global_seed = self.parent.config["seed"] + rw
        rng = torch.Generator()
        rng.manual_seed(global_seed)
        start_rank = 0

        rw_generator = self.random_walk_generator(current_node=start_rank, neighbor_dict=self.parent.neighbors, rng=rng)
        current_rank = start_rank
        start_time = time.time()
        end_time = start_time + self.parent.train_time * 60  # Convert minutes to seconds
        iteration = 0

        while time.time() < end_time:
            #dist.barrier()
            next_rank = next(rw_generator)
            logging.info(f"[{group_name}] Iteration {iteration} at Rank {rank}, Current Active Rank: {current_rank}")

            if rank == current_rank:
                if True:#with self.parent.lock:
                    logging.info(f"[{group_name}] Rank {rank} performing training")
                    logger.log_start("local sgd")
                    epoch = self.parent.local_sgd(task, parameters, state, base_optimizer, base_optimizer_state, batch_data_gen, (time.time() - start_time))
                    iteration += self.parent.tau
                    logger.log_end("local sgd", {"rank": rank, "rw": group_name, "iteration": self.parent.tau, "epoch": epoch})
                    if rank == 0:
                       with self.parent.lock:
                            for param, rw_pre_param_shared_array, global_pre_param_shared_array in zip(parameters, shared_arrays[rw], shared_arrays[shared_array_index.value]):
                                rw_pre_param = np.frombuffer(rw_pre_param_shared_array.get_obj(), dtype=np.float32).reshape(param.shape)
                                global_pre_param = np.frombuffer(global_pre_param_shared_array.get_obj(), dtype=np.float32).reshape(param.shape)
                                param.data = torch.from_numpy(global_pre_param).to(device) + 1 / len(self.parent.group_names) * (param - torch.from_numpy(rw_pre_param).to(device))

                            for param, shared_array in zip(parameters, shared_arrays[rw]):
                                np.copyto(np.frombuffer(shared_array.get_obj(), dtype=np.float32).reshape(param.shape), param.cpu().detach().numpy())
                            for st, shared_array in zip(state, shared_state):
                                np.copyto(np.frombuffer(shared_array.get_obj(), dtype=np.float32).reshape(st.shape), st.cpu().detach().numpy())

                            shared_array_index.value = rw

                    logging.info(f"[{group_name}] Rank {rank} sending model to Rank {next_rank}")

                if next_rank != rank:
                    logger.log_start("communication")
                    buffer = self.parent.pack(parameters)
                    dist.send(tensor=buffer, dst=next_rank)
                    bytes_sent = self.parent.num_bytes(buffer)
                    logger.log_end("communication", {"rw": group_name, "from": rank, "to": next_rank, "bytes_sent": bytes_sent})

            elif rank == next_rank:
                logging.info(f"[{group_name}] Rank {rank} waiting for model from Rank {current_rank}")
                buffer = torch.zeros_like(self.parent.pack(parameters))
                dist.recv(tensor=buffer, src=current_rank)
                new_params = self.parent.unpack(buffer, shapes)
                for param, new_param in zip(parameters, new_params):
                    param.data = new_param
                logging.info(f"[{group_name}] Rank {rank} received model from Rank {current_rank}")

            # logger.log_start("broadcast")
            # if rank == 0:
            #     current_rank = next_rank
            #     next_rank = torch.tensor(self.lazy_metropolis_hastings_step(current_rank.item(), self.parent.neighbors), dtype=torch.int32).to(device)
            #     logging.info(f"[{group_name}] Rank {current_rank.item()} will send model to Rank {next_rank.item()} for the next iteration")
            #     logging.info(f"[{group_name}] Broadcasting current_rank and next_rank")
            #     dist.broadcast(current_rank, src=0)
            #     dist.broadcast(next_rank, src=0)
            #     logging.info(f"[{group_name}] Finished broadcasting current_rank and next_rank")
            # else:
            #     logging.info(f"[{group_name}] Waiting for broadcast of current_rank and next_rank")
            #     dist.broadcast(current_rank, src=0)
            #     dist.broadcast(next_rank, src=0)
            #     logging.info(f"[{group_name}] Received broadcast of current_rank: {current_rank.item()} and next_rank: {next_rank.item()}")
            #
            # logging.info(f"[{group_name}] Current Rank after broadcast: {current_rank.item()}, Next Rank after broadcast: {next_rank.item()}")
            #
            # if time.time() < end_time:
            #     dist.barrier()
            #     logger.log_end("broadcast", {})
            current_rank = next_rank

        dist.barrier()
        dist.destroy_process_group()
        logging.info(f"[{group_name}] Process {rank} finished")

    def run(self, rank):
        shared_arrays = []
        if rank == 0:
            model = self.parent.create_model()
            for _ in range(len(self.parent.group_names)):
                group_shared_arrays = [Array('f', param.numel(), lock=True) for param in model.parameters()]
                shared_arrays.append(group_shared_arrays)
                for param, shared_array in zip(model.parameters(), group_shared_arrays):
                    np.copyto(np.frombuffer(shared_array.get_obj(), dtype=np.float32).reshape(param.shape), param.cpu().detach().numpy())

            shared_array_index = Value('i', 0)
            shared_state = [Array('f', state.numel(), lock=True) for state in model.buffers()]
            eval_process_active = Value('i', 1)
            eval_process = Process(target=self.parent.evaluation_process, args=(self.parent.eval_gpu, shared_arrays, shared_state, eval_process_active, shared_array_index))
            eval_process.start()

        # gpu_idle = Value('i', 1)
        processes = []
        for rw in range(len(self.parent.group_names)):
            if rank == 0:
                p = Process(target=self.train_node, args=(rank, rw, shared_arrays, shared_array_index, shared_state))
            else:
                p = Process(target=self.train_node, args=(rank, rw,))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        if rank == 0:
            eval_process_active.value = 0
            eval_process.join()


class AsyncGossip:
    def __init__(self, parent):
        self.parent = parent

    def async_gossip_computation(self, rank, shared_arrays, shared_grads, comm_process_started, shared_state=None):
        torch.manual_seed(self.parent.config["seed"] + rank)
        np.random.seed(self.parent.config["seed"] + rank)

        device = torch.device(f'cuda:{self.parent.local_rank}' if torch.cuda.is_available() else 'cpu')

        logger = EventLogger(log_file_name=self.parent.log_name)

        task = self.parent.configure_task(rank, device)
        parameters, state = task.initialize(self.parent.config["seed"])
        base_optimizer = configure_base_optimizer(self.parent.config)
        base_optimizer_state = base_optimizer.init(parameters)

        batch_data_gen = task.data.iterator(batch_size=self.parent.config["batch_size"], shuffle=True)

        while comm_process_started.value == 0: #wait for communication process to start
            logging.info(f"[Gossip Computation] wait at Rank {rank}")
            time.sleep(0.5)

        start_time = time.time()
        end_time = start_time + self.parent.train_time * 60  # Convert minutes to seconds
        iteration = 0

        while time.time() < end_time:
            logging.info(f"[Gossip Computation] Iteration {iteration} at Rank {rank}")
            for param, shared_array, shared_grad in zip(parameters, shared_arrays, shared_grads):
                param_data = np.frombuffer(shared_array.get_obj(), dtype=np.float32).reshape(param.shape)
                grad_data = np.frombuffer(shared_grad.get_obj(), dtype=np.float32).reshape(param.shape)
                param.data = torch.from_numpy(param_data).to(device) + torch.from_numpy(grad_data).to(device)

            initial_params = [param.clone() for param in parameters]

            logger.log_start("local sgd")
            epoch = self.parent.local_sgd(task, parameters, state, base_optimizer, base_optimizer_state, batch_data_gen, (time.time() - start_time))
            logger.log_end("local sgd", {"rank": rank, "iteration": self.parent.tau, "epoch": epoch})
            iteration += self.parent.tau




            while any(np.any(np.frombuffer(shared_grad.get_obj(), dtype=np.float32) != 0) for shared_grad in shared_grads) and time.time() < end_time:
                time.sleep(0.1)

            for param, initial_param, shared_grad in zip(parameters, initial_params, shared_grads):
                grad_diff = param - initial_param
                np.copyto(np.frombuffer(shared_grad.get_obj(), dtype=np.float32).reshape(param.shape), grad_diff.cpu().numpy())
            if rank == 0:
                for st, shared_array in zip(state, shared_state):
                    np.copyto(np.frombuffer(shared_array.get_obj(), dtype=np.float32).reshape(st.shape), st.cpu().detach().numpy())

    def async_gossip_communication_active(self, rank, shared_arrays, shared_grads, comm_process_started):
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        self.parent.init_process(rank, self.parent.size, backend, self.parent.ports[0], self.parent.group_names[0])

        comm_process_started.value = 1

        device = torch.device(f'cuda:{self.parent.local_rank}' if torch.cuda.is_available() else 'cpu')
        model = self.parent.create_model()

        logger = EventLogger(log_file_name=self.parent.log_name)

        for param, shared_array in zip(model.parameters(), shared_arrays):
            param_data = np.frombuffer(shared_array.get_obj(), dtype=np.float32).reshape(param.shape)
            param.data = torch.from_numpy(param_data).to(device)

        shapes = [param.shape for param in model.parameters()]

        while True:
            for param, shared_grad in zip(model.parameters(), shared_grads):
                grad_data = np.frombuffer(shared_grad.get_obj(), dtype=np.float32).reshape(param.shape)
                if not np.all(grad_data == 0):
                    param.data += torch.from_numpy(grad_data).to(device)
                    grad_data.fill(0)

            neighbor = random.choice(self.parent.neighbors[rank])
            logging.info(f"[Gossip Active] Rank {rank} notifying and exchanging model with Rank {neighbor}")

            notification = torch.tensor(rank, dtype=torch.int32).to(device)
            dist.send(tensor=notification, dst=neighbor)

            logger.log_start("communication")
            buffer = self.parent.pack(model.parameters())
            dist.send(tensor=buffer, dst=neighbor)
            bytes_sent = self.parent.num_bytes(buffer)
            logger.log_end("communication",
                           {"from": rank, "to": neighbor, "bytes_sent": bytes_sent})

            recv_buffer = torch.zeros_like(buffer)
            dist.recv(tensor=recv_buffer, src=neighbor)

            model_params = self.parent.unpack(recv_buffer, shapes)
            for param, model_param in zip(model.parameters(), model_params):
                param.data = (param.data + model_param) / 2

            for param, shared_array in zip(model.parameters(), shared_arrays):
                np.copyto(np.frombuffer(shared_array.get_obj(), dtype=np.float32).reshape(param.shape), param.cpu().detach().numpy())

    def async_gossip_communication_passive(self, rank, shared_arrays, shared_grads, comm_process_started):
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        self.parent.init_process(rank, self.parent.size, backend, self.parent.ports[0], self.parent.group_names[0])

        comm_process_started.value = 1

        device = torch.device(f'cuda:{self.parent.local_rank}' if torch.cuda.is_available() else 'cpu')
        model = self.parent.create_model()

        logger = EventLogger(log_file_name=self.parent.log_name)

        for param, shared_array in zip(model.parameters(), shared_arrays):
            param_data = np.frombuffer(shared_array.get_obj(), dtype=np.float32).reshape(param.shape)
            param.data = torch.from_numpy(param_data).to(device)

        shapes = [param.shape for param in model.parameters()]

        logging.info(f"[Gossip Passive] Rank {rank} before waiting for notification")

        reqs = [(dist.irecv(tensor=torch.zeros(1, dtype=torch.int32).to(device), src=neighbor), neighbor) for neighbor in self.parent.neighbors[rank]]

        logging.info(f"[Gossip Passive] Rank {rank} made reqs")
        while True:
            for param, shared_grad in zip(model.parameters(), shared_grads):
                grad_data = np.frombuffer(shared_grad.get_obj(), dtype=np.float32).reshape(param.shape)
                if not np.all(grad_data == 0):
                    param.data += torch.from_numpy(grad_data).to(device)
                    grad_data.fill(0)

            logging.info(f"[Gossip Passive] Rank {rank} waiting for notification")

            source_rank = None
            while True:
                for req, neighbor in reqs:
                    if req.is_completed():
                        source_rank = neighbor
                        reqs.remove((req, neighbor))
                        break
                if source_rank is not None:
                    break
                time.sleep(0.1)

            logging.info(f"[Gossip Passive] Rank {rank} received notification from Rank {source_rank}, waiting for model data")

            buffer = torch.zeros_like(self.parent.pack(model.parameters()))
            dist.recv(tensor=buffer, src=source_rank)

            logger.log_start("communication")
            send_buffer = self.parent.pack(model.parameters())
            dist.send(tensor=send_buffer, dst=source_rank)
            bytes_sent = self.parent.num_bytes(send_buffer)
            logger.log_end("communication",
                           {"from": rank, "to": source_rank, "bytes_sent": bytes_sent})

            model_params = self.parent.unpack(buffer, shapes)
            for param, model_param in zip(model.parameters(), model_params):
                param.data = (param.data + model_param) / 2

            reqs.append((dist.irecv(tensor=torch.zeros(1, dtype=torch.int32).to(device), src=source_rank), source_rank))

            for param, shared_array in zip(model.parameters(), shared_arrays):
                np.copyto(np.frombuffer(shared_array.get_obj(), dtype=np.float32).reshape(param.shape), param.cpu().detach().numpy())

    def run(self, rank):
        model = self.parent.create_model()
        shared_arrays = [Array('f', param.numel(), lock=True) for param in model.parameters()]
        shared_grads = [Array('f', param.numel(), lock=True) for param in model.parameters()]

        for param, shared_array, shared_grad in zip(model.parameters(), shared_arrays, shared_grads):
            np.copyto(np.frombuffer(shared_array.get_obj(), dtype=np.float32).reshape(param.shape), param.cpu().detach().numpy())
            grad_data = np.frombuffer(shared_grad.get_obj(), dtype=np.float32).reshape(param.shape)
            grad_data.fill(0)

        comm_process_started = Value('i', 0)

        if rank in self.parent.top_nodes:
            comm_process = Process(target=self.async_gossip_communication_active, args=(rank, shared_arrays, shared_grads, comm_process_started))
        else:
            comm_process = Process(target=self.async_gossip_communication_passive, args=(rank, shared_arrays, shared_grads, comm_process_started))

        if rank == 0:
            shared_state = [Array('f', state.numel(), lock=True) for state in model.buffers()]
            eval_process_active = Value('i', 1)
            compute_process = Process(target=self.async_gossip_computation, args=(rank, shared_arrays, shared_grads, comm_process_started, shared_state))
            eval_process = Process(target=self.parent.evaluation_process, args=(self.parent.eval_gpu, shared_arrays, shared_state, eval_process_active))
            eval_process.start()
        else:
            compute_process = Process(target=self.async_gossip_computation, args=(rank, shared_arrays, shared_grads, comm_process_started))

        compute_process.start()
        comm_process.start()

        compute_process.join()
        comm_process.terminate()

        if rank == 0:
            eval_process_active.value = 0
            eval_process.join()