import torch
import torch.distributed as dist
import random
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


class AsyncGossip:
    def __init__(self, parent):
        self.parent = parent

    def async_gossip_computation(self, rank, shared_arrays, shared_grads, comm_process_started, shared_state=None, eval_process_active=None):
        torch.manual_seed(self.parent.config["seed"] + rank)
        np.random.seed(self.parent.config["seed"] + rank)

        device = torch.device(f'cuda:{self.parent.local_rank}' if torch.cuda.is_available() else 'cpu')

        logger = EventLogger(log_file_name=self.parent.log_name)

        task = self.parent.configure_task(rank, device)
        parameters, state = task.initialize(self.parent.config["seed"])
        base_optimizer = configure_base_optimizer(self.parent.config)
        base_optimizer_state = base_optimizer.init(parameters)

        batch_data_gen = task.data.iterator(batch_size=self.parent.config["batch_size"], shuffle=True)

        while comm_process_started.value != 2: #wait for communication process to start
            logging.info(f"[Gossip Computation] wait at Rank {rank}")
            time.sleep(0.5)

        start_time = time.time()
        end_time = start_time + self.parent.train_time * 60  # Convert minutes to seconds
        iteration = 0

        failure_times = sorted(self.parent.failure_times)
        failure = rank < len(failure_times)
        stop_eval_time = start_time + failure_times[rank] * 60 if failure else None
        start_eval_time = start_time if rank == 0 else (
            start_time + failure_times[rank - 1] * 60 if rank - 1 < len(failure_times) else None
        )

        while time.time() < end_time:
            current_time = time.time()
            current_minutes = (current_time - start_time) / 60.0

            # Activate evaluation when previous rank fails
            if eval_process_active is not None and start_eval_time is not None and eval_process_active.value == 2 and current_time >= start_eval_time:
                logging.info(f"[Gossip Computation] Activating eval_process on Rank {rank} at {current_minutes:.2f} minutes (prev failure)")
                eval_process_active.value = 1

            # Deactivate evaluation at this rank's failure time
            rank_failed = stop_eval_time is not None and current_time >= stop_eval_time
            if eval_process_active is not None and rank_failed and eval_process_active.value == 1:
                logging.info(f"[Gossip Computation] Deactivating eval_process on Rank {rank} at {current_minutes:.2f} minutes (rank failure)")
                eval_process_active.value = 0

            logging.info(f"[Gossip Computation] Iteration {iteration} at Rank {rank}")
            for param, shared_array, shared_grad in zip(parameters, shared_arrays, shared_grads):
                param_data = np.frombuffer(shared_array.get_obj(), dtype=np.float32).reshape(param.shape)
                grad_data = np.frombuffer(shared_grad.get_obj(), dtype=np.float32).reshape(param.shape)
                param.data = torch.from_numpy(param_data).to(device) + torch.from_numpy(grad_data).to(device)

            initial_params = [param.clone() for param in parameters]

            if not rank_failed:
                logger.log_start("local sgd")
                epoch, gradients = self.parent.local_sgd(task, parameters, state, base_optimizer, base_optimizer_state, batch_data_gen, (time.time() - start_time), self.parent.tau)
                logger.log_end("local sgd", {"rank": rank, "iteration": self.parent.tau, "epoch": epoch})
                iteration += self.parent.tau

                while any(np.any(np.frombuffer(shared_grad.get_obj(), dtype=np.float32) != 0) for shared_grad in
                            shared_grads) and time.time() < end_time and (stop_eval_time is None or time.time() < stop_eval_time):
                    time.sleep(0.01)

                for param, initial_param, shared_grad in zip(parameters, initial_params, shared_grads):
                    grad_diff = param - initial_param
                    np.copyto(np.frombuffer(shared_grad.get_obj(), dtype=np.float32).reshape(param.shape), grad_diff.cpu().numpy())
                if shared_state is not None:
                    for st, shared_array in zip(state, shared_state):
                        np.copyto(np.frombuffer(shared_array.get_obj(), dtype=np.float32).reshape(st.shape), st.cpu().detach().numpy())
            else:
                time.sleep(10)

        # Signal evaluation process to terminate
        if eval_process_active is not None:
            eval_process_active.value = 0

    def async_gossip_communication_active_send(self, rank, shared_arrays, shared_grads, comm_process_started, activate_recv, recv_arrays):
        backend = 'gloo'#nccl' if torch.cuda.is_available() else 'gloo'
        self.parent.init_process(rank, self.parent.size, backend, self.parent.ports[0], self.parent.group_names[0])
        dist.barrier()
        logging.info(f"[Gossip Active] Rank {rank} barrier passed")
        with self.parent.lock:
            comm_process_started.value += 1

        device = 'cpu'#torch.device(f'cuda:{self.parent.local_rank}' if torch.cuda.is_available() else 'cpu')
        model = self.parent.create_model()

        logger = EventLogger(log_file_name=self.parent.log_name)

        for param, shared_array in zip(model.parameters(), shared_arrays):
            param_data = np.frombuffer(shared_array.get_obj(), dtype=np.float32).reshape(param.shape)
            param.data = torch.from_numpy(param_data).to(device)

        shapes = [param.shape for param in model.parameters()]
        
        # Initialize active neighbors list and track failures
        active_neighbors = list(self.parent.neighbors[rank])
        start_time = time.time()
        failure_times = sorted(self.parent.failure_times)
        rank_failed = rank < len(failure_times)
        stop_time = start_time + failure_times[rank] * 60 if rank_failed else None
        
        while True:
            current_time = time.time()
            current_minutes = (current_time - start_time) / 60.0
            
            # Check if this node has failed - if so, stop sending
            if stop_time is not None and current_time >= stop_time:
                logging.info(f"[Gossip Active] Rank {rank} has failed at {current_minutes:.2f} minutes, stopping communication")
                time.sleep(10)
                continue
            
            # Remove failed neighbors from the active list
            for i, neighbor in enumerate(active_neighbors[:]):  # Use slice copy to iterate safely
                if neighbor < len(failure_times):
                    neighbor_failure_time = failure_times[neighbor] * 60
                    if current_time >= start_time + neighbor_failure_time:
                        active_neighbors.remove(neighbor)
                        logging.info(f"[Gossip Active] Rank {rank} removed failed neighbor {neighbor} from active list at {current_minutes:.2f} minutes")
            
            # If no active neighbors, skip communication
            if not active_neighbors:
                logging.info(f"[Gossip Active] Rank {rank} has no active neighbors, skipping communication")
                time.sleep(10)  # Small sleep to avoid busy waiting
                continue
            
            for param, shared_grad in zip(model.parameters(), shared_grads):
                grad_data = np.frombuffer(shared_grad.get_obj(), dtype=np.float32).reshape(param.shape)
                if not np.all(grad_data == 0):
                    param.data += torch.from_numpy(grad_data).to(device)
                    grad_data.fill(0)

            neighbor = random.choice(active_neighbors)
            logging.info(f"[Gossip Active] Rank {rank} notifying and sending model with Rank {neighbor}")

            notification = torch.tensor(rank, dtype=torch.int32).to(device)
            dist.send(tensor=notification, dst=neighbor)

            activate_recv.value = neighbor

            logger.log_start("communication")
            buffer = pack(model.parameters())
            dist.send(tensor=buffer, dst=neighbor)
            bytes_sent = num_bytes(buffer)
            logger.log_end("communication",
                           {"from": rank, "to": neighbor, "bytes_sent": bytes_sent})

            while activate_recv.value != -1:
                time.sleep(0.01)

            for param, recv_array in zip(model.parameters(), recv_arrays):
                recv_param_data = np.frombuffer(recv_array.get_obj(), dtype=np.float32).reshape(param.shape)
                param.data = (param.data + recv_param_data) / 2


            for param, shared_array in zip(model.parameters(), shared_arrays):
                np.copyto(np.frombuffer(shared_array.get_obj(), dtype=np.float32).reshape(param.shape), param.cpu().detach().numpy())

    def async_gossip_communication_active_recv(self, rank, comm_process_started, activate_recv, recv_arrays):
        backend = 'gloo'#nccl' if torch.cuda.is_available() else 'gloo'
        self.parent.init_process(rank, self.parent.size, backend, self.parent.ports[1], self.parent.group_names[1])
        dist.barrier()
        logging.info(f"[Gossip Passive] Rank {rank} barrier passed")
        with self.parent.lock:
            comm_process_started.value += 1

        device = 'cpu'#torch.device(f'cuda:{self.parent.local_rank}' if torch.cuda.is_available() else 'cpu')
        model = self.parent.create_model()

        shapes = [param.shape for param in model.parameters()]

        while True:

            while activate_recv.value == -1:
                time.sleep(0.01)

            neighbor = activate_recv.value
            logging.info(f"[Gossip Active] Rank {rank} receiving model with Rank {neighbor}")

            buffer = pack(model.parameters())
            recv_buffer = torch.zeros_like(buffer)
            dist.recv(tensor=recv_buffer, src=neighbor)

            model_params = unpack(recv_buffer, shapes)

            for param, recv_array in zip(model_params, recv_arrays):
                np.copyto(np.frombuffer(recv_array.get_obj(), dtype=np.float32).reshape(param.shape), param.cpu().detach().numpy())

            activate_recv.value = -1

    def async_gossip_communication_passive_recv(self, rank, shared_arrays, shared_grads, comm_process_started, activate_recv):
        backend = 'gloo'#'nccl' if torch.cuda.is_available() else 'gloo'
        self.parent.init_process(rank, self.parent.size, backend, self.parent.ports[0], self.parent.group_names[0])
        dist.barrier()
        logging.info(f"[Gossip Passive] Rank {rank} barrier passed")
        with self.parent.lock:
            comm_process_started.value += 1

        device = 'cpu'#torch.device(f'cuda:{self.parent.local_rank}' if torch.cuda.is_available() else 'cpu')
        model = self.parent.create_model()

        logger = EventLogger(log_file_name=self.parent.log_name)

        for param, shared_array in zip(model.parameters(), shared_arrays):
            param_data = np.frombuffer(shared_array.get_obj(), dtype=np.float32).reshape(param.shape)
            param.data = torch.from_numpy(param_data).to(device)

        shapes = [param.shape for param in model.parameters()]

        logging.info(f"[Gossip Passive] Rank {rank} before waiting for notification")

        # Step 1: Create a list of tensors initialized with -1
        tensors = { neighbor : torch.full((1,), -1, dtype=torch.int32).to(device) for neighbor in self.parent.neighbors[rank]}

        # Step 2: Assign each tensor to the reqs
        reqs = [(dist.irecv(tensor=tensors[neighbor], src=neighbor), neighbor) for neighbor in self.parent.neighbors[rank]]

        # reqs = [(dist.irecv(tensor=torch.full((1,), -1, dtype=torch.int32).to(device), src=neighbor), neighbor) for neighbor in self.parent.neighbors[rank]]

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
                    if tensors[neighbor].item() != -1:
                        source_rank = neighbor
                        reqs.remove((req, neighbor))
                        break
                if source_rank is not None:
                    break
                time.sleep(0.01)

            logging.info(f"[Gossip Passive] Rank {rank} received notification from Rank {source_rank}, waiting for model data")

            activate_recv.value = source_rank

            buffer = torch.zeros_like(pack(model.parameters()))
            dist.recv(tensor=buffer, src=source_rank)

            model_params = unpack(buffer, shapes)
            for param, model_param in zip(model.parameters(), model_params):
                param.data = (param.data + model_param) / 2

            for param, shared_array in zip(model.parameters(), shared_arrays):
                np.copyto(np.frombuffer(shared_array.get_obj(), dtype=np.float32).reshape(param.shape), param.cpu().detach().numpy())

            tensors[source_rank].fill_(-1)
            reqs.append((dist.irecv(tensor=tensors[source_rank], src=source_rank), source_rank))

            while activate_recv.value != -1:
                time.sleep(0.01)

    def async_gossip_communication_passive_send(self, rank, shared_arrays, shared_grads, comm_process_started, activate_recv):
        backend = 'gloo'  # 'nccl' if torch.cuda.is_available() else 'gloo'
        self.parent.init_process(rank, self.parent.size, backend, self.parent.ports[1], self.parent.group_names[1])
        dist.barrier()
        logging.info(f"[Gossip Passive] Rank {rank} barrier passed")
        with self.parent.lock:
            comm_process_started.value += 1

        device = 'cpu'  # torch.device(f'cuda:{self.parent.local_rank}' if torch.cuda.is_available() else 'cpu')
        model = self.parent.create_model()

        logger = EventLogger(log_file_name=self.parent.log_name)

        shapes = [param.shape for param in model.parameters()]

        while True:

            while activate_recv.value == -1:
                time.sleep(0.01)

            for param,shared_array, shared_grad in zip(model.parameters(), shared_arrays, shared_grads):
                grad_data = np.frombuffer(shared_grad.get_obj(), dtype=np.float32).reshape(param.shape)
                model_data = np.frombuffer(shared_array.get_obj(), dtype=np.float32).reshape(param.shape)
                param.data = torch.from_numpy(model_data).to(device) + torch.from_numpy(grad_data).to(device)

            neighbor = activate_recv.value
            logging.info(f"[Gossip Active] Rank {rank} sending model with Rank {neighbor}")


            logger.log_start("communication")
            send_buffer = pack(model.parameters())
            dist.send(tensor=send_buffer, dst=neighbor)
            bytes_sent = num_bytes(send_buffer)
            logger.log_end("communication",
                           {"from": rank, "to": neighbor, "bytes_sent": bytes_sent})

            activate_recv.value = -1



    def run(self, rank):
        model = self.parent.create_model()
        shared_arrays = [Array('f', param.numel(), lock=True) for param in model.parameters()]
        shared_grads = [Array('f', param.numel(), lock=True) for param in model.parameters()]

        for param, shared_array, shared_grad in zip(model.parameters(), shared_arrays, shared_grads):
            np.copyto(np.frombuffer(shared_array.get_obj(), dtype=np.float32).reshape(param.shape), param.cpu().detach().numpy())
            grad_data = np.frombuffer(shared_grad.get_obj(), dtype=np.float32).reshape(param.shape)
            grad_data.fill(0)

        comm_process_started = Value('i', 0)
        activate_recv = Value('i', -1)

        if rank in self.parent.top_nodes:
            recv_arrays = [Array('f', param.numel(), lock=True) for param in model.parameters()]
            send_comm_process = Process(target=self.async_gossip_communication_active_send, args=(rank, shared_arrays, shared_grads, comm_process_started, activate_recv, recv_arrays))
            recv_comm_process = Process(target=self.async_gossip_communication_active_recv, args=(
            rank, comm_process_started, activate_recv, recv_arrays))
        else:
            send_comm_process = Process(target=self.async_gossip_communication_passive_send, args=(rank, shared_arrays, shared_grads,comm_process_started, activate_recv))
            recv_comm_process = Process(target=self.async_gossip_communication_passive_recv,
                                        args=(rank, shared_arrays, shared_grads, comm_process_started, activate_recv))

        # Allow evaluation on any rank up to the number of potential failures (inclusive)
        max_potential_failures = min(self.parent.size, len(self.parent.failure_times) + 1)
        is_potential_eval_rank = rank < max_potential_failures

        shared_state = None
        eval_process = None
        eval_process_active = None

        if is_potential_eval_rank:
            shared_state = [Array('f', state.numel(), lock=True) for state in model.buffers()]
            eval_process_active = Value('i', 1 if rank == 0 else 2)
            compute_process = Process(target=self.async_gossip_computation, args=(rank, shared_arrays, shared_grads, comm_process_started, shared_state, eval_process_active))
            eval_process = Process(target=self.parent.evaluation_process, args=(self.parent.eval_gpu, shared_arrays, shared_state, eval_process_active))
            eval_process.start()
        else:
            compute_process = Process(target=self.async_gossip_computation, args=(rank, shared_arrays, shared_grads, comm_process_started))

        compute_process.start()
        send_comm_process.start()
        recv_comm_process.start()

        compute_process.join()
        send_comm_process.terminate()
        recv_comm_process.terminate()

        if eval_process is not None:
            eval_process_active.value = 0
            eval_process.join()
