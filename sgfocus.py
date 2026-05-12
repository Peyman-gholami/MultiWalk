import torch
import torch.distributed as dist
import random
import logging
import numpy as np
from torch.multiprocessing import Process, Array, Value, Lock
import time
from tasks.api import Task
from base_optimizers import configure_base_optimizer
from utils.communication import (
    pack,
    unpack,
    num_bytes
)
from logger import EventLogger


class SGFocus:
    def __init__(self, parent):
        self.parent = parent
        # SG-FOCUS specific parameters
        self.participation_rate = parent.config.get('participation_rate', 1)
        self.global_learning_rate = parent.config.get('global_learning_rate', 1.0) # Global learning rate

    def select_participating_clients(self, round_number, total_clients):
        """Return the actual ranks of selected participating clients for current round"""
        torch.manual_seed(self.parent.config["seed"] + round_number)
        num_participants = max(1, int(self.participation_rate * total_clients)) # At least 1 client
        # Client ranks are from 1 to total_clients (server is rank 0)
        participant_indices = torch.randperm(total_clients)[:num_participants].tolist()
        participating_client_ranks = [idx + 1 for idx in participant_indices]  # Actual distributed ranks
        return participating_client_ranks

    def send_to_clients(self, participating_clients, global_parameters, device, logger, round_number, server_rank):
        """Send global model to all participating clients separately"""
        # Send notifications to all clients
        notification_requests = []
        for client_rank in range(1, self.parent.size):
            if client_rank not in participating_clients:
                no_training_signal = torch.tensor(0, dtype=torch.int32).to(device)  # 0 = end training
                notification_requests.append(dist.isend(tensor=no_training_signal, dst=client_rank))
                logging.info(f"[SG-FOCUS Server] Round {round_number}, notifications sent to rank {client_rank}!")
            else:
                training_notification = torch.tensor(1, dtype=torch.int32).to(device)
                notification_requests.append(dist.isend(tensor=training_notification, dst=client_rank))
                logging.info(f"[SG-FOCUS Server] Round {round_number}, notifications sent to rank {client_rank}!")

        for notification_request in notification_requests:
            notification_request.wait()
        logging.info(f"[SG-FOCUS Server] Round {round_number}, all notifications sent!")

        # Send global model to participating clients
        model_send_requests = []
        for client_rank in participating_clients:
            logging.info(f"[SG-FOCUS Server] Round {round_number}, model is to be sent to rank {client_rank}!")
            logger.log_start("communication")
            model_buffer = pack(global_parameters)
            model_send_requests.append(dist.isend(tensor=model_buffer, dst=client_rank))
            bytes_transmitted = num_bytes(model_buffer)
            logger.log_end("communication", {"round": round_number, "from": server_rank, "to": client_rank, "bytes_sent": bytes_transmitted})
            logging.info(f"[SG-FOCUS Server] Round {round_number}, model was sent to rank {client_rank}!")

        for model_request in model_send_requests:
            model_request.wait()

    def receive_from_clients(self, participating_clients, global_parameters):
        """Receive gradient updates from participating clients separately"""
        client_updates = {}
        
        # Initiate all non-blocking receives for gradient updates
        grad_receive_info = []
        for client_rank in participating_clients:
            grad_buffer = torch.zeros_like(pack(global_parameters))
            grad_request = dist.irecv(tensor=grad_buffer, src=client_rank)
            grad_receive_info.append((grad_request, grad_buffer, client_rank))

        # Wait for all gradient update receives
        for grad_request, grad_buffer, client_rank in grad_receive_info:
            grad_request.wait()
            if client_rank not in client_updates:
                client_updates[client_rank] = {}
            client_updates[client_rank]["gradient_update"] = unpack(grad_buffer, [param.shape for param in global_parameters])

        return client_updates

    def server_process(self, server_rank, shared_parameter_arrays):
        """SG-FOCUS Server process (rank 0) - handles aggregation and client coordination"""
        torch.manual_seed(self.parent.config["seed"])
        np.random.seed(self.parent.config["seed"])
        communication_device = 'cpu'
        event_logger = EventLogger(log_file_name=self.parent.log_name)

        # Initialize global model
        global_model = self.parent.create_model()
        global_parameters = [param.to(communication_device) for param in global_model.parameters()]

        # Initialize global aggregated gradient (y_r)
        global_aggregated_grad = [torch.zeros_like(param).to(communication_device) for param in global_parameters]

        # Initialize communication
        communication_backend = 'gloo'
        self.parent.init_process(server_rank, self.parent.size, communication_backend, self.parent.ports[0], self.parent.group_names[0])

        current_round = 0
        training_start_time = time.time()
        training_end_time = training_start_time + self.parent.train_time * 60

        while time.time() < training_end_time:
            # Select participating clients
            participating_clients = self.select_participating_clients(current_round, self.parent.size-1)
            logging.info(f"[SG-FOCUS Server] Round {current_round}, Participants: {participating_clients}")

            # Send global model to all participating clients
            self.send_to_clients(participating_clients, global_parameters, communication_device, event_logger, current_round, server_rank)

            # Receive updates from participating clients
            client_updates = self.receive_from_clients(participating_clients, global_parameters)

            # Aggregate gradient updates and update global aggregated gradient
            # y_{r+1} = y_r + sum_{i in S_r}(gradient_difference_i)
            for client_rank, updates in client_updates.items():
                gradient_difference = updates["gradient_update"]
                for global_grad, grad_diff in zip(global_aggregated_grad, gradient_difference):
                    global_grad.data += grad_diff.to(communication_device)

            # Update global parameters
            # x_{r+1} = x_r - eta * y_{r+1}
            time_for_lr_schedule = time.time() - training_start_time
            local_lr_reference = self.parent.config["learning_rate"] * self.parent.learning_rate_schedule(time_for_lr_schedule)
                    
            for global_param, global_grad in zip(global_parameters, global_aggregated_grad):
                global_param.data -= local_lr_reference * global_grad

            # Update shared arrays
            for param, shared_array in zip(global_parameters, shared_parameter_arrays):
                np.copyto(np.frombuffer(shared_array.get_obj(), dtype=np.float32).reshape(param.shape),
                         param.cpu().detach().numpy())

            current_round += 1
            dist.barrier()

        # Signal end to all clients
        notification_requests = []
        for client_rank in range(1, self.parent.size):
            termination_signal = torch.tensor(-10, dtype=torch.int32).to(communication_device)
            notification_requests.append(dist.isend(tensor=termination_signal, dst=client_rank))

        for notification_request in notification_requests:
            notification_request.wait()
        dist.barrier()

        dist.destroy_process_group()
        logging.info(f"[SG-FOCUS Server] Finished {current_round} rounds")


    def client_process(self, client_rank,):
        """SG-FOCUS Client process (client_rank > 0) - performs local training"""
        torch.manual_seed(self.parent.config["seed"] + client_rank)
        np.random.seed(self.parent.config["seed"] + client_rank)
        comm_device = 'cpu'
        training_device = torch.device(f'cuda:{self.parent.local_rank}' if torch.cuda.is_available() else 'cpu')
        event_logger = EventLogger(log_file_name=self.parent.log_name)

        # Initialize task and model
        training_task = self.parent.configure_task(client_rank, training_device)
        parameters, state = training_task.initialize(self.parent.config["seed"])
        # SG-FOCUS local step configuration: plain SGD behavior
        # (momentum = 0, weight_decay = 0 for this algorithm).
        sgfocus_optimizer_config = dict(self.parent.config)
        sgfocus_optimizer_config["momentum"] = 0.0
        sgfocus_optimizer_config["weight_decay"] = 0.0
        base_optimizer = configure_base_optimizer(sgfocus_optimizer_config)
        base_optimizer_state = base_optimizer.init(parameters)
        if hasattr(training_task, "_weight_decay_per_param"):
            training_task._weight_decay_per_param = [0.0 for _ in training_task._weight_decay_per_param]
        # Initialize previous stochastic gradient for local updates
        prev_stochastic_grad = [torch.zeros_like(param).to(training_device) for param in parameters]
        # # Initialize y
        # y_i = [torch.zeros_like(param).to(training_device) for param in parameters]


        batch_data_gen = training_task.data.iterator(
            batch_size=self.parent.config["batch_size"],
            shuffle=True,
        )

        # Initialize communication
        communication_backend = 'gloo'
        self.parent.init_process(client_rank, self.parent.size, communication_backend, self.parent.ports[0], self.parent.group_names[0])

        training_start_time = time.time()

        while True:
            # Wait for server notification
            notification = torch.tensor(-1, dtype=torch.int32).to(comm_device)
            dist.recv(tensor=notification, src=0)
            logging.info(f"[SG-FOCUS Client {client_rank}] notification received!")
            if notification.item() == -10:
                break
            elif notification.item() == 0:
                 dist.barrier()
            elif notification.item() == 1:

                # Receive global model from server
                logging.info(f"[SG-FOCUS Client {client_rank}] receiving the model!")
                model_buffer = torch.zeros_like(pack(parameters), device=comm_device)
                dist.recv(tensor=model_buffer, src=0)
                global_params = unpack(model_buffer, [p.shape for p in parameters])
                logging.info(f"[SG-FOCUS Client {client_rank}] received the model!")
                
                global_params = [global_param.to(training_device) for global_param in global_params]
                
                # Update local parameters with global model
                for local_param, global_param in zip(parameters, global_params):
                    local_param.data = global_param.clone()

                # y_{0,i}^{(r)} <- 0 each round (Alg. 2). The subtracted "previous" stochastic gradient
                # ∇f_i(x_{t-1,i}^{(r)}; ξ_{t-1}) is stored in prev_stochastic_grad: for t=0 it is the
                # gradient from the end of the worker's last participating round (paper text after (22)),
                # so we do not reset prev_stochastic_grad here.

                # Initialize y_0,i = 0 for the current round
                y_i = [torch.zeros_like(p).to(training_device) for p in parameters]

                event_logger.log_start("local sgd")
                
                for step in range(self.parent.tau):
                    time_for_lr_schedule = time.time() - training_start_time
                    local_lr = self.parent.config["learning_rate"] * self.parent.learning_rate_schedule(time_for_lr_schedule)
                    # Use the shared local step method for gradient computation + parameter update.
                    # Base optimizer is configured above with momentum=0 and weight_decay=0.
                    epoch, current_stochastic_grad = self.parent.local_sgd(
                        training_task,
                        parameters,
                        state,
                        base_optimizer,
                        base_optimizer_state,
                        batch_data_gen,
                        time_for_lr_schedule,
                        1,
                    )
                    prev_y_i = [y.clone() for y in y_i]
                    # y_{t+1,i} = y_{t,i} + current_grad - prev_grad
                    for y, current_grad, prev_grad in zip(y_i, current_stochastic_grad, prev_stochastic_grad):
                        y.data += current_grad - prev_grad

                    # local_sgd already applied: x <- x - eta * current_grad.
                    # Add correction to recover SG-FOCUS update x <- x - eta * y_{t+1}:
                    # correction = +eta * prev_grad - eta * y_t.
                    for param, prev_grad, prev_y in zip(parameters, prev_stochastic_grad, prev_y_i):
                        param.data += local_lr * prev_grad
                        param.data -= local_lr * prev_y

                    # Update previous gradient for the next local step
                    for prev_grad, current_grad in zip(prev_stochastic_grad, current_stochastic_grad):
                        prev_grad.data = current_grad.clone()
                
                event_logger.log_end("local sgd", {"rank": client_rank, "iteration": self.parent.tau, "epoch": epoch})
                
                
                # Send the gradient difference to the server
                event_logger.log_start("communication")
                grad_update_buffer = pack(y_i).to(comm_device)
                dist.send(tensor=grad_update_buffer, dst=0)
                bytes_sent = num_bytes(grad_update_buffer)
                event_logger.log_end("communication", {"from": client_rank, "to": 0, "bytes_sent": bytes_sent})
                

                dist.barrier()

        dist.barrier()
        dist.destroy_process_group()
        logging.info(f"[SG-FOCUS Client {client_rank}] Finished")

    def run(self, rank):
        """Main SG-FOCUS execution"""
        model = self.parent.create_model()
        if rank == 0:
            shared_arrays = [Array('f', param.numel(), lock=True) for param in model.parameters()]

            for param, shared_array in zip(model.parameters(), shared_arrays):
                np.copyto(np.frombuffer(shared_array.get_obj(), dtype=np.float32).reshape(param.shape),
                        param.cpu().detach().numpy())

            eval_process_active = Value('i', 1)
            eval_process = Process(target=self.parent.evaluation_process,
                                 args=(self.parent.eval_gpu, shared_arrays, None, eval_process_active, None))
            eval_process.start()

            server_process_func = Process(target=self.server_process, args=(rank, shared_arrays))
            server_process_func.start()
            server_process_func.join()
            eval_process_active.value = 0
            eval_process.join()
        else:
            self.client_process(rank,)

