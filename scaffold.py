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
from utils.tools import (
    load_from_shared
)
from logger import EventLogger


class Scaffold:
    def __init__(self, parent):
        self.parent = parent
        # SCAFFOLD specific parameters
        self.participation_rate = parent.config.get('participation_rate', 0.1)  # Fraction of clients per round
        self.min_clients = max(1, int(self.participation_rate * (parent.size - 1)))  # At least 1 client
        self.global_learning_rate = parent.config.get('global_learning_rate', 1.0) # Global learning rate for SCAFFOLD

    def select_participating_clients(self, round_number, total_clients):
        """Return the actual ranks of selected participating clients for current round"""
        torch.manual_seed(self.parent.config["seed"] + round_number)
        num_participants = max(self.min_clients, int(self.participation_rate * total_clients))
        # Client ranks are from 1 to total_clients (server is rank 0)
        participant_indices = torch.randperm(total_clients)[:num_participants].tolist()
        participating_client_ranks = [idx + 1 for idx in participant_indices]  # Actual distributed ranks
        return participating_client_ranks

    def send_to_clients(self, participating_clients, global_parameters, global_control_variates, device, logger, round_number, server_rank):
        """Send global model and control variates to all participating clients separately"""
        # Determine which client should send state (first participating client)
        designated_state_sender = participating_clients[0] if participating_clients else None

        # Send notifications to all clients
        notification_requests = []
        for client_rank in range(1, self.parent.size):
            if client_rank not in participating_clients:
                no_training_signal = torch.tensor(0, dtype=torch.int32).to(device)  # 0 = end training
                notification_requests.append(dist.isend(tensor=no_training_signal, dst=client_rank))
                logging.info(f"[SCAFFOLD Server] Round {round_number}, notifications sent to rank {client_rank}!")
            else:
                # Send notification with state sender info: 1 = start training, 2 = start training + send state
                training_signal = 2 if client_rank == designated_state_sender else 1
                training_notification = torch.tensor(training_signal, dtype=torch.int32).to(device)
                notification_requests.append(dist.isend(tensor=training_notification, dst=client_rank))
                logging.info(f"[SCAFFOLD Server] Round {round_number}, notifications sent to rank {client_rank}!")

        for notification_request in notification_requests:
            notification_request.wait()
        logging.info(f"[SCAFFOLD Server] Round {round_number}, all notifications sent!")

        # Send global model to participating clients
        model_send_requests = []
        for client_rank in participating_clients:
            logging.info(f"[SCAFFOLD Server] Round {round_number}, model is to be sent to rank {client_rank}!")
            logger.log_start("communication")
            model_buffer = pack(global_parameters)
            model_send_requests.append(dist.isend(tensor=model_buffer, dst=client_rank))
            bytes_transmitted = num_bytes(model_buffer)
            logger.log_end("communication", {"round": round_number, "from": server_rank, "to": client_rank, "bytes_sent": bytes_transmitted})
            logging.info(f"[SCAFFOLD Server] Round {round_number}, model was sent to rank {client_rank}!")

        for model_request in model_send_requests:
            model_request.wait()

        # Send global control variates to participating clients
        control_variate_send_requests = []
        for client_rank in participating_clients:
            logging.info(f"[SCAFFOLD Server] Round {round_number}, control variates are to be sent to rank {client_rank}!")
            logger.log_start("communication")
            control_variates_buffer = pack(global_control_variates)
            control_variate_send_requests.append(dist.isend(tensor=control_variates_buffer, dst=client_rank))
            bytes_transmitted = num_bytes(control_variates_buffer)
            logger.log_end("communication", {"round": round_number, "from": server_rank, "to": client_rank, "bytes_sent": bytes_transmitted})
            logging.info(f"[SCAFFOLD Server] Round {round_number}, control variates were sent to rank {client_rank}!")

        for request in control_variate_send_requests:
            request.wait()


    def receive_from_clients(self, participating_clients, global_parameters, device):
        """Receive updates (parameter differences and client control variate updates) from participating clients separately"""
        client_updates = {}
        receive_info = []

        # Initiate all non-blocking receives for parameter differences
        param_diff_receive_info = []
        for client_rank in participating_clients:
            param_diff_buffer = torch.zeros_like(pack(global_parameters))
            param_diff_request = dist.irecv(tensor=param_diff_buffer, src=client_rank)
            param_diff_receive_info.append((param_diff_request, param_diff_buffer, client_rank))

        # Wait for all parameter difference receives
        for param_diff_request, param_diff_buffer, client_rank in param_diff_receive_info:
            param_diff_request.wait()
            if client_rank not in client_updates:
                client_updates[client_rank] = {}
            client_updates[client_rank]["parameter_difference"] = unpack(param_diff_buffer, [param.shape for param in global_parameters])


        # Initiate all non-blocking receives for client control variate updates
        client_control_variate_receive_info = []
        for client_rank in participating_clients:
            client_control_variate_update_buffer = torch.zeros_like(pack(global_parameters)) # Control variates have same shape as parameters
            client_control_variate_update_request = dist.irecv(tensor=client_control_variate_update_buffer, src=client_rank)
            client_control_variate_receive_info.append((client_control_variate_update_request, client_control_variate_update_buffer, client_rank))

        # Wait for all client control variate update receives
        for client_control_variate_update_request, client_control_variate_update_buffer, client_rank in client_control_variate_receive_info:
            client_control_variate_update_request.wait()
            if client_rank not in client_updates:
                client_updates[client_rank] = {}
            client_updates[client_rank]["client_control_variate_update"] = unpack(client_control_variate_update_buffer, [param.shape for param in global_parameters])


        return client_updates

    def server_process(self, server_rank, shared_parameter_arrays, shared_state_arrays, shared_control_variate_arrays):
        """SCAFFOLD Server process (rank 0) - handles aggregation and client coordination"""
        torch.manual_seed(self.parent.config["seed"])
        np.random.seed(self.parent.config["seed"])
        communication_device = 'cpu'
        event_logger = EventLogger(log_file_name=self.parent.log_name)

        # Initialize global model
        global_model = self.parent.create_model()
        global_parameters = [param.to(communication_device) for param in global_model.parameters()]
        global_state = [state.to(communication_device) for state in global_model.buffers()]

        # Initialize global control variates (initialized to zeros)
        global_control_variates = [torch.zeros_like(param).to(communication_device) for param in global_parameters]


        # Initialize communication
        communication_backend = 'gloo'  # Use gloo for CPU communication
        self.parent.init_process(server_rank, self.parent.size, communication_backend, self.parent.ports[0], self.parent.group_names[0])

        current_round = 0
        training_start_time = time.time()
        training_end_time = training_start_time + self.parent.train_time * 60

        while time.time() < training_end_time:
            # Select participating clients
            participating_clients = self.select_participating_clients(current_round, self.parent.size-1)
            logging.info(f"[SCAFFOLD Server] Round {current_round}, Participants: {participating_clients}")

            # Send global model and control variates to all participating clients
            self.send_to_clients(participating_clients, global_parameters, global_control_variates, communication_device, event_logger, current_round, server_rank)

            # Receive updates from participating clients
            client_updates = self.receive_from_clients(participating_clients, global_parameters, communication_device)

            # Receive state from only one participating client (the first one)
            if participating_clients:
                designated_state_sender = participating_clients[0]  # Choose the first participating client
                logging.info(f"[SCAFFOLD Server] Round {current_round}, receiving state from client {designated_state_sender}")

                # Receive state from the selected client
                state_buffer = torch.zeros_like(pack(global_state))
                dist.recv(tensor=state_buffer, src=designated_state_sender)
                received_client_state = unpack(state_buffer, [state.shape for state in global_state])

                # Set the global state from the selected client
                for global_state_param, client_state_param in zip(global_state, received_client_state):
                    global_state_param.data = client_state_param.to(communication_device)


            # Aggregate parameter differences and update global parameters in place
            for client_rank, updates in client_updates.items():
                parameter_difference = updates["parameter_difference"]
                for global_param, client_diff in zip(global_parameters, parameter_difference):
                    global_param.data += self.global_learning_rate * (client_diff.to(communication_device) / len(participating_clients))

            # Aggregate client control variate updates and update global control variates in place
            for client_rank, updates in client_updates.items():
                client_control_variate_update = updates["client_control_variate_update"]
                for global_control_variate, client_update in zip(global_control_variates, client_control_variate_update):
                     global_control_variate.data += (client_update.to(communication_device) / len(participating_clients))


            # Update shared arrays
            for param, shared_array in zip(global_parameters, shared_parameter_arrays):
                np.copyto(np.frombuffer(shared_array.get_obj(), dtype=np.float32).reshape(param.shape),
                         param.cpu().detach().numpy())

            for state_param, shared_state_array in zip(global_state, shared_state_arrays):
                np.copyto(np.frombuffer(shared_state_array.get_obj(), dtype=np.float32).reshape(state_param.shape),
                         state_param.cpu().detach().numpy())

            for control_variate, shared_array in zip(global_control_variates, shared_control_variate_arrays):
                 np.copyto(np.frombuffer(shared_array.get_obj(), dtype=np.float32).reshape(control_variate.shape),
                           control_variate.cpu().detach().numpy())

            current_round += 1
            dist.barrier()

        # Signal end to all clients
        notification_requests = []
        for client_rank in range(1, self.parent.size):
            termination_signal = torch.tensor(-10, dtype=torch.int32).to(communication_device)  # -10 = end training
            notification_requests.append(dist.isend(tensor=termination_signal, dst=client_rank))

        for notification_request in notification_requests:
            notification_request.wait()
        dist.barrier()

        dist.destroy_process_group()
        logging.info(f"[SCAFFOLD Server] Finished {current_round} rounds")


    def client_process(self, client_rank,):
        """SCAFFOLD Client process (client_rank > 0) - performs local training with control variates"""
        torch.manual_seed(self.parent.config["seed"] + client_rank)
        np.random.seed(self.parent.config["seed"] + client_rank)
        comm_device = 'cpu'
        training_device = torch.device(f'cuda:{self.parent.local_rank}' if torch.cuda.is_available() else 'cpu')
        event_logger = EventLogger(log_file_name=self.parent.log_name)

        # Initialize task and model
        training_task = self.parent.configure_task(client_rank, training_device)
        parameters, state = training_task.initialize(self.parent.config["seed"])
        base_optimizer = configure_base_optimizer(self.parent.config)
        base_optimizer_state = base_optimizer.init(parameters)

        # Initialize client control variates (initialized to zeros)
        client_control_variates = [torch.zeros_like(param).to(training_device) for param in parameters]


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
            logging.info(f"[SCAFFOLD Client {client_rank}] notification received!")
            if notification.item() == -10:  # End signal
                break
            elif notification.item() == 0:  # End training
                 dist.barrier()
            elif notification.item() in [1,2]:  # Start training

                # Receive global model from server
                logging.info(f"[SCAFFOLD Client {client_rank}] receiveing the model!")
                model_buffer = torch.zeros_like(pack(parameters), device=comm_device)
                dist.recv(tensor=model_buffer, src=0)
                global_params = unpack(model_buffer, [p.shape for p in parameters])
                logging.info(f"[SCAFFOLD Client {client_rank}] received the model!")

                # Receive global control variates from server
                logging.info(f"[SCAFFOLD Client {client_rank}] receiveing the control variates!")
                control_variates_buffer = torch.zeros_like(pack(client_control_variates), device=comm_device) # Control variates have same shape as parameters
                dist.recv(tensor=control_variates_buffer, src=0)
                global_control_variates = unpack(control_variates_buffer, [cv.shape for cv in client_control_variates])
                logging.info(f"[SCAFFOLD Client {client_rank}] received the control variates!")

                
                # Update local parameters with global model
                for local_param, global_param in zip(parameters, global_params):
                    local_param.data = global_param.to(training_device)


                # Perform local SGD with control variates
                event_logger.log_start("local sgd")
                # Corrected the start_time variable name
                for step in range(self.parent.tau):
                    time_for_lr_schedule = time.time() - training_start_time
                    epoch = self.parent.local_sgd(training_task, parameters, state, base_optimizer, base_optimizer_state, batch_data_gen, time_for_lr_schedule, 1)
                    local_lr_reference = self.parent.config["learning_rate"] * self.parent.learning_rate_schedule(time_for_lr_schedule)
                    for param, local_c, global_c  in zip(parameters, client_control_variates, global_control_variates):
                        param += - local_lr_reference * (global_c - local_c)
                event_logger.log_end("local sgd", {"rank": client_rank, "iteration": self.parent.tau, "epoch": epoch})

                # Calculate parameter difference
                parameter_difference = [param.to(comm_device) - global_param for param, global_param in zip(parameters, global_params)]

                # Calculate client control variate update
                client_control_variate_updates = [- global_c.to(comm_device) - param_difference / self.parent.tau / local_lr_reference for global_c, param_difference in zip(global_control_variates, parameter_difference)]

                client_control_variates = [client_control_variate + client_control_variate_update.to(training_device) for client_control_variate, client_control_variate_update in zip(client_control_variates, client_control_variate_updates)]

                # Send parameter difference to server
                event_logger.log_start("communication")
                param_diff_buffer = pack(parameter_difference)
                dist.send(tensor=param_diff_buffer, dst=0)
                bytes_sent = num_bytes(param_diff_buffer)
                event_logger.log_end("communication", {"from": client_rank, "to": 0, "bytes_sent": bytes_sent})


                # Send client control variate update to server
                event_logger.log_start("communication")
                client_control_variate_update_buffer = pack(client_control_variate_updates)
                dist.send(tensor=client_control_variate_update_buffer, dst=0)
                bytes_sent = num_bytes(client_control_variate_update_buffer)
                event_logger.log_end("communication", {"from": client_rank, "to": 0, "bytes_sent": bytes_sent})


                if notification.item() == 2:  # Start training + send state
                    # Send state to server (only this client sends state)
                    logging.info(f"[SCAFFOLD Client {client_rank}] sending state to server!")
                    buffer = pack(state).to(comm_device)
                    dist.send(tensor=buffer, dst=0)
                    bytes_sent = num_bytes(buffer)
                    logging.info(f"[SCAFFOLD Client {client_rank}] state sent to server!")

                dist.barrier()

        dist.barrier()
        dist.destroy_process_group()
        logging.info(f"[SCAFFOLD Client {client_rank}] Finished")


    def run(self, rank):
        """Main SCAFFOLD execution"""
        model = self.parent.create_model()
        if rank == 0:
            shared_arrays = [Array('f', param.numel(), lock=True) for param in model.parameters()]
            shared_state = [Array('f', state.numel(), lock=True) for state in model.buffers()]
            shared_control_variate_arrays = [Array('f', param.numel(), lock=True) for param in model.parameters()] # Shared arrays for control variates

            # Initialize shared arrays for parameters and state (control variates are initialized to zeros)
            for param, shared_array in zip(model.parameters(), shared_arrays):
                np.copyto(np.frombuffer(shared_array.get_obj(), dtype=np.float32).reshape(param.shape),
                        param.cpu().detach().numpy())

            eval_process_active = Value('i', 1)
            # Server process
            eval_process = Process(target=self.parent.evaluation_process,
                                 args=(self.parent.eval_gpu, shared_arrays, shared_state, eval_process_active, None))
            eval_process.start()

            # Corrected variable name from server_process to server_process_func to avoid shadowing
            server_process_func = Process(target=self.server_process, args=(rank, shared_arrays, shared_state, shared_control_variate_arrays))
            server_process_func.start()
            server_process_func.join()
            eval_process_active.value = 0
            eval_process.join()
        else:
            # Client process
            self.client_process(rank,)