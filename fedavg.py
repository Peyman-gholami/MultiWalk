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



class FedAVG:
    def __init__(self, parent):
        self.parent = parent
        # FedAVG specific parameters
        self.participation_rate = parent.config.get('participation_rate', 0.1)  # Fraction of clients per round
        self.min_clients = max(1, int(self.participation_rate * (parent.size - 1)))  # At least 1 client
        
    def select_participating_clients(self, round_number, total_clients):
        """Return the actual ranks of selected participating clients for current round"""
        torch.manual_seed(self.parent.config["seed"] + round_number)
        num_participants = max(self.min_clients, int(self.participation_rate * total_clients))
        # Client ranks are from 1 to total_clients (server is rank 0)
        participant_indices = torch.randperm(total_clients)[:num_participants].tolist()
        participating_client_ranks = [idx + 1 for idx in participant_indices]  # Actual distributed ranks
        return participating_client_ranks
    
    def send_to_clients(self, participating_clients, global_parameters, device, logger, round_number, server_rank):
        """Send global model to all participating clients"""
        # Determine which client should send state (first participating client)
        designated_state_sender = participating_clients[0] if participating_clients else None
        
        # Send notifications to all clients
        notification_requests = []
        for client_rank in range(1, self.parent.size):
            if client_rank not in participating_clients:
                no_training_signal = torch.tensor(0, dtype=torch.int32).to(device)  # 0 = end training
                notification_requests.append(dist.isend(tensor=no_training_signal, dst=client_rank))
                logging.info(f"[FedAVG Server] Round {round_number}, notifications sent to rank {client_rank}!")
            else:
                # Send notification with state sender info: 1 = start training, 2 = start training + send state
                training_signal = 2 if client_rank == designated_state_sender else 1
                training_notification = torch.tensor(training_signal, dtype=torch.int32).to(device)
                notification_requests.append(dist.isend(tensor=training_notification, dst=client_rank))
                logging.info(f"[FedAVG Server] Round {round_number}, notifications sent to rank {client_rank}!")
        
        for notification_request in notification_requests:
            notification_request.wait()
        logging.info(f"[FedAVG Server] Round {round_number}, all notifications sent!")
        
        # Send global model to participating clients
        model_send_requests = []
        for client_rank in participating_clients:
            logging.info(f"[FedAVG Server] Round {round_number}, model is to be sent to rank {client_rank}!")
            logger.log_start("communication")
            model_buffer = pack(global_parameters)
            model_send_requests.append(dist.isend(tensor=model_buffer, dst=client_rank))
            bytes_transmitted = num_bytes(model_buffer)
            logger.log_end("communication", {"round": round_number, "from": server_rank, "to": client_rank, "bytes_sent": bytes_transmitted})
            logging.info(f"[FedAVG Server] Round {round_number}, model was sent to rank {client_rank}!")
        
        for model_request in model_send_requests:
            model_request.wait()
    
    def receive_from_clients(self, participating_clients, global_parameters, global_state, device):
        """Receive updates from participating clients"""
        # Receive model updates from all participants
        client_parameter_updates = {}
        model_update_receive_info = []

        # Initiate all non-blocking receives for model updates
        for client_rank in participating_clients:
            model_update_buffer = torch.zeros_like(pack(global_parameters))
            model_update_request = dist.irecv(tensor=model_update_buffer, src=client_rank)
            model_update_receive_info.append((model_update_request, model_update_buffer, client_rank))

        # Wait for model updates
        for model_update_request, model_update_buffer, client_rank in model_update_receive_info:
            model_update_request.wait()
            client_parameter_updates[client_rank] = unpack(model_update_buffer, [param.shape for param in global_parameters])

        return client_parameter_updates
    
    def server_process(self, server_rank, shared_parameter_arrays, shared_state_arrays):
        """Server process (rank 0) - handles aggregation and client coordination"""
        torch.manual_seed(self.parent.config["seed"])
        np.random.seed(self.parent.config["seed"])
        communication_device = 'cpu'
        event_logger = EventLogger(log_file_name=self.parent.log_name)
        
        # Initialize global model
        global_model = self.parent.create_model()
        global_parameters = [param.to(communication_device) for param in global_model.parameters()]
        global_state = [state.to(communication_device) for state in global_model.buffers()]
        
        
        # Initialize communication
        communication_backend = 'gloo'  # Use gloo for CPU communication
        self.parent.init_process(server_rank, self.parent.size, communication_backend, self.parent.ports[0], self.parent.group_names[0])
        
        current_round = 0
        training_start_time = time.time()
        training_end_time = training_start_time + self.parent.train_time * 60
        
        while time.time() < training_end_time:
            event_logger.log_start("fedavg_round")
            
            # Select participating clients
            participating_clients = self.select_participating_clients(current_round, self.parent.size-1)
            logging.info(f"[FedAVG Server] Round {current_round}, Participants: {participating_clients}")
            
            # Send global model to all participating clients
            self.send_to_clients(participating_clients, global_parameters, communication_device, event_logger, current_round, server_rank)
            
            # Receive updates from participating clients
            client_parameter_updates = self.receive_from_clients(participating_clients, global_parameters, global_optimizer_state, communication_device)
            
            # Receive state from only one participating client (the first one)
            if participating_clients:
                designated_state_sender = participating_clients[0]  # Choose the first participating client
                logging.info(f"[FedAVG Server] Round {current_round}, receiving state from client {designated_state_sender}")
                
                # Receive state from the selected client
                state_buffer = torch.zeros_like(pack(global_optimizer_state))
                dist.recv(tensor=state_buffer, src=designated_state_sender)
                received_client_state = unpack(state_buffer, [state.shape for state in global_optimizer_state])
                
                # Set the global state from the selected client
                for global_state_param, client_state_param in zip(global_state, received_client_state):
                    global_state_param.data = client_state_param.to(communication_device)
            
            
            # Weighted aggregation
            for client_rank, parameter_update in client_parameter_updates.items():
                aggregation_weight = 1 / len(participating_clients)
                for global_param, client_param in zip(global_parameters, parameter_update):
                    global_param.data += aggregation_weight * client_param.to(communication_device)

            # Update shared arrays
            for param, shared_array in zip(global_parameters, shared_parameter_arrays):
                np.copyto(np.frombuffer(shared_array.get_obj(), dtype=np.float32).reshape(param.shape),
                         param.cpu().detach().numpy())
            
            for state_param, shared_state_array in zip(global_state, shared_state_arrays):
                np.copyto(np.frombuffer(shared_state_array.get_obj(), dtype=np.float32).reshape(state_param.shape),
                         state_param.cpu().detach().numpy())
            
            event_logger.log_end("fedavg_round", {"round": current_round, "participants": len(participating_clients), "total_data_size": total_dataset_size})
            current_round += 1
            dist.barrier()
            
            
        
        # Signal end to all clients
        for client_rank in range(1, self.parent.size):
            termination_signal = torch.tensor(-10, dtype=torch.int32).to(communication_device)  # -10 = end training
            dist.isend(tensor=termination_signal, dst=client_rank)
        dist.barrier()

        dist.destroy_process_group()
        logging.info(f"[FedAVG Server] Finished {current_round} rounds")
    
    def client_process(self, client_rank,):
        """Client process (client_rank > 0) - performs local training"""
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
        
        

        batch_data_iterator = training_task.data.iterator(
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
            logging.info(f"[FedAVG Client {client_rank}] notification received!")
            if notification.item() == -10:  # End signal
                break
            elif notification.item() == 0:  # End training
                 dist.barrier()
            elif notification.item() in [1,2]:  # Start training (no state sending)
                
                # Receive global model from server
                logging.info(f"[FedAVG Client {client_rank}] receiveing the model!")
                buffer = torch.zeros_like(pack(parameters), device=comm_device)
                dist.recv(tensor=buffer, src=0)
                global_params = unpack(buffer, [p.shape for p in parameters])
                logging.info(f"[FedAVG Client {client_rank}] received the model!")
                
                # Update local parameters with global model
                for local_param, global_param in zip(parameters, global_params):
                    local_param.data = global_param.to(device)
                
                # Perform local SGD
                logger.log_start("local sgd")
                epoch = self.parent.local_sgd(task, parameters, state, base_optimizer, base_optimizer_state, batch_data_gen, (time.time() - start_time))
                logger.log_end("local sgd", {"rank": client_rank, "iteration": self.parent.tau, "epoch": epoch})
                
                # Send data size to server
                data_size_tensor = torch.tensor(local_data_size, dtype=torch.int32).to(comm_device)
                dist.send(tensor=data_size_tensor, dst=0)
                
                dif_parameters = [param.to(comm_device) - global_param.to(comm_device) for param, global_param in zip(parameters, global_params)]
                # Send updated model to server
                logger.log_start("communication")
                buffer = pack(dif_parameters)
                dist.send(tensor=buffer, dst=0)
                bytes_sent = num_bytes(buffer)
                logger.log_end("communication", {"from": client_rank, "to": 0, "bytes_sent": bytes_sent})

                if notification.item() == 2:  # Start training + send state
                    # Send state to server (only this client sends state)
                    logging.info(f"[FedAVG Client {client_rank}] sending state to server!")
                    buffer = pack(state).to(comm_device)
                    dist.send(tensor=buffer, dst=0)
                    bytes_sent = num_bytes(buffer)
                    logging.info(f"[FedAVG Client {client_rank}] state sent to server!")

                dist.barrier()
        
        dist.barrier()
        dist.destroy_process_group()
        logging.info(f"[FedAVG Client {client_rank}] Finished")
    
    def run(self, rank):
        """Main FedAVG execution"""
        model = self.parent.create_model()
        if rank == 0:
            shared_arrays = [Array('f', param.numel(), lock=True) for param in model.parameters()]
            shared_state = [Array('f', state.numel(), lock=True) for state in model.buffers()]
        
            # Initialize shared arrays
            for param, shared_array in zip(model.parameters(), shared_arrays):
                np.copyto(np.frombuffer(shared_array.get_obj(), dtype=np.float32).reshape(param.shape),
                        param.cpu().detach().numpy())

            eval_process_active = Value('i', 1)
            # Server process
            eval_process = Process(target=self.parent.evaluation_process, 
                                 args=(self.parent.eval_gpu, shared_arrays, shared_state, eval_process_active, None))
            eval_process.start()
            
            server_process = self.server_process(rank, shared_arrays, shared_state)
            server_process.join()
            eval_process_active.value = 0
            eval_process.join()
        else:
            # Client process
            self.client_process(rank,)
