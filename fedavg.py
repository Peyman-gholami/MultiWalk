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
from distributed_training import EventLogger


class FedAVG:
    def __init__(self, parent):
        self.parent = parent
        # FedAVG specific parameters
        self.participation_rate = parent.config.get('participation_rate', 0.1)  # Fraction of clients per round
        self.min_clients = max(1, int(self.participation_rate * (parent.size - 1)))  # At least 1 client
        
    def select_participating_clients(self, round_num, total_clients):
        """Return the actual ranks of selected participating clients for current round"""
        torch.manual_seed(self.parent.config["seed"] + round_num)
        num_participants = max(self.min_clients, int(self.participation_rate * total_clients))
        # Client ranks are from 1 to total_clients (server is rank 0)
        participants = torch.randperm(total_clients)[:num_participants].tolist()
        client_ranks = [p + 1 for p in participants]  # Actual distributed ranks
        return client_ranks
    
    def server_process(self, rank, shared_arrays, shared_state):
        """Server process (rank 0) - handles aggregation and client coordination"""
        torch.manual_seed(self.parent.config["seed"])
        np.random.seed(self.parent.config["seed"])
        
        device = torch.device(f'cuda:{self.parent.local_rank}' if torch.cuda.is_available() else 'cpu')
        logger = EventLogger(log_file_name=self.parent.log_name)
        
        # Initialize global model
        model = self.parent.create_model()
        global_parameters = [param.to(device) for param in model.parameters()]
        global_state = [state.to(device) for state in model.buffers()]
        
        
        # Initialize communication
        backend = 'gloo'  # Use gloo for CPU communication
        self.parent.init_process(rank, self.parent.size, backend, self.parent.ports[0], 'fedavg_server')
        
        round_num = 0
        start_time = time.time()
        end_time = start_time + self.parent.train_time * 60
        
        while time.time() < end_time:
            logger.log_start("fedavg_round")
            
            # Select participating clients
            participants = self.select_participating_clients(round_num, self.parent.size-1)
            logging.info(f"[FedAVG Server] Round {round_num}, Participants: {participants}")
            
            # Send global model to all participating clients
            for client_rank in range(1, self.parent.size):
                if client_rank not in participants:
                    notification = torch.tensor(0, dtype=torch.int32).to(device)  # 0 = end training
                    dist.isend(tensor=notification, dst=client_rank)
                else:
                    notification = torch.tensor(1, dtype=torch.int32).to(device)  # 1 = start training
                    dist.isend(tensor=notification, dst=client_rank)
            dist.barrier()
            
            for client_rank in participants:
                logger.log_start("communication")
                buffer = pack(global_parameters)
                dist.isend(tensor=buffer, dst=client_rank)
                bytes_sent = num_bytes(buffer)
                logger.log_end("communication", {"round": round_num, "from": rank, "to": client_rank, "bytes_sent": bytes_sent})
            
            dist.barrier()

            # Collect updates from participating clients
            client_updates = {}
            client_data_sizes = {}
            data_size_reqs = []
            model_update_reqs = []
            for client_rank in participants:
                
                # Receive data size information
                data_size_tensor = torch.tensor(0, dtype=torch.int32).to(device)
                req = dist.irecv(tensor=data_size_tensor, src=client_rank)
                data_size_reqs.append((req, client_rank))
                
            
            for req in data_size_reqs:
                req.wait()
            
            for req, client_rank in data_size_reqs:
                client_data_sizes[client_rank] = req.tensor().item()
            

            for client_rank in participants:
                buffer = torch.zeros_like(pack(global_parameters))
                req = dist.irecv(tensor=buffer, src=client_rank)
                model_update_reqs.append((req, client_rank))

            for req in model_update_reqs:
                req.wait()
            
            for req, client_rank in model_update_reqs:
                client_updates[client_rank] = unpack(req.tensor(), [p.shape for p in global_parameters])
            
            # Aggregate updates using weighted average
            total_data_size = sum(client_data_sizes.values())
            
            # Initialize aggregated parameters
            for i, param in enumerate(global_parameters):
                param.data.zero_()
            
            # Weighted aggregation
            for client_rank, update in client_updates.items():
                weight = client_data_sizes[client_rank] / total_data_size
                for global_param, client_param in zip(global_parameters, update):
                    global_param.data += weight * client_param.to(device)
            
            # Update shared arrays
            for param, shared_array in zip(global_parameters, shared_arrays):
                np.copyto(np.frombuffer(shared_array.get_obj(), dtype=np.float32).reshape(param.shape),
                         param.cpu().detach().numpy())
            
            for state, shared_state_array in zip(global_state, shared_state):
                np.copyto(np.frombuffer(shared_state_array.get_obj(), dtype=np.float32).reshape(state.shape),
                         state.cpu().detach().numpy())
            
            logger.log_end("fedavg_round", {"round": round_num, "participants": len(participants), "total_data_size": total_data_size})
            round_num += 1
            dist.barrier()
            
            
        
        # Signal end to all clients
        for client_rank in range(1, self.parent.size):
            end_signal = torch.tensor(-10, dtype=torch.int32).to(device)  # -10 = end training
            dist.isend(tensor=end_signal, dst=client_rank)
        dist.barrier()

        dist.destroy_process_group()
        logging.info(f"[FedAVG Server] Finished {round_num} rounds")
    
    def client_process(self, rank,):
        """Client process (rank > 0) - performs local training"""
        torch.manual_seed(self.parent.config["seed"] + rank)
        np.random.seed(self.parent.config["seed"] + rank)
        
        device = torch.device(f'cuda:{self.parent.local_rank}' if torch.cuda.is_available() else 'cpu')
        logger = EventLogger(log_file_name=self.parent.log_name)
        
        # Initialize task and model
        task = self.parent.configure_task(rank, device)
        parameters, state = task.initialize(self.parent.config["seed"])
        base_optimizer = configure_base_optimizer(self.parent.config)
        base_optimizer_state = base_optimizer.init(parameters)
        
        batch_data_gen = task.data.iterator(
            batch_size=self.parent.config["batch_size"],
            shuffle=True,
        )
        
        # Initialize communication
        backend = 'gloo'
        self.parent.init_process(rank, self.parent.size, backend, self.parent.ports[0], 'fedavg_client')
        
        # Get local data size for weighted aggregation
        local_data_size = len(task.data)
        
        while True:
            # Wait for server notification
            notification = torch.tensor(-1, dtype=torch.int32).to(device)
            dist.recv(tensor=notification, src=0)
            dist.barrier()
            if notification.item() == -10:  # End signal
                break
            elif notification.item() == 0:  # End training
                dist.barrier()
                dist.barrier() 
            elif notification.item() == 1:  # Start training
                
                # Receive global model from server
                buffer = torch.zeros_like(pack(parameters))
                dist.recv(tensor=buffer, src=0)
                global_params = unpack(buffer, [p.shape for p in parameters])
                
                dist.barrier()

                # Update local parameters with global model
                for local_param, global_param in zip(parameters, global_params):
                    local_param.data = global_param.to(device)
                
                # Perform local SGD
                logger.log_start("local sgd")
                epoch = self.parent.local_sgd(task, parameters, state, base_optimizer, base_optimizer_state, batch_data_gen, (time.time() - start_time))
                logger.log_end("local sgd", {"rank": rank, "iteration": self.parent.tau, "epoch": epoch})
                iteration += self.parent.tau
                
                # Send data size to server
                data_size_tensor = torch.tensor(local_data_size, dtype=torch.int32).to(device)
                dist.send(tensor=data_size_tensor, dst=0)
                
                # Send updated model to server
                logger.log_start("communication")
                buffer = pack(parameters)
                dist.send(tensor=buffer, dst=0)
                bytes_sent = num_bytes(buffer)
                logger.log_end("communication", {"from": rank, "to": 0, "bytes_sent": bytes_sent})
                dist.barrier()

        dist.destroy_process_group()
        logging.info(f"[FedAVG Client {rank}] Finished")
    
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
