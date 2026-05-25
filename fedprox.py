import logging
import time

import numpy as np
import torch
import torch.distributed as dist

from base_optimizers import configure_base_optimizer
from fedavg import FedAVG
from logger import EventLogger
from utils.communication import pack, unpack, num_bytes


class FedProx(FedAVG):
    def __init__(self, parent):
        super().__init__(parent)
        self.mu = parent.config.get("fedprox_param", 0.0)

    def client_process(self, client_rank):
        """Client process (client_rank > 0) - performs local training with FedProx regularization."""
        torch.manual_seed(self.parent.config["seed"] + client_rank)
        np.random.seed(self.parent.config["seed"] + client_rank)
        comm_device = "cpu"
        training_device = torch.device(f"cuda:{self.parent.local_rank}" if torch.cuda.is_available() else "cpu")
        event_logger = EventLogger(log_file_name=self.parent.log_name)

        training_task = self.parent.configure_task(client_rank, training_device)
        parameters, state = training_task.initialize(self.parent.config["seed"])
        base_optimizer = configure_base_optimizer(self.parent.config)
        base_optimizer_state = base_optimizer.init(parameters)

        batch_data_gen = training_task.data.iterator(
            batch_size=self.parent.config["batch_size"],
            shuffle=True,
        )

        communication_backend = "gloo"
        self.parent.init_process(client_rank, self.parent.size, communication_backend, self.parent.ports[0], self.parent.group_names[0])

        training_start_time = time.time()

        while True:
            notification = torch.tensor(-1, dtype=torch.int32).to(comm_device)
            dist.recv(tensor=notification, src=0)
            logging.info(f"[FedProx Client {client_rank}] notification received!")
            if notification.item() == -10:
                break
            if notification.item() == 0:
                dist.barrier()
                continue

            logging.info(f"[FedProx Client {client_rank}] receiving the model!")
            buffer = torch.zeros_like(pack(parameters), device=comm_device)
            dist.recv(tensor=buffer, src=0)
            global_params = unpack(buffer, [p.shape for p in parameters])
            logging.info(f"[FedProx Client {client_rank}] received the model!")

            global_params = [global_param.to(training_device) for global_param in global_params]
            for local_param, global_param in zip(parameters, global_params):
                local_param.data = global_param.clone()

            event_logger.log_start("local sgd")
            for _ in range(self.parent.tau):
                time_for_lr_schedule = time.time() - training_start_time
                epoch, gradients = self.parent.local_sgd(
                    training_task,
                    parameters,
                    state,
                    base_optimizer,
                    base_optimizer_state,
                    batch_data_gen,
                    time_for_lr_schedule,
                    1,
                )

                # Apply FedProx correction at each local step:
                # w <- w - eta * mu * (w - w_t)
                local_learning_rate = self.parent.config["learning_rate"] * self.parent.learning_rate_schedule(
                    time_for_lr_schedule
                )
                for param, global_param in zip(parameters, global_params):
                    param.data += -local_learning_rate * self.mu * (param.data - global_param)

            event_logger.log_end("local sgd", {"rank": client_rank, "iteration": self.parent.tau, "epoch": epoch})

            dif_parameters = [param.to(comm_device) - global_param.to(comm_device) for param, global_param in zip(parameters, global_params)]
            event_logger.log_start("communication")
            send_buffer = pack(dif_parameters)
            dist.send(tensor=send_buffer, dst=0)
            bytes_sent = num_bytes(send_buffer)
            event_logger.log_end("communication", {"from": client_rank, "to": 0, "bytes_sent": bytes_sent})
            dist.barrier()

        dist.barrier()
        dist.destroy_process_group()
        logging.info(f"[FedProx Client {client_rank}] Finished")
