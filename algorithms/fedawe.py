import torch
import torch.distributed as dist
import logging
import numpy as np
from torch.multiprocessing import Process, Array, Value
import time
from base_optimizers import configure_base_optimizer
from utils.communication import pack, unpack, num_bytes
from logger import EventLogger
from utils.participation import select_participating_clients as sample_participating_clients


class FedAWE:
    """Federated Averaging with Weighted Elasticity (FedAWE).

    Clients keep a local model between rounds. Active clients run s local SGD
    steps from their local model, form
        G_i^t = x_i^t - x_i^{(t,s)}
        x_i^{t†} = x_i^{(t,0)} - η_g (t - τ_i(t)) G_i^t
    and report x_i^{t†}. The server averages those reports. Active clients then
    sync to the new global model; inactive clients keep their local model and τ.
    """

    def __init__(self, parent):
        self.parent = parent
        self.participation_rate = parent.config.get("participation_rate", 1)
        self.global_learning_rate = parent.config.get("global_learning_rate", 1.0)

    def select_participating_clients(self, round_number, total_clients):
        return sample_participating_clients(self.parent.config, round_number, total_clients)

    def notify_clients(self, participating_clients, device, round_number):
        notification_requests = []
        for client_rank in range(1, self.parent.size):
            if client_rank not in participating_clients:
                notification_requests.append(
                    dist.isend(tensor=torch.tensor(0, dtype=torch.int32).to(device), dst=client_rank)
                )
                logging.info(f"[FedAWE Server] Round {round_number}, skip notification to rank {client_rank}")
            else:
                notification_requests.append(
                    dist.isend(tensor=torch.tensor(1, dtype=torch.int32).to(device), dst=client_rank)
                )
                logging.info(f"[FedAWE Server] Round {round_number}, train notification to rank {client_rank}")
        for req in notification_requests:
            req.wait()
        logging.info(f"[FedAWE Server] Round {round_number}, all notifications sent")

    def receive_from_clients(self, participating_clients, global_parameters):
        receive_info = []
        for client_rank in participating_clients:
            buf = torch.zeros_like(pack(global_parameters))
            receive_info.append((dist.irecv(tensor=buf, src=client_rank), buf, client_rank))

        out = {}
        for req, buf, client_rank in receive_info:
            req.wait()
            out[client_rank] = unpack(buf, [p.shape for p in global_parameters])
        return out

    def send_global_to_clients(self, participating_clients, global_parameters, device, logger, round_number, server_rank):
        model_send_requests = []
        for client_rank in participating_clients:
            logger.log_start("communication")
            model_buffer = pack(global_parameters)
            model_send_requests.append(dist.isend(tensor=model_buffer, dst=client_rank))
            logger.log_end(
                "communication",
                {
                    "round": round_number,
                    "from": server_rank,
                    "to": client_rank,
                    "bytes_sent": num_bytes(model_buffer),
                },
            )
        for req in model_send_requests:
            req.wait()

    def server_process(self, server_rank, shared_parameter_arrays):
        torch.manual_seed(self.parent.config["seed"])
        np.random.seed(self.parent.config["seed"])
        communication_device = "cpu"
        event_logger = EventLogger(log_file_name=self.parent.log_name)

        global_model = self.parent.create_model()
        global_parameters = [p.to(communication_device) for p in global_model.parameters()]
        num_clients = self.parent.size - 1

        communication_backend = "gloo"
        self.parent.init_process(
            server_rank, self.parent.size, communication_backend, self.parent.ports[0], self.parent.group_names[0]
        )

        current_round = 0
        training_start_time = time.time()
        training_end_time = training_start_time + self.parent.train_time * 60

        while time.time() < training_end_time:
            participating = self.select_participating_clients(current_round, num_clients)
            logging.info(f"[FedAWE Server] Round {current_round}, participants: {participating}")

            self.notify_clients(participating, communication_device, current_round)

            # Active clients train from their local models and report x_i^{t†}
            client_updates = self.receive_from_clients(participating, global_parameters)

            # x^{t+1} <- (1/|A^t|) sum_{i in A^t} x_i^{t†}
            for gp in global_parameters:
                gp.data.zero_()
            inv_n = 1.0 / len(participating)
            for tensors in client_updates.values():
                for gp, update in zip(global_parameters, tensors):
                    gp.data.add_(update.to(communication_device), alpha=inv_n)

            # Sync new global model to active clients only
            self.send_global_to_clients(
                participating, global_parameters, communication_device, event_logger, current_round, server_rank
            )

            for param, shared_array in zip(global_parameters, shared_parameter_arrays):
                np.copyto(
                    np.frombuffer(shared_array.get_obj(), dtype=np.float32).reshape(param.shape),
                    param.cpu().detach().numpy(),
                )

            current_round += 1
            dist.barrier()

        for client_rank in range(1, self.parent.size):
            dist.isend(
                tensor=torch.tensor(-10, dtype=torch.int32).to(communication_device), dst=client_rank
            ).wait()
        dist.barrier()
        dist.destroy_process_group()
        logging.info(f"[FedAWE Server] finished {current_round} rounds")

    def client_process(self, client_rank):
        torch.manual_seed(self.parent.config["seed"] + client_rank)
        np.random.seed(self.parent.config["seed"] + client_rank)
        comm_device = "cpu"
        training_device = torch.device(f"cuda:{self.parent.local_rank}" if torch.cuda.is_available() else "cpu")
        event_logger = EventLogger(log_file_name=self.parent.log_name)

        # Official FedAWE uses plain SGD: no momentum / weight decay.
        self.parent.config["momentum"] = 0.0
        self.parent.config["weight_decay"] = 0.0

        training_task = self.parent.configure_task(client_rank, training_device)
        parameters, state = training_task.initialize(self.parent.config["seed"])
        base_optimizer = configure_base_optimizer(self.parent.config)
        base_optimizer_state = base_optimizer.init(parameters)

        batch_data_gen = training_task.data.iterator(
            batch_size=self.parent.config["batch_size"],
            shuffle=True,
        )

        self.parent.init_process(client_rank, self.parent.size, "gloo", self.parent.ports[0], self.parent.group_names[0])

        training_start_time = time.time()
        # τ_i(0) = -1; clients keep local x_i between inactive rounds
        last_active_round = -1
        current_round = 0

        while True:
            notification = torch.tensor(-1, dtype=torch.int32).to(comm_device)
            dist.recv(tensor=notification, src=0)
            if notification.item() == -10:
                break
            if notification.item() == 0:
                # Inactive: keep x_i and τ_i
                current_round += 1
                dist.barrier()
                continue

            # Active: train from local x_i^t (do not pull global first)
            x_start = [p.detach().clone() for p in parameters]

            event_logger.log_start("local sgd")
            epoch, _ = self.parent.local_sgd(
                training_task,
                parameters,
                state,
                base_optimizer,
                base_optimizer_state,
                batch_data_gen,
                time.time() - training_start_time,
                self.parent.tau,
            )
            event_logger.log_end(
                "local sgd",
                {"rank": client_rank, "iteration": self.parent.tau, "epoch": epoch},
            )

            # G_i^t = x_i^t - x_i^{(t,s)}
            # x_i^{t†} = x_i^{(t,0)} - η_g (t - τ_i(t)) G_i^t
            elasticity = self.global_learning_rate * (current_round - last_active_round)
            x_dagger = [
                start.to(comm_device) - elasticity * (start.to(comm_device) - p.to(comm_device))
                for start, p in zip(x_start, parameters)
            ]

            event_logger.log_start("communication")
            out_buf = pack(x_dagger)
            dist.send(tensor=out_buf, dst=0)
            event_logger.log_end(
                "communication",
                {"from": client_rank, "to": 0, "bytes_sent": num_bytes(out_buf)},
            )

            # Receive aggregated global model and set x_i^{t+1} <- x^{t+1}
            buffer = torch.zeros_like(pack(parameters), device=comm_device)
            dist.recv(tensor=buffer, src=0)
            global_params = unpack(buffer, [p.shape for p in parameters])
            for local_p, g in zip(parameters, global_params):
                local_p.data = g.to(training_device)

            last_active_round = current_round
            current_round += 1
            dist.barrier()

        dist.barrier()
        dist.destroy_process_group()
        logging.info(f"[FedAWE Client {client_rank}] finished")

    def run(self, rank):
        model = self.parent.create_model()
        if rank == 0:
            shared_arrays = [Array("f", p.numel(), lock=True) for p in model.parameters()]
            for p, arr in zip(model.parameters(), shared_arrays):
                np.copyto(
                    np.frombuffer(arr.get_obj(), dtype=np.float32).reshape(p.shape),
                    p.cpu().detach().numpy(),
                )

            eval_process_active = Value("i", 1)
            eval_process = Process(
                target=self.parent.evaluation_process,
                args=(self.parent.eval_gpu, shared_arrays, None, eval_process_active, None),
            )
            eval_process.start()

            server_proc = Process(target=self.server_process, args=(rank, shared_arrays))
            server_proc.start()
            server_proc.join()
            eval_process_active.value = 0
            eval_process.join()
        else:
            self.client_process(rank)
