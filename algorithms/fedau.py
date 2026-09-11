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


class FedAU:
    """FedAvg with FedAU aggregation weights.

    Active clients train from the current global model and return
        Δ_t^n = y_{t,I}^n - x_t
    (inactive clients contribute Δ = 0). Aggregation is
        x_{t+1} = x_t + (η / N) Σ_n ω_t^n Δ_t^n
    where ω_t^n is computed from participation history (Algorithm 2) with
    interval cap K (config key ``fedau_k``, default 50).
    """

    def __init__(self, parent):
        self.parent = parent
        self.participation_rate = parent.config.get("participation_rate", 1)
        self.global_learning_rate = parent.config.get("global_learning_rate", 1.0)
        self.K = int(parent.config.get("fedau_k", 50))

    def select_participating_clients(self, round_number, total_clients):
        return sample_participating_clients(self.parent.config, round_number, total_clients)

    def send_to_clients(self, participating_clients, global_parameters, device, logger, round_number, server_rank):
        notification_requests = []
        for client_rank in range(1, self.parent.size):
            if client_rank not in participating_clients:
                notification_requests.append(
                    dist.isend(tensor=torch.tensor(0, dtype=torch.int32).to(device), dst=client_rank)
                )
                logging.info(f"[FedAU Server] Round {round_number}, skip notification to rank {client_rank}")
            else:
                notification_requests.append(
                    dist.isend(tensor=torch.tensor(1, dtype=torch.int32).to(device), dst=client_rank)
                )
                logging.info(f"[FedAU Server] Round {round_number}, train notification to rank {client_rank}")
        for req in notification_requests:
            req.wait()

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

    def advance_weights(self, omega, M, S_diamond, participated_prev):
        """Advance ω_t using Algorithm 2 given I_{t-1} in participated_prev."""
        for client_rank in omega:
            S_diamond[client_rank] += 1
            if participated_prev[client_rank] or S_diamond[client_rank] == self.K:
                S_n = S_diamond[client_rank]
                if M[client_rank] == 0:
                    omega[client_rank] = float(S_n)
                else:
                    omega[client_rank] = (M[client_rank] * omega[client_rank] + S_n) / (M[client_rank] + 1)
                M[client_rank] += 1
                S_diamond[client_rank] = 0

    def server_process(self, server_rank, shared_parameter_arrays):
        torch.manual_seed(self.parent.config["seed"])
        np.random.seed(self.parent.config["seed"])
        communication_device = "cpu"
        event_logger = EventLogger(log_file_name=self.parent.log_name)

        global_model = self.parent.create_model()
        global_parameters = [p.to(communication_device) for p in global_model.parameters()]
        num_clients = self.parent.size - 1

        # Algorithm 2 state: ω_0^n = 1, M_n = 0, S_n^♦ = 0
        omega = {r: 1.0 for r in range(1, self.parent.size)}
        M = {r: 0 for r in range(1, self.parent.size)}
        S_diamond = {r: 0 for r in range(1, self.parent.size)}
        participated_prev = {r: False for r in range(1, self.parent.size)}

        self.parent.init_process(
            server_rank, self.parent.size, "gloo", self.parent.ports[0], self.parent.group_names[0]
        )

        current_round = 0
        training_start_time = time.time()
        training_end_time = training_start_time + self.parent.train_time * 60

        while time.time() < training_end_time:
            # For t >= 1, refresh ω_t from I_{t-1} before aggregating this round
            if current_round >= 1:
                self.advance_weights(omega, M, S_diamond, participated_prev)

            participating = self.select_participating_clients(current_round, num_clients)
            logging.info(
                f"[FedAU Server] Round {current_round}, participants: {participating}, "
                f"weights: {{{', '.join(f'{r}:{omega[r]:.3f}' for r in sorted(omega))}}}"
            )

            self.send_to_clients(
                participating, global_parameters, communication_device, event_logger, current_round, server_rank
            )
            updates = self.receive_from_clients(participating, global_parameters)

            # x <- x + (η / N) Σ_n ω_t^n Δ_t^n  (Δ=0 for inactive)
            scale = self.global_learning_rate / num_clients
            for client_rank, delta in updates.items():
                w = omega[client_rank] * scale
                for gp, d in zip(global_parameters, delta):
                    gp.data.add_(d.to(communication_device), alpha=w)

            for param, shared_array in zip(global_parameters, shared_parameter_arrays):
                np.copyto(
                    np.frombuffer(shared_array.get_obj(), dtype=np.float32).reshape(param.shape),
                    param.cpu().detach().numpy(),
                )

            participated_prev = {
                r: (r in participating) for r in range(1, self.parent.size)
            }
            current_round += 1
            dist.barrier()

        for client_rank in range(1, self.parent.size):
            dist.isend(
                tensor=torch.tensor(-10, dtype=torch.int32).to(communication_device), dst=client_rank
            ).wait()
        dist.barrier()
        dist.destroy_process_group()
        logging.info(f"[FedAU Server] finished {current_round} rounds")

    def client_process(self, client_rank):
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

        self.parent.init_process(client_rank, self.parent.size, "gloo", self.parent.ports[0], self.parent.group_names[0])
        training_start_time = time.time()

        while True:
            notification = torch.tensor(-1, dtype=torch.int32).to(comm_device)
            dist.recv(tensor=notification, src=0)
            if notification.item() == -10:
                break
            if notification.item() == 0:
                dist.barrier()
                continue

            buffer = torch.zeros_like(pack(parameters), device=comm_device)
            dist.recv(tensor=buffer, src=0)
            global_params = unpack(buffer, [p.shape for p in parameters])

            for local_p, g in zip(parameters, global_params):
                local_p.data = g.to(training_device)

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

            # Δ_t^n = y_{t,I}^n - x_t
            delta = [
                p.to(comm_device) - g.to(comm_device) for p, g in zip(parameters, global_params)
            ]

            event_logger.log_start("communication")
            out_buf = pack(delta)
            dist.send(tensor=out_buf, dst=0)
            event_logger.log_end(
                "communication",
                {"from": client_rank, "to": 0, "bytes_sent": num_bytes(out_buf)},
            )

            dist.barrier()

        dist.barrier()
        dist.destroy_process_group()
        logging.info(f"[FedAU Client {client_rank}] finished")

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
