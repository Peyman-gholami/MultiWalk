import torch
import torch.distributed as dist
import logging
import numpy as np
from torch.multiprocessing import Process, Array, Value
import time
from base_optimizers import configure_base_optimizer
from utils.communication import pack, unpack, num_bytes
from logger import EventLogger


class MIFA:
    """Memory-augmented Impatient Federated Averaging (MIFA).

    Server keeps a memory tensor G^i per client i. Each round, only active clients
    overwrite their memory; the global update is
        w_{t+1} <- w_t - (eta_t / N) * sum_i G^i
    where N is the total number of clients. Active clients return
        (1/eta_t) * (w_t - w_{t,K}^i) after K local SGD steps (tau in this codebase).
    Uses only config learning_rate and lr schedule (no global_learning_rate).
    """

    def __init__(self, parent):
        self.parent = parent
        self.participation_rate = parent.config.get("participation_rate", 1)

    def select_participating_clients(self, round_number, total_clients):
        torch.manual_seed(self.parent.config["seed"] + round_number)
        num_participants = max(1, int(self.participation_rate * total_clients))
        participant_indices = torch.randperm(total_clients)[:num_participants].tolist()
        return [idx + 1 for idx in participant_indices]

    def send_to_clients(self, participating_clients, global_parameters, device, logger, round_number, server_rank):
        notification_requests = []
        for client_rank in range(1, self.parent.size):
            if client_rank not in participating_clients:
                notification_requests.append(
                    dist.isend(tensor=torch.tensor(0, dtype=torch.int32).to(device), dst=client_rank)
                )
                logging.info(f"[MIFA Server] Round {round_number}, skip notification to rank {client_rank}")
            else:
                notification_requests.append(
                    dist.isend(tensor=torch.tensor(1, dtype=torch.int32).to(device), dst=client_rank)
                )
                logging.info(f"[MIFA Server] Round {round_number}, train notification to rank {client_rank}")

        for req in notification_requests:
            req.wait()
        logging.info(f"[MIFA Server] Round {round_number}, all notifications sent")

        model_send_requests = []
        for client_rank in participating_clients:
            logger.log_start("communication")
            model_buffer = pack(global_parameters)
            model_send_requests.append(dist.isend(tensor=model_buffer, dst=client_rank))
            logger.log_end(
                "communication",
                {"round": round_number, "from": server_rank, "to": client_rank, "bytes_sent": num_bytes(model_buffer)},
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

    def server_process(self, server_rank, shared_parameter_arrays):
        torch.manual_seed(self.parent.config["seed"])
        np.random.seed(self.parent.config["seed"])
        communication_device = "cpu"
        event_logger = EventLogger(log_file_name=self.parent.log_name)

        global_model = self.parent.create_model()
        global_parameters = [p.to(communication_device) for p in global_model.parameters()]
        num_clients = self.parent.size - 1

        # G^i for each client rank 1..N — initialized to zero; inactive rounds keep previous value
        client_memory = {
            r: [torch.zeros_like(p, device=communication_device) for p in global_parameters]
            for r in range(1, self.parent.size)
        }

        communication_backend = "gloo"
        self.parent.init_process(server_rank, self.parent.size, communication_backend, self.parent.ports[0], self.parent.group_names[0])

        current_round = 0
        training_start_time = time.time()
        training_end_time = training_start_time + self.parent.train_time * 60

        while time.time() < training_end_time:
            participating = self.select_participating_clients(current_round, num_clients)
            logging.info(f"[MIFA Server] Round {current_round}, participants: {participating}")

            self.send_to_clients(participating, global_parameters, communication_device, event_logger, current_round, server_rank)

            updates = self.receive_from_clients(participating, global_parameters)
            for client_rank, tensors in updates.items():
                for mem, u in zip(client_memory[client_rank], tensors):
                    mem.copy_(u.to(communication_device))

            elapsed = time.time() - training_start_time
            eta_t = self.parent.config["learning_rate"] * self.parent.learning_rate_schedule(elapsed)

            sum_G = [torch.zeros_like(p) for p in global_parameters]
            for r in range(1, self.parent.size):
                for acc, m in zip(sum_G, client_memory[r]):
                    acc.add_(m)

            scale = eta_t / num_clients
            for gp, sg in zip(global_parameters, sum_G):
                gp.data -= scale * sg

            for param, shared_array in zip(global_parameters, shared_parameter_arrays):
                np.copyto(
                    np.frombuffer(shared_array.get_obj(), dtype=np.float32).reshape(param.shape),
                    param.cpu().detach().numpy(),
                )

            current_round += 1
            dist.barrier()

        for client_rank in range(1, self.parent.size):
            dist.isend(tensor=torch.tensor(-10, dtype=torch.int32).to(communication_device), dst=client_rank).wait()
        dist.barrier()
        dist.destroy_process_group()
        logging.info(f"[MIFA Server] finished {current_round} rounds")

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
            w_t = unpack(buffer, [p.shape for p in parameters])

            for local_p, w in zip(parameters, w_t):
                local_p.data = w.to(training_device)

            w_t_snapshot = [w.clone() for w in w_t]

            elapsed = time.time() - training_start_time
            eta_t = self.parent.config["learning_rate"] * self.parent.learning_rate_schedule(elapsed)

            event_logger.log_start("local sgd")
            epoch, _ = self.parent.local_sgd(
                training_task,
                parameters,
                state,
                base_optimizer,
                base_optimizer_state,
                batch_data_gen,
                elapsed,
                self.parent.tau,
            )
            event_logger.log_end("local sgd", {"rank": client_rank, "iteration": self.parent.tau, "epoch": epoch})

            # Return (1/eta_t) * (w_t - w_{t,K}^i); avoid div by zero
            inv_eta = 1.0 / eta_t if eta_t > 0 else 0.0
            G_i = [inv_eta * (w0.to(comm_device) - p.to(comm_device)) for w0, p in zip(w_t_snapshot, parameters)]

            event_logger.log_start("communication")
            out_buf = pack(G_i)
            dist.send(tensor=out_buf, dst=0)
            event_logger.log_end("communication", {"from": client_rank, "to": 0, "bytes_sent": num_bytes(out_buf)})

            dist.barrier()

        dist.barrier()
        dist.destroy_process_group()
        logging.info(f"[MIFA Client {client_rank}] finished")

    def run(self, rank):
        model = self.parent.create_model()
        if rank == 0:
            shared_arrays = [Array("f", p.numel(), lock=True) for p in model.parameters()]
            for p, arr in zip(model.parameters(), shared_arrays):
                np.copyto(np.frombuffer(arr.get_obj(), dtype=np.float32).reshape(p.shape), p.cpu().detach().numpy())

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
