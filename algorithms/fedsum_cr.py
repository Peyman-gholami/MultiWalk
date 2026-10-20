"""FedSUM-CR, Algorithm 3 in You and Pu (2025), arXiv:2512.18275.

``learning_rate`` is the paper's eta_l: the actual local SGD step is eta_l/N.
``global_learning_rate`` is eta_g, ``tau`` is K, and N excludes server rank 0.
Local training uses the same optimizer and local_sgd helper as HScaffold, with
the FedSUM-CR correction applied separately after each optimizer step. Plain SGD
with momentum 0 matches Algorithm 3; momentum and Adam are optional extensions.
The model-difference correction requires constant learning rates; use
--lr_warmup_time 0 --no_lr_schedule. Task weight decay is reused unchanged.
"""

import logging
import math
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.multiprocessing import Array, Process, Value

from base_optimizers import configure_base_optimizer
from logger import EventLogger
from utils.communication import num_bytes, pack, unpack
from utils.participation import select_participating_clients


class _ClientMemory:
    def __init__(self, parameters):
        self.h = [torch.zeros_like(p) for p in parameters]
        self.z = [p.detach().clone() for p in parameters]
        self.a = -1


class _ScaledOptimizer:
    """Keep the shared local_sgd path while using the paper's eta_l / N step."""

    def __init__(self, optimizer, num_clients):
        self.optimizer = optimizer
        self.num_clients = num_clients

    def step(self, parameters, gradients, optimizer_state, lr):
        with torch.no_grad():
            self.optimizer.step(parameters, gradients, optimizer_state,
                                lr=lr / self.num_clients)


class _TaskState:
    """Retain returned task state, which local_sgd does not return to its caller."""

    def __init__(self, task, state):
        self.task = task
        self.state = state

    def loss_and_gradient(self, parameters, state, batch):
        loss, gradients, self.state = self.task.loss_and_gradient(parameters, state, batch)
        return loss, gradients, self.state


class FedSUMCR:
    """One model downlink and one delta uplink per active client and round."""

    def __init__(self, parent):
        self.parent = parent
        self.num_clients = parent.size - 1
        self.local_learning_rate = float(parent.config['learning_rate'])
        self.global_learning_rate = float(parent.config.get('global_learning_rate', 1))
        self.local_steps = parent.tau
        if self.num_clients < 1:
            raise ValueError('FedSUM-CR requires rank 0 plus at least one training client.')
        if (isinstance(self.local_steps, bool) or self.local_steps < 1
                or int(self.local_steps) != self.local_steps):
            raise ValueError('FedSUM-CR requires a positive integer tau (local steps K).')
        self.local_steps = int(self.local_steps)
        for name, value in [('learning_rate', self.local_learning_rate),
                            ('global_learning_rate', self.global_learning_rate)]:
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'FedSUM-CR requires a finite, positive {name}.')
        if (parent.config.get('lr_warmup_time', 0) != 0
                or parent.config.get('lr_schedule_milestones', [])):
            raise ValueError('FedSUM-CR requires constant learning rates; use '
                             '--lr_warmup_time 0 --no_lr_schedule.')

    def select_participating_clients(self, round_number, total_clients):
        # Empty active sets are valid: the server still follows its stored y.
        return select_participating_clients(
            self.parent.config, round_number, total_clients, ensure_at_least_one=False)

    def send_to_clients(self, participating, parameters, device, logger, round_number, server_rank):
        for client_rank in range(1, self.parent.size):
            notification = torch.tensor(int(client_rank in participating), dtype=torch.int32,
                                        device=device)
            dist.send(tensor=notification, dst=client_rank)
        if not participating:
            return
        # A single reusable buffer avoids allocating one model per participant.
        buffer = pack(parameters)
        for client_rank in participating:
            logger.log_start('communication')
            dist.send(tensor=buffer, dst=client_rank)
            logger.log_end('communication', {
                'round': round_number, 'from': server_rank, 'to': client_rank,
                'bytes_sent': num_bytes(buffer)})

    def local_update(self, task, parameters, state, batches, memory, round_number,
                     base_optimizer, base_optimizer_state):
        """FedSUM-CR controls with HScaffold's shared local optimizer path."""
        if round_number <= memory.a:
            raise ValueError('FedSUM-CR client rounds must increase strictly.')
        reference = [p.detach().clone() for p in parameters]
        delay = round_number - memory.a
        correction_scale = self.num_clients / (
            self.global_learning_rate * self.local_learning_rate * self.local_steps * delay)
        correction = [(z - x) * correction_scale - h
                      for z, x, h in zip(memory.z, reference, memory.h)]

        local_optimizer = _ScaledOptimizer(base_optimizer, self.num_clients)
        local_task = _TaskState(task, state)
        for _ in range(self.local_steps):
            # Rates are constant (validated above), so schedule time is immaterial.
            epoch, _ = self.parent.local_sgd(
                local_task, parameters, local_task.state, local_optimizer,
                base_optimizer_state, batches, time=0, local_steps=1)
            with torch.no_grad():
                for parameter, direction in zip(parameters, correction):
                    parameter.add_(direction, alpha=-self.local_learning_rate / self.num_clients)

        # With plain SGD, h_new is the mean gradient. With momentum/Adam it is
        # the mean optimizer direction, excluding the separately applied correction.
        displacement_scale = self.num_clients / (self.local_learning_rate * self.local_steps)
        with torch.no_grad():
            new_h = [(x - p) * displacement_scale - direction
                     for x, p, direction in zip(reference, parameters, correction)]
            delta = [new - old for new, old in zip(new_h, memory.h)]
        memory.h = new_h
        memory.z = reference  # Cache received x^(t), never the locally updated model.
        memory.a = round_number
        return epoch, local_task.state, delta

    def aggregate(self, parameters, control, updates):
        """Algorithm 3, line 15: update the persistent sum, then the global model."""
        with torch.no_grad():
            for delta in updates:
                for y, value in zip(control, delta):
                    y.add_(value)
            scale = self.global_learning_rate * self.local_learning_rate * self.local_steps
            for parameter, y in zip(parameters, control):
                parameter.add_(y, alpha=-scale / self.num_clients)

    def _receive_updates(self, participating, parameters):
        # Accumulate each delta immediately, keeping server memory independent of N.
        buffer = torch.empty_like(pack(parameters))
        shapes = [p.shape for p in parameters]
        for client_rank in participating:
            dist.recv(tensor=buffer, src=client_rank)
            yield unpack(buffer, shapes)

    def server_process(self, server_rank, shared_parameter_arrays):
        torch.manual_seed(self.parent.config['seed'])
        np.random.seed(self.parent.config['seed'])
        event_logger = EventLogger(log_file_name=self.parent.log_name)
        model = self.parent.create_model()
        parameters = [p.detach().to('cpu') for p in model.parameters()]
        control = [torch.zeros_like(p) for p in parameters]  # y^(-1) = 0.
        self.parent.init_process(server_rank, self.parent.size, 'gloo',
                                 self.parent.ports[0], self.parent.group_names[0])
        current_round = 0
        training_end_time = time.time() + self.parent.train_time * 60
        try:
            while self.parent.keep_training(current_round, training_end_time):
                participating = self.select_participating_clients(current_round, self.num_clients)
                logging.info('[FedSUM-CR Server] Round %s, participants: %s',
                             current_round, participating)
                # Use the existing notification + one-model downlink protocol.
                self.send_to_clients(participating, parameters, 'cpu', event_logger,
                                     current_round, server_rank)
                self.aggregate(parameters, control, self._receive_updates(participating, parameters))
                for parameter, shared in zip(parameters, shared_parameter_arrays):
                    np.copyto(np.frombuffer(shared.get_obj(), dtype=np.float32)
                              .reshape(parameter.shape), parameter.float().cpu().numpy())
                current_round += 1
                dist.barrier()
            for client_rank in range(1, self.parent.size):
                dist.send(tensor=torch.tensor(-10, dtype=torch.int32), dst=client_rank)
            dist.barrier()
        finally:
            dist.destroy_process_group()
        logging.info('[FedSUM-CR Server] Finished %s rounds', current_round)

    def client_process(self, client_rank):
        torch.manual_seed(self.parent.config['seed'] + client_rank)
        np.random.seed(self.parent.config['seed'] + client_rank)
        device = torch.device(f'cuda:{self.parent.local_rank}'
                              if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda':
            torch.cuda.set_device(device)
        event_logger = EventLogger(log_file_name=self.parent.log_name)
        task = self.parent.configure_task(client_rank, device)
        parameters, state = task.initialize(self.parent.config['seed'])
        base_optimizer = configure_base_optimizer(self.parent.config)
        base_optimizer_state = base_optimizer.init(parameters)
        memory = _ClientMemory(parameters)
        batches = task.data.iterator(batch_size=self.parent.config['batch_size'], shuffle=True)
        shapes = [p.shape for p in parameters]
        buffer = torch.empty_like(pack(parameters), device='cpu')
        self.parent.init_process(client_rank, self.parent.size, 'gloo',
                                 self.parent.ports[0], self.parent.group_names[0])
        current_round = 0
        try:
            while True:
                notification = torch.empty((), dtype=torch.int32)
                dist.recv(tensor=notification, src=0)
                if notification.item() == -10:
                    break
                if notification.item() == 1:
                    dist.recv(tensor=buffer, src=0)
                    with torch.no_grad():
                        for parameter, value in zip(parameters, unpack(buffer, shapes)):
                            parameter.copy_(value.to(device))
                    event_logger.log_start('local sgd')
                    epoch, state, delta = self.local_update(
                        task, parameters, state, batches, memory, current_round,
                        base_optimizer, base_optimizer_state)
                    event_logger.log_end('local sgd', {
                        'rank': client_rank, 'iteration': self.local_steps, 'epoch': epoch})
                    event_logger.log_start('communication')
                    update_buffer = pack([value.to('cpu') for value in delta])
                    dist.send(tensor=update_buffer, dst=0)
                    event_logger.log_end('communication', {
                        'from': client_rank, 'to': 0, 'bytes_sent': num_bytes(update_buffer)})
                elif notification.item() != 0:
                    raise RuntimeError(f'Unexpected FedSUM-CR notification: {notification.item()}')
                # Advance on skipped rounds too; a_i only advances when training.
                current_round += 1
                dist.barrier()
            dist.barrier()
        finally:
            dist.destroy_process_group()

    def run(self, rank):
        if rank != 0:
            self.client_process(rank)
            return

        model = self.parent.create_model()
        shared_arrays = [Array('f', p.numel(), lock=True) for p in model.parameters()]
        for parameter, shared in zip(model.parameters(), shared_arrays):
            np.copyto(np.frombuffer(shared.get_obj(), dtype=np.float32).reshape(parameter.shape),
                      parameter.detach().float().cpu().numpy())
        active = Value('i', 1)
        evaluator = Process(target=self.parent.evaluation_process,
                            args=(self.parent.eval_gpu, shared_arrays, None, active, None))
        server = Process(target=self.server_process, args=(rank, shared_arrays))
        processes = (server, evaluator)
        try:
            server.start()
            evaluator.start()
            while server.is_alive():
                server.join(timeout=0.1)
                if evaluator.exitcode is not None:
                    raise RuntimeError('FedSUM-CR evaluator exited before training finished '
                                       f'(exit code {evaluator.exitcode}).')
            if server.exitcode != 0:
                raise RuntimeError(f'FedSUM-CR server failed (exit code {server.exitcode}).')
        except BaseException:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            raise
        finally:
            active.value = 0
            for process in processes:
                if process.pid is not None:
                    process.join()
        if evaluator.exitcode != 0:
            raise RuntimeError(f'FedSUM-CR evaluator failed (exit code {evaluator.exitcode}).')
