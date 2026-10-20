"""Equation-level and real Gloo checks for Algorithm 3 of arXiv:2512.18275."""

import itertools
import json
import os
import subprocess
import sys
import time
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from algorithms.fedsum_cr import FedSUMCR, _ClientMemory
from base_optimizers import configure_base_optimizer
from distributed_training import DecentralizedTraining


INITIAL = np.array([0.7, -0.4, 0.2])
SCHEDULE = [[1, 2], [2], [], [3], [1], [1, 3], [2, 3]]


class LocalTrainingParent:
    def __init__(self, **settings):
        self.__dict__.update(settings)

    local_sgd = DecentralizedTraining.local_sgd
    learning_rate_schedule = DecentralizedTraining.learning_rate_schedule


def setup(num_clients=3, steps=3, **overrides):
    config = dict(learning_rate=0.3, global_learning_rate=0.4, base_optimizer='SGD',
                  momentum=0, lr_warmup_time=0, lr_schedule_milestones=[], seed=42,
                  batch_size=2, participation_pattern='uniform', participation_rate=1)
    config.update(overrides)
    return LocalTrainingParent(size=num_clients + 1, tau=steps, config=config)


def optimizer_for(parent, parameters):
    optimizer = configure_base_optimizer(parent.config)
    return optimizer, optimizer.init(parameters)


def reference_optimizer_states(clients):
    return {i: dict(velocity=np.zeros(3), mean=np.zeros(3), square=np.zeros(3), step=0)
            for i in clients}


def tensors(vector, dtype=torch.float64):
    return [torch.tensor(vector[:2], dtype=dtype), torch.tensor(vector[2], dtype=dtype)]


def vector(parameters):
    return torch.cat([p.detach().reshape(-1) for p in parameters]).cpu().numpy()


def batches():
    for index in itertools.count():
        yield float(index), index


class QuadraticTask:
    """Autograd gradients of heterogeneous quadratics with optional batch noise."""

    def __init__(self, client, noise=0, dtype=torch.float64):
        self.client = client
        self.noise = noise
        self.dtype = dtype
        self.data = self
        self._test_data = self

    def initialize(self, seed):
        return tensors(INITIAL, self.dtype), [torch.tensor(0)]

    def iterator(self, **kwargs):
        return batches()

    def loss_and_gradient(self, parameters, state, batch):
        x = torch.cat([p.detach().reshape(-1) for p in parameters]).requires_grad_()
        weights = x.new_tensor([1 + self.client / 10, 0.7, 1.3])
        target = x.new_tensor([self.client / 4, -self.client / 5, 0.1])
        batch_noise = self.noise * ((batch % 3) - 1)
        loss = (weights * (x - target).square()).sum() / 2 + batch_noise * x.sum()
        gradient, = torch.autograd.grad(loss, x)
        return loss.item(), [gradient[:2], gradient[2]], [state[0] + 1]

    def recalibrate_state(self, dataset, parameters, state):
        return state

    def evaluate(self, dataset, parameters, state):
        x = vector(parameters)
        return {'loss': float(np.square(x).mean()), 'accuracy': 0.0}


def reference_step(x, controls, last_round, history, selected, step, local_lr, global_lr,
                   num_clients, batch_counts, steps, noise, optimizer_states=None,
                   optimizer='SGD', momentum=0):
    """Independent reference using server history and direct optimizer equations.

    Production reconstructs the correction from model differences and derives h
    from local displacement. This reference uses neither of those calculations,
    nor the production optimizer or shared local_sgd implementation.
    """
    local_models = {}
    for client in selected:
        delay = step - last_round[client]
        between = history[max(0, last_round[client]):step]
        correction = sum(between, np.zeros_like(x)) / delay - controls[client]
        local = x.copy()
        directions = []
        for _ in range(steps):
            gradient = np.array([1 + client / 10, 0.7, 1.3]) * (
                local - np.array([client / 4, -client / 5, 0.1]))
            gradient += noise * ((batch_counts[client] % 3) - 1)
            batch_counts[client] += 1
            direction = gradient
            if optimizer_states is not None:
                state = optimizer_states[client]
                if optimizer == 'SGD':
                    state['velocity'] = momentum * state['velocity'] + gradient
                    # Repository SGD uses Nesterov momentum, including on step 1.
                    direction = gradient + momentum * state['velocity']
                else:
                    # Match the existing shared Adam: it increments the counter
                    # before functional Adam, which increments it again. Keep
                    # that behavior here to verify compatibility with HScaffold.
                    state['step'] += 2
                    state['mean'] = 0.9 * state['mean'] + 0.1 * gradient
                    state['square'] = 0.999 * state['square'] + 0.001 * gradient ** 2
                    mean = state['mean'] / (1 - 0.9 ** state['step'])
                    square = state['square'] / (1 - 0.999 ** state['step'])
                    direction = mean / (np.sqrt(square) + 1e-8)
            directions.append(direction.copy())
            local -= local_lr / num_clients * (direction + correction)
        controls[client] = np.mean(directions, axis=0)
        last_round[client] = step
        local_models[client] = local
    y = sum(controls.values(), np.zeros_like(x))
    return x - global_lr * local_lr * steps / num_clients * y, y, local_models


@pytest.mark.parametrize('num_clients', [1, 3, 10])
@pytest.mark.parametrize('steps', [1, 4])
@pytest.mark.parametrize('noise', [0, 0.05])
@pytest.mark.parametrize('optimizer, momentum', [('SGD', 0), ('SGD', 0.9), ('Adam', 0)])
def test_matches_independent_reference_with_staleness_and_empty_rounds(
        num_clients, steps, noise, optimizer, momentum):
    parent = setup(num_clients, steps, base_optimizer=optimizer, momentum=momentum)
    algorithm = FedSUMCR(parent)
    parameters = tensors(INITIAL)
    y = [torch.zeros_like(p) for p in parameters]
    clients = {i: (QuadraticTask(i, noise), tensors(INITIAL), [torch.tensor(0)],
                   batches(), _ClientMemory(parameters), *optimizer_for(parent, tensors(INITIAL)))
               for i in range(1, num_clients + 1)}
    controls = {i: np.zeros(3) for i in clients}
    last_round = {i: -1 for i in clients}
    batch_counts = {i: 0 for i in clients}
    optimizer_states = reference_optimizer_states(clients)
    x = INITIAL.copy()
    history = []
    schedule = SCHEDULE + [[num_clients], []]
    schedule = [[i for i in selected if i <= num_clients] for selected in schedule]
    for step, selected in enumerate(schedule):
        previous = vector(parameters).copy()
        before = {i: (vector(c[4].h).copy(), vector(c[4].z).copy(), c[4].a)
                  for i, c in clients.items()}
        updates = []
        new_states = {}
        for i in selected:
            task, local, state, stream, memory, base, base_state = clients[i]
            for local_p, global_p in zip(local, parameters):
                local_p.copy_(global_p)
            _, state, delta = algorithm.local_update(
                task, local, state, stream, memory, step, base, base_state)
            updates.append(delta)
            new_states[i] = state
        x, expected_y, expected_local = reference_step(
            x, controls, last_round, history, selected, step, parent.config['learning_rate'],
            parent.config['global_learning_rate'], num_clients, batch_counts, steps, noise,
            optimizer_states, optimizer, momentum)
        algorithm.aggregate(parameters, y, updates)
        history.append(expected_y.copy())
        np.testing.assert_allclose(vector(parameters), x, atol=2e-12, rtol=2e-12)
        np.testing.assert_allclose(vector(y), expected_y, atol=2e-12, rtol=2e-12)
        for i in clients:
            task, local, state, stream, memory, base, base_state = clients[i]
            if i in selected:
                np.testing.assert_allclose(vector(local), expected_local[i], atol=2e-12, rtol=2e-12)
                np.testing.assert_allclose(vector(memory.h), controls[i], atol=2e-12, rtol=2e-12)
                np.testing.assert_array_equal(vector(memory.z), previous)
                assert memory.a == step
                assert new_states[i][0].item() == batch_counts[i]
                clients[i] = (task, local, new_states[i], stream, memory, base, base_state)
            else:
                np.testing.assert_array_equal(vector(memory.h), before[i][0])
                np.testing.assert_array_equal(vector(memory.z), before[i][1])
                assert memory.a == before[i][2]


@pytest.mark.parametrize('num_clients', [1, 3, 10])
def test_one_local_step_full_participation_matches_centralized_gradient_descent(num_clients):
    algorithm = FedSUMCR(setup(num_clients, steps=1))
    parameters = tensors(INITIAL)
    y = [torch.zeros_like(p) for p in parameters]
    memories = [_ClientMemory(parameters) for _ in range(num_clients)]
    optimizers = [optimizer_for(algorithm.parent, parameters) for _ in range(num_clients)]
    expected = INITIAL.copy()
    for step in range(5):
        gradients = [np.array([1 + i / 10, 0.7, 1.3]) *
                     (expected - np.array([i / 4, -i / 5, 0.1]))
                     for i in range(1, num_clients + 1)]
        updates = []
        for client, memory in enumerate(memories, 1):
            local = [p.clone() for p in parameters]
            _, _, delta = algorithm.local_update(
                QuadraticTask(client), local, [torch.tensor(0)], batches(), memory, step,
                *optimizers[client - 1])
            updates.append(delta)
        expected -= 0.3 * 0.4 * np.mean(gradients, axis=0)
        algorithm.aggregate(parameters, y, updates)
        np.testing.assert_allclose(vector(parameters), expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('updates, message', [
    ({'learning_rate': 0}, 'learning_rate'),
    ({'learning_rate': float('nan')}, 'learning_rate'),
    ({'global_learning_rate': -1}, 'global_learning_rate'),
    ({'global_learning_rate': float('inf')}, 'global_learning_rate'),
    ({'lr_warmup_time': 2}, 'constant learning rates'),
    ({'lr_schedule_milestones': [(10, 0.1)]}, 'constant learning rates'),
])
def test_rejects_invalid_rates_and_schedules(updates, message):
    with pytest.raises(ValueError, match=message):
        FedSUMCR(setup(**updates))


@pytest.mark.parametrize('clients, steps', [(0, 1), (3, 0), (3, 1.5)])
def test_rejects_invalid_client_or_step_counts(clients, steps):
    with pytest.raises(ValueError):
        FedSUMCR(setup(clients, steps))


def test_zero_participation_is_not_replaced_by_a_forced_client():
    algorithm = FedSUMCR(setup(participation_rate=0))
    assert algorithm.select_participating_clients(0, 3) == []


def test_control_state_does_not_retain_autograd_graphs():
    algorithm = FedSUMCR(setup())
    parameters = [p.requires_grad_() for p in tensors(INITIAL)]
    memory = _ClientMemory(parameters)
    _, _, delta = algorithm.local_update(
        QuadraticTask(1), parameters, [torch.tensor(0)], batches(), memory, 0,
        *optimizer_for(algorithm.parent, parameters))
    assert all(not p.requires_grad and p.grad_fn is None for p in memory.h + memory.z + delta)


def test_dispatch_uses_the_new_algorithm(monkeypatch):
    import algorithms
    import distributed_training

    calls = []
    parent = SimpleNamespace(algorithm='fedsum_cr')
    monkeypatch.setattr(distributed_training, 'FedSUMCR',
                        lambda supplied: SimpleNamespace(run=lambda rank: calls.append((supplied, rank))))
    DecentralizedTraining.run(parent, rank=7)
    assert algorithms.FedSUMCR is FedSUMCR
    assert calls == [(parent, 7)]


def test_cli_lists_fedsum_cr():
    from pathlib import Path

    environment = dict(os.environ, LOCAL_RANK='0', WORLD_SIZE='4', RANK='0',
                       MASTER_ADDR='127.0.0.1')
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(root / 'main.py'), '--help'],
                            env=environment, cwd=root, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'fedsum_cr' in result.stdout


@pytest.mark.parametrize('failure', ['server', 'evaluator'])
def test_child_failure_is_reported_and_other_child_is_stopped(monkeypatch, failure):
    import algorithms.fedsum_cr as module

    processes = []

    class FakeProcess:
        def __init__(self, target, args):
            self.role = 'server' if target.__name__ == 'server_process' else 'evaluator'
            self.pid = None
            self.exitcode = None
            self.alive = False
            processes.append(self)

        def start(self):
            self.pid = len(processes)
            self.exitcode = 9 if self.role == failure else None
            self.alive = self.exitcode is None

        def is_alive(self):
            return self.alive

        def join(self, timeout=None):
            pass

        def terminate(self):
            self.alive = False
            self.exitcode = -15

    parent = setup()
    parent.create_model = lambda: torch.nn.Linear(2, 1)
    parent.evaluation_process = lambda *args: None
    parent.eval_gpu = 0
    monkeypatch.setattr(module, 'Process', FakeProcess)
    with pytest.raises(RuntimeError, match=f'FedSUM-CR {failure}'):
        FedSUMCR(parent).run(0)
    assert len(processes) == 2
    assert all(not process.is_alive() for process in processes)


class DistributedParent(LocalTrainingParent):
    def __init__(self, rank, store, output, momentum):
        self.__dict__.update(setup(momentum=momentum).__dict__)
        self.local_rank = 0
        self.ports = [0]
        self.group_names = ['fedsum-cr-test']
        self.train_time = 1
        self.log_name = f'rank{rank}.log'
        self.store = store
        self.output = output
        self.eval_gpu = 0
        self.evaluate_interval = 0.01
        self.no_test_set_eval = False

    def create_model(self):
        model = torch.nn.Module()
        model.register_parameter('vector', torch.nn.Parameter(tensors(INITIAL, torch.float32)[0]))
        model.register_parameter('scalar', torch.nn.Parameter(tensors(INITIAL, torch.float32)[1]))
        return model

    def configure_task(self, rank, device):
        return QuadraticTask(rank, dtype=torch.float32)

    def init_process(self, rank, size, backend, port, group_name):
        dist.init_process_group(backend, rank=rank, world_size=size,
                                init_method='file://' + self.store, timeout=timedelta(seconds=30))

    def keep_training(self, current_round, training_end_time):
        return current_round < len(SCHEDULE)

    evaluation_process = DecentralizedTraining.evaluation_process


class ScheduledFedSUMCR(FedSUMCR):
    def select_participating_clients(self, round_number, total_clients):
        return SCHEDULE[round_number]

    def server_process(self, server_rank, shared_arrays):
        super().server_process(server_rank, shared_arrays)
        values = np.concatenate([np.frombuffer(a.get_obj(), dtype=np.float32) for a in shared_arrays])
        with open(self.parent.output, 'w') as result:
            json.dump(values.tolist(), result)


def distributed_worker(rank, store, output, directory, momentum):
    os.chdir(directory)
    torch.set_num_threads(1)
    ScheduledFedSUMCR(DistributedParent(rank, store, output, momentum)).run(rank)


@pytest.mark.parametrize('momentum', [0, 0.9])
def test_real_gloo_server_clients_and_evaluator_match_reference(tmp_path, momentum):
    store = str(tmp_path / 'gloo-store')
    output = str(tmp_path / 'parameters.json')
    context = mp.get_context('spawn')
    workers = [context.Process(target=distributed_worker,
                               args=(i, store, output, str(tmp_path), momentum))
               for i in range(4)]
    try:
        for worker in workers:
            worker.start()
        deadline = time.monotonic() + 60
        for worker in workers:
            worker.join(timeout=max(0, deadline - time.monotonic()))
        assert not any(worker.is_alive() for worker in workers), 'Gloo protocol hung.'
        assert [worker.exitcode for worker in workers] == [0, 0, 0, 0]
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
        for worker in workers:
            if worker.pid is not None:
                worker.join()
    x = INITIAL.copy()
    controls = {i: np.zeros(3) for i in range(1, 4)}
    last_round = {i: -1 for i in controls}
    counts = {i: 0 for i in controls}
    history = []
    optimizer_states = reference_optimizer_states(controls)
    for step, selected in enumerate(SCHEDULE):
        x, y, _ = reference_step(x, controls, last_round, history, selected,
                                 step, 0.3, 0.4, 3, counts, 3, 0,
                                 optimizer_states, momentum=momentum)
        history.append(y.copy())
    np.testing.assert_allclose(json.loads((tmp_path / 'parameters.json').read_text()), x,
                               atol=2e-6, rtol=2e-6)
    # Exactly one parameter-sized tensor per direction, per active client.
    server_lines = (tmp_path / 'log' / 'rank0.log').read_text().splitlines()
    downlinks = [line for line in server_lines if line.startswith('communication,')]
    assert len(downlinks) == sum(map(len, SCHEDULE))
    assert all('bytes_sent: 12' in line for line in downlinks)
    for client in range(1, 4):
        lines = (tmp_path / 'log' / f'rank{client}.log').read_text().splitlines()
        uplinks = [line for line in lines if line.startswith('communication,')]
        assert len(uplinks) == sum(client in selected for selected in SCHEDULE)
        assert all('bytes_sent: 12' in line for line in uplinks)
