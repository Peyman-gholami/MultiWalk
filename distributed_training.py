import logging
import time

import torch
import torch.distributed as dist
from torch.multiprocessing import Lock

from algorithms import (
    AsyncGossip,
    FedAVG,
    FedProx,
    HScaffold,
    HUScaffold,
    MIFA,
    RandomWalk,
    SGFocus,
    Scaffold,
    SplitRandomWalk,
)
from logger import EventLogger
from tasks.api import Task
from tasks.cifar import fork_rng_with_seed
from utils.tools import load_from_shared

logging.basicConfig(level=logging.CRITICAL)
# logging.basicConfig(level=logging.INFO)


class DecentralizedTraining:
    def __init__(self, size, local_rank, tau, train_time, neighbors, top_nodes, rw_starting_ranks, master_address, ports, group_names, algorithm, config, evaluate_interval,
                 eval_gpu, train_eval_frac, no_test_set_eval, log_name, failure_times=None):
        self.size = size
        self.local_rank = local_rank
        self.tau = tau
        self.train_time = train_time
        self.neighbors = neighbors
        self.top_nodes = top_nodes
        self.rw_starting_ranks = rw_starting_ranks
        self.master_address = master_address
        self.ports = ports
        self.group_names = group_names
        self.algorithm = algorithm
        self.config = config
        self.lock = Lock()
        self.lock_comm_send = Lock()
        self.lock_comm_recv = Lock()
        self.evaluate_interval = evaluate_interval
        self.eval_gpu = eval_gpu
        self.train_eval_frac = train_eval_frac
        self.no_test_set_eval = no_test_set_eval
        self.log_name = log_name
        self.failure_times = failure_times if failure_times is not None else []

    def init_process(self, rank, size, backend, port, group_name):
        init_method = f'tcp://{self.master_address}:{port}'
        logging.info(f"Before [{group_name}] Rank {rank} initialized with backend {backend} on port {port}")
        dist.init_process_group(backend, rank=rank, world_size=size, init_method=init_method)
        logging.info(f"[{group_name}] Rank {rank} initialized with backend {backend} on port {port}")

    def configure_task(self, rank, device) -> Task:
        if self.config["task"] == "Cifar":
            from tasks.cifar import CifarTask

            return CifarTask(
                rank=rank,
                device=device,
                num_workers=self.size,
                weight_decay=self.config["weight_decay"],
                model_name=self.config["model_name"],
                data_split_method=self.config["data_split_method"],
                train_eval_frac=self.train_eval_frac,
                lock=self.lock,
                non_iid_alpha=self.config["non_iid_alpha"],
                seed=self.config["seed"] + 100
            )
        if self.config["task"] == "MNLI":
            from tasks.mnli import MNLITask

            return MNLITask(
                rank=rank,
                device=device,
                num_workers=self.size,
                weight_decay=self.config["weight_decay"],
                model_name=self.config["model_name"],
                data_split_method=self.config["data_split_method"],
                train_eval_frac=self.train_eval_frac,
                lock=self.lock,
                non_iid_alpha=self.config["non_iid_alpha"],
                seed=self.config["seed"] + 100
            )

    def create_model(self):
        if self.config["model_name"] == "ResNet20":
            from tasks.models.resnet20 import ResNet20
            with fork_rng_with_seed(self.config["seed"]):
                model = ResNet20()
        elif "opt" in self.config["model_name"]:
            from tasks.models.llm import LLM

            with fork_rng_with_seed(self.config["seed"]):
                model = LLM(self.config["model_name"])
        return model

    def get_model_separation_point(self, model):
        if self.config["model_name"] == "ResNet20":
            from tasks.models.resnet20 import get_resnet_separation_point
            return get_resnet_separation_point(model)
        elif "opt" in self.config["model_name"]:
            from tasks.models.llm import get_resnet_separation_point
            return get_resnet_separation_point(model)
        return None

    def learning_rate_schedule(self, time):
        lr = 1.0
        if self.size > 1 and self.config["lr_warmup_time"] > 0:
            warmup_time = self.config["lr_warmup_time"] * 60
            max_factor = 1.0
            factor = 0 + (max_factor - 0) * min(time / warmup_time, 1.0)
            lr *= factor
        for (milestone, factor) in self.config["lr_schedule_milestones"]:
            if time >= milestone:
                lr *= factor
            else:
                return lr
        return lr

    def evaluation_process(self, gpu_id, shared_arrays, shared_state, eval_process_active, shared_array_index=None):
        logger = EventLogger(log_file_name=self.log_name)
        device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        task = self.configure_task(-1, device)
        parameters, state = task.initialize(self.config["seed"])

        while True:
            # Wait for activation (eval_process_active == 1) or termination (eval_process_active == 0)
            while eval_process_active.value == 2:
                time.sleep(0.1)  # Check periodically if activated

            # Check if we should terminate (eval_process_active == 0 means terminate)
            if eval_process_active.value == 0:
                break

            # eval_process_active == 1, so we're active - perform evaluation
            time.sleep(self.evaluate_interval)
            if shared_array_index is not None:
                extracted_params = shared_arrays[shared_array_index.value]
            else:
                extracted_params = shared_arrays

            parameters = load_from_shared(parameters, extracted_params, device)
            if shared_state is not None:
                state = load_from_shared(state, shared_state, device)
            else:
                state = task.recalibrate_state(task.data, parameters, state)

            logger.log_start("evaluation")
            train_loss = task.evaluate(task.data, parameters, state)
            logging.info(f"Evaluation at interval: Train loss: {train_loss}")
            test_loss = train_loss
            if not self.no_test_set_eval:
                test_loss = task.evaluate(task._test_data, parameters, state)
                logging.info(f"Evaluation at interval: Test loss: {test_loss}")
            logger.log_end("evaluation", {"test loss": test_loss, "train loss": train_loss})

    def local_sgd(self, task, parameters, state, base_optimizer, base_optimizer_state, batch_data_gen, time, local_steps):
        for _ in range(local_steps):
            epoch, batch = next(batch_data_gen)
            last_loss, gradients, state = task.loss_and_gradient(parameters, state, batch)
            base_optimizer.step(
                parameters,
                gradients,
                base_optimizer_state,
                lr=self.config["learning_rate"] * self.learning_rate_schedule(time),
            )
        return epoch, gradients

    def run(self, rank):
        if self.algorithm == 'random_walk':
            random_walk = RandomWalk(self)
            random_walk.run(rank)
        elif self.algorithm == 'async_gossip':
            async_gossip = AsyncGossip(self)
            async_gossip.run(rank)
        elif self.algorithm == 'split_random_walk':
            split_random_walk = SplitRandomWalk(self)
            split_random_walk.run(rank)
        elif self.algorithm == 'fedavg':
            fedavg = FedAVG(self)
            fedavg.run(rank)
        elif self.algorithm == 'fedprox':
            fedprox = FedProx(self)
            fedprox.run(rank)
        elif self.algorithm == 'mifa':
            mifa = MIFA(self)
            mifa.run(rank)
        elif self.algorithm == 'scaffold':
            scaffold = Scaffold(self)
            scaffold.run(rank)
        elif self.algorithm == 'huscaffold':
            huscaffold = HUScaffold(self)
            huscaffold.run(rank)
        elif self.algorithm == 'hscaffold':
            hscaffold = HScaffold(self)
            hscaffold.run(rank)
        elif self.algorithm == 'sgfocus':
            sgfocus = SGFocus(self)
            sgfocus.run(rank)
