import os
import contextlib
import itertools
import re
from typing import Iterable, List, Tuple
import torch
from torch.random import fork_rng
import torchvision
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset, random_split
from .utils.non_iid_dirichlet import distribute_data_dirichlet


from .api import Batch, Dataset, Gradient, Loss, Parameters, Quality, State, Task
from .cifar import PyTorchDataset, parameter_type, fork_rng_with_seed

from transformers import AutoTokenizer
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO)

class MNLITask(Task):
    def __init__(
            self, device, rank, num_workers, weight_decay, model_name, data_split_method, train_eval_frac, lock, non_iid_alpha=None, seed=0
    ):
        logging.info("MNLI Begin")
        self._device = device
        self._model_name = model_name
        self._model = self._create_model()
        logging.info("MNLI model creatd!")
        self.tokenizer = self._creat_tokenizer()
        logging.info("MNLI tokenizer creatd!")
        self._criterion = torch.nn.CrossEntropyLoss(reduction='mean').to(self._device)
        logging.info("MNLI criterion creatd!")
        self._weight_decay_per_param = [
            0 if parameter_type(p) == "batch_norm" else weight_decay
            for p, _ in self._model.named_parameters()
        ]
        logging.info("MNLI weight_decay creatd!")
        self.data = MNLIDataset("train", lock, self.tokenizer, device=self._device)
        self.max_batch_size = self.data.max_batch_size
        if rank > -1:
            if num_workers > 1:
                # Splitting data by worker
                if data_split_method == "dirichlet":
                    splits = self.data.dirichlet_split(
                        num_workers, non_iid_alpha, seed=seed
                    )
                elif data_split_method == "random":
                    splits = self.data.random_split(
                        fractions=[1 / num_workers for _ in range(num_workers)], seed=seed
                    )
                else:
                    raise ValueError(
                        f"Unknown value {data_split_method} for data_split_method"
                    )
                self.mean_num_data_per_worker = (
                        sum(len(split) for split in splits) / num_workers
                )
                print(
                    f"Splitting data using {data_split_method} according to",
                    [len(split) for split in splits],
                )
                self.data = splits[rank]
            else:
                self.mean_num_data_per_worker = len(self.data)
        else:
            splits = self.data.random_split(
                fractions=[train_eval_frac, 1-train_eval_frac], seed=seed+85
            )
            self.data = splits[0]
        self._test_data = MNLIDataset("test", lock, device=self._device)

    def initialize(self, seed) -> Tuple[Parameters]:
        with fork_rng_with_seed(seed):
            self._model = self._create_model()
        parameters = [p.data for p in self._model.parameters()]
        state = [b.data for b in self._model.buffers()]
        return parameters, state

    def loss(
            self,
            parameters: List[torch.Tensor],
            state: List[torch.Tensor],
            batch,
            random_seed=None,
    ) -> Tuple[Loss, State]:
        with torch.no_grad():
            with fork_rng_with_seed(random_seed):
                output, state = self.forward(
                    batch, parameters, state, is_training=True
                )
        loss = self._criterion(
                            output.logits[:, :-1, :].flatten(0, -2),
                            batch['input_ids'][:, 1:].flatten(),
                            )
        return loss.item(), state

    def loss_and_gradient(
            self,
            parameters: List[torch.Tensor],
            state: List[torch.Tensor],
            batch,
            random_seed=None,
    ) -> Tuple[Loss, Gradient, State]:
        with fork_rng_with_seed(random_seed):
            output, state = self._forward(batch, parameters, state, is_training=True)
        loss = self._criterion(
            output.logits[:, :-1, :].flatten(0, -2),
            batch['input_ids'][:, 1:].flatten(),
        )
        gradients = torch.autograd.grad(loss, list(self._model.parameters()))

        for g, wd, p in zip(gradients, self._weight_decay_per_param, parameters):
            g.add_(p, alpha=wd)

        return loss.item(), gradients, state

    def quality(
            self, parameters: List[torch.Tensor], state: List[torch.Tensor], batch: Batch
    ) -> Quality:
        """Average quality on the batch"""
        with torch.no_grad():
            output, _ = self._forward(batch, parameters, state, is_training=False)
        loss = self._criterion(
            output.logits[:, :-1, :].flatten(0, -2),
            batch['input_ids'][:, 1:].flatten(),
        )
        accuracy = loss
        return {"loss": loss.item(), "accuracy": accuracy.item()}

    def evaluate(
            self,
            dataset: Dataset,
            parameters: List[torch.Tensor],
            state: List[torch.Tensor],
    ) -> Quality:
        """Average quality on a dataset"""
        mean_quality = None
        count = 0
        for _, batch in dataset.iterator(batch_size=250, shuffle=False, repeat=False):
            quality = self.quality(parameters, state, batch)
            if mean_quality is None:
                count = len(batch)
                mean_quality = quality
            else:
                count += len(batch)
                weight = float(len(batch)) / count
                for key, value in mean_quality.items():
                    mean_quality[key] += weight * (quality[key] - mean_quality[key])
        return mean_quality

    def _forward(
            self,
            input,
            parameters: List[torch.Tensor],
            state: List[torch.Tensor],
            is_training=False,
            max_sequence_length = 128,
    ) -> Tuple[torch.Tensor, State]:
        if is_training:
            self._model.train()
        else:
            self._model.eval()

        for param, value in zip(self._model.parameters(), parameters):
            param.data = value

        for buffer, value in zip(self._model.buffers(), state):
            buffer.data = value

        output = self._model.forward(**input, )
        state = [b.data for b in self._model.buffers()]

        return output, state

    def _create_model(self):
        from .models.llm import LLM

        model = LLM(self._model_name)
        model.to(self._device)
        model.train()
        return model

    def _create_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        return tokenizer


class MNLIDataset(PyTorchDataset):

    max_batch_size = 128

    def __init__(
            self, split, lock, tokenizer, device="cuda"
    ):
        self.tokenizer = tokenizer
        with lock:
            dataset = load_dataset("multi_nli", split=split)

        dataset = dataset.map(
            self.form_training_prompts,
            remove_columns=["promptID", "pairID", "premise_binary_parse", "premise_parse",
                            "hypothesis_binary_parse", "hypothesis_parse", "hypothesis", "premise", "label"],
            load_from_cache_file=False,
            desc="Generating and tokenizing text prompts",
        )

        super().__init__(dataset, device=device)

    def dirichlet_split(
            self,
            num_workers: int,
            alpha: float = 1,
            seed: int = 0,
            distribute_evenly: bool = True,
    ) -> List[Dataset]:
        indices_per_worker = distribute_data_dirichlet(
            self._set['genre'], alpha, num_workers, num_auxiliary_workers=10, seed=seed
        )

        if distribute_evenly:
            indices_per_worker = np.array_split(
                np.concatenate(indices_per_worker), num_workers
            )

        return [
            PyTorchDataset(Subset(self._set, indices), self._device)
            for indices in indices_per_worker
        ]

    def prepare_batch(self, batch):
        batch = self.tokenizer(batch['text'],
                               truncation=True,
                               padding=True,
                               max_length=128,
                               return_tensors='pt')
        batch = {k: v.to(self._device) for k, v in batch.items()}
        return batch



def form_training_prompts(example):
    hypothesis = example["hypothesis"]
    premise = example["premise"]
    class_label = ["entailment", "neutral", "contradiction"][example["label"]]
    example[
        "text"
    ] = f"mnli hypothesis: {hypothesis} premise: {premise} target: {class_label}<|endoftext|>"
    genre_dict = {"government": 0, "fiction": 1, "travel": 2, "slate": 3, "telephone": 4, "letters": 5,
                  "verbatim": 6,
                  "facetoface": 7, "oup": 8, "nineeleven": 9, }
    example["genre"] = genre_dict[example["genre"]]
    return example



