import fcntl
import itertools
import os

import numpy as np
import torch
from datasets import ClassLabel, load_dataset
from torch.utils.data.dataset import Subset, random_split
from transformers import AutoTokenizer

from .api import Task
from .cifar import PyTorchDataset, fork_rng_with_seed, parameter_type
from .models.llama import Llama
from .utils.non_iid_dirichlet import distribute_data_dirichlet


class DollyTask(Task):
    def __init__(
            self, device, rank, num_workers, weight_decay, model_name,
            data_split_method, train_eval_frac, lock, non_iid_alpha=None, seed=0, training_ranks=None):
        self._device = device
        self._rank = rank
        self._model_name = model_name
        self._model = self._create_model()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self._criterion = torch.nn.CrossEntropyLoss().to(device)
        self._weight_decay_per_param = [
            0 if parameter_type(name) == 'batch_norm' else weight_decay
            for name, _ in self._model.named_parameters()
        ]
        self.data = DollyDataset('train', lock, self.tokenizer, device=device, seed=seed)
        self.max_batch_size = self.data.max_batch_size
        training_ranks = range(num_workers) if training_ranks is None else training_ranks
        num_clients = len(training_ranks)
        if rank in training_ranks and num_clients > 1:
            if data_split_method == 'dirichlet':
                splits = self.data.dirichlet_split(num_clients, non_iid_alpha, seed=seed)
            elif data_split_method == 'random':
                splits = self.data.random_split([1 / num_clients] * num_clients, seed=seed)
            else:
                raise ValueError(f'Unknown data_split_method: {data_split_method}')
            partitions_by_rank = dict(zip(training_ranks, splits))
            self.mean_num_data_per_worker = sum(map(len, splits)) / num_clients
            self.data = partitions_by_rank[rank]
            print(f'[Dolly rank={rank}] client partition size={len(self.data)}', flush=True)
        else:
            self.mean_num_data_per_worker = len(self.data)
            if rank < 0:
                self.data = self.data.random_split(
                    [train_eval_frac, 1 - train_eval_frac], seed=seed + 85)[0]
        self._test_data = (
            DollyDataset('test', lock, self.tokenizer, device=device, seed=seed)
            if rank < 0 else None)

    def initialize(self, seed):
        with fork_rng_with_seed(seed):
            self._model = self._create_model()
        return [p.data for p in self._model.parameters()], [b.data for b in self._model.buffers()]

    def _batch_loss(self, output, batch):
        return self._criterion(output.logits[:, :-1, :].float().flatten(0, -2),
                               batch['labels'][:, 1:].flatten())

    def loss(self, parameters, state, batch, random_seed=None):
        with torch.no_grad(), fork_rng_with_seed(random_seed):
            output, state = self._forward(batch, parameters, state, is_training=True)
            loss = self._batch_loss(output, batch)
        return loss.item(), state

    def loss_and_gradient(self, parameters, state, batch, random_seed=None):
        with fork_rng_with_seed(random_seed):
            output, state = self._forward(batch, parameters, state, is_training=True)
        loss = self._batch_loss(output, batch)
        gradients = torch.autograd.grad(loss, list(self._model.parameters()))
        for gradient, wd, parameter in zip(gradients, self._weight_decay_per_param, parameters):
            gradient.add_(parameter, alpha=wd)
        return loss.item(), gradients, state

    def quality(self, parameters, state, batch):
        with torch.no_grad():
            output, _ = self._forward(batch, parameters, state, is_training=False)
            loss = self._batch_loss(output, batch).item()
        # Preserve the accuracy field expected by the existing logs and plots.
        perplexity = float(np.exp(min(loss, 80)))
        return {'loss': loss, 'accuracy': perplexity, 'perplexity': perplexity}

    def evaluate(self, dataset, parameters, state):
        total_loss, total_tokens = 0.0, 0
        for _, batch in dataset.iterator(batch_size=2, shuffle=False, repeat=False):
            quality = self.quality(parameters, state, batch)
            tokens = int((batch['labels'][:, 1:] != -100).sum())
            total_loss += quality['loss'] * tokens
            total_tokens += tokens
        loss = total_loss / total_tokens
        perplexity = float(np.exp(min(loss, 80)))
        return {'loss': loss, 'accuracy': perplexity, 'perplexity': perplexity}

    def recalibrate_state(self, dataset, parameters, state):
        return state

    def _forward(self, batch, parameters, state, is_training=False):
        self._model.train(is_training)
        for parameter, value in zip(self._model.parameters(), parameters):
            parameter.data = value
        for buffer, value in zip(self._model.buffers(), state):
            buffer.data = value
        output = self._model(**{key: value for key, value in batch.items() if key != 'labels'})
        return output, [b.data for b in self._model.buffers()]

    def _create_model(self):
        if torch.device(self._device).type == 'cuda':
            torch.cuda.set_device(self._device)
        model = Llama(self._model_name).to(self._device)
        model.train()
        return model


class _DollyBatches(PyTorchDataset):
    """Include small clients and the final partial evaluation minibatch."""
    def iterator(self, batch_size, shuffle=True, repeat=True):
        loader = torch.utils.data.DataLoader(self._set, batch_size=batch_size,
                                            shuffle=shuffle, drop_last=False, num_workers=0)
        for epoch in itertools.count() if repeat else [0]:
            for index, batch in enumerate(loader):
                yield epoch + min((index + 1) * batch_size, len(self)) / len(self), self.prepare_batch(batch)


class DollyDataset(_DollyBatches):
    max_batch_size = 128

    def __init__(self, split, lock, tokenizer, device='cuda', seed=0):
        self.tokenizer = tokenizer
        os.makedirs('./data', exist_ok=True)
        with open(f'./data/dolly_{split}_map.lock', 'w') as flock_f:
            fcntl.flock(flock_f, fcntl.LOCK_EX)
            with lock:
                dataset = load_dataset('databricks/databricks-dolly-15k', split='train')
            dataset = dataset.cast_column('category', ClassLabel(names=sorted(set(dataset['category']))))
            dataset = dataset.train_test_split(test_size=0.1, seed=seed, stratify_by_column='category')[split]
            dataset = dataset.map(form_dolly_prompts, remove_columns=dataset.column_names,
                                  load_from_cache_file=True, desc='Generating Dolly prompts')
        super().__init__(dataset, device=device, prepare_batch=self.prepare_batch)

    def random_split(self, fractions, seed=0):
        lengths = [int(fraction * len(self)) for fraction in fractions]
        lengths[0] += len(self) - sum(lengths)
        return [_DollyBatches(split, self._device, prepare_batch=self.prepare_batch)
                for split in random_split(self._set, lengths, torch.Generator().manual_seed(seed))]

    def dirichlet_split(self, num_workers, alpha=1, seed=0, distribute_evenly=True):
        indices_per_worker = distribute_data_dirichlet(
            self._set['category'], alpha, num_workers, num_auxiliary_workers=10, seed=seed)
        if distribute_evenly:
            indices_per_worker = np.array_split(np.concatenate(indices_per_worker), num_workers)
        return [_DollyBatches(Subset(self._set, list(map(int, indices))), self._device,
                              prepare_batch=self.prepare_batch) for indices in indices_per_worker]

    def prepare_batch(self, batch):
        input_ids, labels = [], []
        # 1 BOS + 127 prompt + 127 response + 1 EOS = at most 256 tokens.
        for prompt, response in zip(batch['text'], batch['response']):
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)[-127:]
            response_ids = self.tokenizer.encode(response, add_special_tokens=False)[:127] + [self.tokenizer.eos_token_id]
            input_ids.append([self.tokenizer.bos_token_id] + prompt_ids + response_ids)
            labels.append([-100] * (1 + len(prompt_ids)) + response_ids)
        length = max(map(len, input_ids))
        masks = [[1] * len(ids) + [0] * (length - len(ids)) for ids in input_ids]
        labels = [row + [-100] * (length - len(row)) for row in labels]
        input_ids = [row + [self.tokenizer.pad_token_id] * (length - len(row)) for row in input_ids]
        return {key: torch.tensor(value, device=self._device) for key, value in
                [('input_ids', input_ids), ('attention_mask', masks), ('labels', labels)]}


def form_dolly_prompts(example):
    text = f"### Instruction:\n{example['instruction']}\n\n"
    if example['context']:
        text += f"### Context:\n{example['context']}\n\n"
    return {'text': text + '### Response:\n', 'response': example['response'], 'category': example['category']}
