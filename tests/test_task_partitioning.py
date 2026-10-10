import importlib
from collections import Counter
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset as HFDataset

from distributed_training import DecentralizedTraining
from tasks.cifar import PyTorchDataset


TASKS = [
    ('Cifar', 'tasks.cifar', 'CifarTask', 'CifarDataset'),
    ('Cifar100', 'tasks.cifar100', 'Cifar100Task', 'Cifar100Dataset'),
    ('SVHN', 'tasks.svhn', 'SVHNTask', 'SVHNDataset'),
    ('MNLI', 'tasks.mnli', 'MNLITask', 'MNLIDataset'),
    ('Dolly', 'tasks.dolly', 'DollyTask', 'DollyDataset'),
]
SERVER_ALGORITHMS = [
    'fedavg', 'fedawe', 'fedau', 'fedprox', 'mifa',
    'scaffold', 'huscaffold', 'hscaffold', 'sgfocus',
]
PEER_ALGORITHMS = ['random_walk', 'split_random_walk', 'async_gossip', 'async_gossip_general']


def parent(task, algorithm, split='random', size=11):
    # These checks exercise task construction without starting distributed processes.
    return SimpleNamespace(
        size=size, algorithm=algorithm, train_eval_frac=0.25, lock=None,
        config=dict(task=task, model_name='fixture', weight_decay=0,
                    data_split_method=split, non_iid_alpha=0.5, seed=42))


@pytest.mark.parametrize('task,module_name,task_class,dataset_class', TASKS)
@pytest.mark.parametrize('algorithm', SERVER_ALGORITHMS + PEER_ALGORITHMS)
def test_factory_preserves_ranks_and_selects_training_ranks(
        monkeypatch, task, module_name, task_class, dataset_class, algorithm):
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, task_class, lambda **kwargs: kwargs)
    expected = list(range(1 if algorithm in SERVER_ALGORITHMS else 0, 11))
    for rank in [-1, 0, 1, 10]:
        kwargs = DecentralizedTraining.configure_task(parent(task, algorithm), rank, 'cpu')
        assert kwargs['rank'] == rank
        assert kwargs['num_workers'] == 11
        assert list(kwargs['training_ranks']) == expected


class Rows:
    def __init__(self, count):
        self.targets = [i % 7 for i in range(count)]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError(index)
        return int(index)


@pytest.mark.parametrize('task,module_name,task_class,dataset_class', TASKS)
@pytest.mark.parametrize('algorithm,size', [('fedavg', 11), ('random_walk', 11),
                                            ('fedavg', 2), ('random_walk', 1)])
@pytest.mark.parametrize('split', ['random', 'dirichlet'])
def test_all_training_samples_are_assigned_once(
        monkeypatch, task, module_name, task_class, dataset_class, algorithm, size, split):
    module = importlib.import_module(module_name)
    original_dataset = getattr(module, dataset_class)
    count = 1103  # Exercise uneven partition sizes as well as all 10/11 ranks.

    def fixture_dataset(*args, device='cpu', **kwargs):
        dataset = original_dataset.__new__(original_dataset)
        if task in ('MNLI', 'Dolly'):
            rows = HFDataset.from_dict(dict(
                id=list(range(count)), text=[str(i) for i in range(count)],
                genre=[i % 7 for i in range(count)], category=[i % 7 for i in range(count)]))
        else:
            rows = Rows(count)
        PyTorchDataset.__init__(dataset, rows, device=device)
        return dataset

    monkeypatch.setattr(module, dataset_class, fixture_dataset)
    cls = getattr(module, task_class)
    monkeypatch.setattr(cls, '_create_model', lambda self: torch.nn.Linear(1, 1))
    if task == 'MNLI':
        monkeypatch.setattr(cls, '_create_tokenizer', lambda self: None)
    elif task == 'Dolly':
        monkeypatch.setattr(module.AutoTokenizer, 'from_pretrained',
                            lambda *args, **kwargs: SimpleNamespace(eos_token='</s>'))

    setup = parent(task, algorithm, split, size=size)
    ranks = range(1 if algorithm in SERVER_ALGORITHMS else 0, setup.size)
    assigned = []
    lengths = []
    for rank in ranks:
        client = DecentralizedTraining.configure_task(setup, rank, 'cpu')
        assigned.extend(row['id'] if isinstance(row, dict) else row for row in client.data._set)
        lengths.append(len(client.data))
        assert client.mean_num_data_per_worker == pytest.approx(count / len(ranks))
    assert Counter(assigned) == Counter(range(count))
    assert max(lengths) - min(lengths) <= len(ranks)

    # Evaluation still samples the complete training set, independently of client partitions.
    evaluator = DecentralizedTraining.configure_task(setup, -1, 'cpu')
    assert len(evaluator.data) == count - int(0.75 * count)
    assert len(evaluator._test_data) == count
