"""Dirichlet regression checks that need NumPy, without Torch or dataset downloads."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import List
import unittest
from unittest.mock import patch

import numpy as np


spec = importlib.util.spec_from_file_location(
    'non_iid_dirichlet',
    Path(__file__).resolve().parents[1] / 'tasks/utils/non_iid_dirichlet.py')
partitioner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(partitioner)


class DirichletPartitioningTests(unittest.TestCase):
    def test_default_samples_in_groups_of_ten(self):
        # Record draw dimensions to check both full groups and leftover groups.
        targets = np.tile(np.arange(10), 100)
        for clients in [10, 11, 25, 50, 75, 100]:
            with self.subTest(clients=clients):
                dimensions = []

                class UniformRandomState:
                    def shuffle(self, values):
                        pass

                    def dirichlet(self, alpha):
                        dimensions.append(len(alpha))
                        return np.full(len(alpha), 1 / len(alpha))

                with patch.object(partitioner.np.random, 'RandomState',
                                  return_value=UniformRandomState()):
                    parts = partitioner.distribute_data_dirichlet(targets, .01, clients)
                self.assertEqual(len(parts), clients)
                self.assertTrue(dimensions)
                expected = {min(10, clients)}
                if clients % 10:
                    expected.add(clients % 10)
                self.assertEqual(set(dimensions), expected)
                np.testing.assert_array_equal(np.sort(np.concatenate(parts)), np.arange(len(targets)))

    def test_default_matches_explicit_groups_of_ten(self):
        targets = np.tile(np.arange(10), 100)
        for clients in [10, 11]:
            with self.subTest(clients=clients):
                implicit = partitioner.distribute_data_dirichlet(targets, .1, clients, seed=142)
                explicit = partitioner.distribute_data_dirichlet(
                    targets, .1, clients, seed=142, num_auxiliary_workers=10)
                self.assertEqual(implicit, explicit)

    def test_impossible_minimum_does_not_loop_forever(self):
        # This draw always assigns the only class to client 0, so client 1 can
        # never meet the rejection threshold. The wrapper can still rebalance.
        draws = []

        class SkewedRandomState:
            def shuffle(self, values):
                pass

            def dirichlet(self, alpha):
                if len(draws) >= 1000:
                    raise AssertionError('Unbounded rejection loop')
                draws.append(None)
                return np.array([1., 0.])

        with patch.object(partitioner.np.random, 'RandomState', return_value=SkewedRandomState()):
            parts = partitioner.distribute_data_dirichlet(np.zeros(100, dtype=int), .01, 2)
        self.assertLessEqual(len(draws), 1000)
        balanced = np.array_split(np.concatenate(parts), 2)
        self.assertEqual([len(part) for part in balanced], [50, 50])
        np.testing.assert_array_equal(np.sort(np.concatenate(balanced)), np.arange(100))

    def test_cifar_rebalances_empty_partitions_into_indexable_datasets(self):
        # Exercise the real dataset wrapper without importing Torch. A sparse
        # raw split promotes np.concatenate to float unless the wrapper casts.
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'tasks/cifar.py').read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == 'PyTorchDataset')
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                      and node.name == 'dirichlet_split')
        raw = [[0, 1, 2, 3], [], [], []]
        rows = np.arange(4)
        subset = lambda dataset, indices: [dataset[index] for index in indices]
        ns = dict(List=List, Dataset=object, np=np, Subset=subset,
                  distribute_data_dirichlet=lambda *args, **kwargs: raw,
                  PyTorchDataset=lambda data, device: data)
        exec(compile(ast.Module(body=[method], type_ignores=[]), 'CIFAR wrapper', 'exec'), ns)
        dataset = SimpleNamespace(_set=SimpleNamespace(targets=rows), _device='cpu')
        # The stand-in indexes a NumPy array just as torchvision indexes image
        # arrays; passing floating-point indices raises instead of hiding a bug.
        ns['Subset'] = lambda dataset, indices: subset(rows, indices)
        balanced = ns['dirichlet_split'](dataset, 4, alpha=.01)
        self.assertEqual(balanced, [[0], [1], [2], [3]])


if __name__ == '__main__':
    unittest.main()
