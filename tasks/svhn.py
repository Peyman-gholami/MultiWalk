import logging
import tempfile
import urllib.request
from pathlib import Path
from typing import List, Tuple

import torch
import torchvision

from .api import Batch, Dataset, Gradient, Loss, Parameters, Quality, State, Task
from .cifar import PyTorchDataset, _file_md5, fork_rng_with_seed, parameter_type

# Filenames and MD5s match torchvision.datasets.SVHN.
SVHN_SPLITS = {
    "train": {
        "filename": "train_32x32.mat",
        "md5": "e26dedcc434d2e4c54c9b2d4a06d8373",
        "urls": (
            "http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
        ),
    },
    "test": {
        "filename": "test_32x32.mat",
        "md5": "eb5a983be6a315427106f1b164d9cef3",
        "urls": (
            "http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
        ),
    },
}


def ensure_svhn_data(data_root: str = "./data/", splits=("train", "test")) -> None:
    """Download SVHN .mat files if torchvision has not already fetched them."""
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)

    for split in splits:
        meta = SVHN_SPLITS[split]
        target = root / meta["filename"]
        if target.is_file() and _file_md5(target) == meta["md5"]:
            continue

        last_error = None
        for url in meta["urls"]:
            tmp_path = None
            try:
                logging.info("Downloading SVHN %s from %s", split, url)
                with tempfile.NamedTemporaryFile(delete=False, dir=root, suffix=".mat.part") as tmp:
                    tmp_path = Path(tmp.name)
                    with urllib.request.urlopen(url, timeout=120) as response:
                        while True:
                            chunk = response.read(1 << 20)
                            if not chunk:
                                break
                            tmp.write(chunk)
                if _file_md5(tmp_path) != meta["md5"]:
                    raise ValueError(f"MD5 mismatch for download from {url}")
                tmp_path.replace(target)
                logging.info("SVHN %s ready at %s", split, target)
                break
            except Exception as exc:
                last_error = exc
                logging.warning("SVHN %s download failed from %s: %s", split, url, exc)
                if tmp_path is not None and tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
        else:
            raise RuntimeError(f"Failed to download SVHN {split} from all mirrors: {last_error}")


class SVHNTask(Task):
    def __init__(
            self, device, rank, num_workers, weight_decay, model_name, data_split_method, train_eval_frac, lock, non_iid_alpha=None, seed=0
    ):
        self._device = device

        self.data = SVHNDataset("train", lock, device=self._device)
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
        self._test_data = SVHNDataset("test", lock, device=self._device)

        self._model_name = model_name
        self._model = self._create_model()
        self._criterion = torch.nn.CrossEntropyLoss().to(self._device)

        self._weight_decay_per_param = [
            0 if parameter_type(p) == "batch_norm" else weight_decay
            for p, _ in self._model.named_parameters()
        ]

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
            batch: Batch,
            random_seed=None,
    ) -> Tuple[Loss, State]:
        with torch.no_grad():
            with fork_rng_with_seed(random_seed):
                output, state = self.forward(
                    batch._y, parameters, state, is_training=True
                )
        loss = self._criterion(output, batch._y).item()
        return loss, state

    def loss_and_gradient(
            self,
            parameters: List[torch.Tensor],
            state: List[torch.Tensor],
            batch: Batch,
            random_seed=None,
    ) -> Tuple[Loss, Gradient, State]:
        with fork_rng_with_seed(random_seed):
            output, state = self._forward(batch._x, parameters, state, is_training=True)
        loss = self._criterion(output, batch._y)
        gradients = torch.autograd.grad(loss, list(self._model.parameters()))

        for g, wd, p in zip(gradients, self._weight_decay_per_param, parameters):
            g.add_(p, alpha=wd)

        return loss.item(), gradients, state

    def quality(
            self, parameters: List[torch.Tensor], state: List[torch.Tensor], batch: Batch
    ) -> Quality:
        """Average quality on the batch"""
        with torch.no_grad():
            output, _ = self._forward(batch._x, parameters, state, is_training=False)
        accuracy = torch.argmax(output, 1).eq(batch._y).sum().float() / len(batch)
        loss = self._criterion(output, batch._y)
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

    def recalibrate_state(
            self,
            dataset: Dataset,
            parameters: List[torch.Tensor],
            state: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Runs a forward pass on a random 10% of the dataset to update BatchNorm buffers."""

        current_state = state

        # Calculate the 10% threshold.
        # (Replace len(dataset) with however your custom Dataset exposes its total size)
        target_samples = int(0.10 * len(dataset))
        processed_samples = 0

        with torch.no_grad():
            # CRITICAL: shuffle=True ensures we get a different random 10% slice every time
            for _, batch in dataset.iterator(batch_size=250, shuffle=True, repeat=False):

                _, current_state = self._forward(
                    batch._x,
                    parameters,
                    current_state,
                    is_training=True
                )

                processed_samples += len(batch)

                # Break early to save computation once we hit the 10% limit
                if processed_samples >= target_samples:
                    break

        return current_state

    def _forward(
            self,
            input,
            parameters: List[torch.Tensor],
            state: List[torch.Tensor],
            is_training=False,
    ) -> Tuple[torch.Tensor, State]:
        if is_training:
            self._model.train()
        else:
            self._model.eval()

        for param, value in zip(self._model.parameters(), parameters):
            param.data = value

        for buffer, value in zip(self._model.buffers(), state):
            buffer.data = value

        output = self._model(input)
        state = [b.data for b in self._model.buffers()]

        return output, state

    def _create_model(self):
        if self._model_name == "ResNet20":
            from .models.resnet20 import ResNet20

            model = ResNet20(dataset="svhn")
            model.to(self._device)
            model.train()
        elif self._model_name == "VGG-11":
            from .models.vgg import vgg11

            model = vgg11()
            model.to(self._device)
            model.train()
        return model


class SVHNDataset(PyTorchDataset):
    data_mean = (0.4377, 0.4438, 0.4728)
    data_stddev = (0.1980, 0.2010, 0.1970)

    max_batch_size = 128

    def __init__(
            self, split, lock, data_root='./data/', device="cuda"
    ):
        if split == "train":
            # No horizontal flip: digit symmetry (e.g. 6/9) makes it harmful on SVHN.
            transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.RandomCrop(32, padding=4),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(self.data_mean, self.data_stddev),
                ]
            )
        elif split == "test":
            transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(self.data_mean, self.data_stddev),
                ]
            )
        else:
            raise ValueError(f"Unknown split '{split}'.")

        with lock:
            ensure_svhn_data(data_root, splits=(split,))
            dataset = torchvision.datasets.SVHN(
                root=data_root, split=split, download=True, transform=transform
            )
            # Dirichlet split expects `.targets` (CIFAR-style); SVHN only has `.labels`.
            dataset.targets = dataset.labels
        super().__init__(dataset, device=device)


def download(lock):
    ensure_svhn_data()
    SVHNDataset("train", lock)
