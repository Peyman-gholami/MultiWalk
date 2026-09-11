"""Simple CNN architectures from common FL CIFAR/SVHN setups.

CIFAR-10: C(3,32)-R-M-C(32,32)-R-M-L(256)-R-L(64)-R-L(10)
SVHN:     C(3,32)-R-M-C(32,32)-R-M-L(128)-R-L(10)

C(in, out): Conv2d(kernel=3, stride=1, padding=1)
R: ReLU
M: MaxPool2d(kernel=2, stride=2)
L(out): Linear
"""

from torch import nn

__all__ = ["SimpleCNN", "get_simple_cnn_separation_point"]


class SimpleCNN(nn.Module):
    def __init__(self, dataset="cifar10"):
        super().__init__()
        if dataset not in ("cifar10", "svhn"):
            raise ValueError(f"Unsupported SimpleCNN dataset '{dataset}'")

        self.dataset = dataset
        # 32x32 -> 16x16 -> 8x8 with two max-pools; 8*8*32 = 2048.
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        if dataset == "cifar10":
            self.classifier = nn.Sequential(
                nn.Linear(2048, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 10),
            )
            # Final classification layer (used for split-RW separation).
            self.fc = self.classifier[-1]
        else:
            self.classifier = nn.Sequential(
                nn.Linear(2048, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 10),
            )
            self.fc = self.classifier[-1]

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def get_simple_cnn_separation_point(model: nn.Module) -> int:
    """
    Return the separation index before the first classifier (Linear) parameter,
    matching the ResNet convention used by split random walk.
    """
    if not hasattr(model, "classifier"):
        raise ValueError("Model does not have a `classifier` attribute")

    classifier_param_ids = {id(p) for p in model.classifier.parameters()}
    first_classifier_idx = None
    for idx, p in enumerate(model.parameters()):
        if id(p) in classifier_param_ids:
            first_classifier_idx = idx
            break

    if first_classifier_idx is None:
        raise ValueError("Could not locate classifier parameters within model.parameters() order")

    return max(0, first_classifier_idx - 1)
