from typing import Any, List, Mapping, NamedTuple
import torch


def configure_base_optimizer(config) -> "BaseOptimizer":
    if config["base_optimizer"] == "SGD":
        return SGD(config)
    elif config["base_optimizer"] == "Adam":
        return Adam(config)
    else:
        raise ValueError("Unknown base optimizer {}".format(config["base_optimizer"]))


OptimizerState = Mapping[str, Any]


class BaseOptimizer:
    def __init__(self, config):
        self.config = config

    def init(self, parameters: List[torch.Tensor]) -> OptimizerState:
        raise NotImplementedError()

    def step(
        self,
        parameters: List[torch.Tensor],
        gradients: List[torch.Tensor],
        optimizer_state: OptimizerState,
        lr: float,
    ) -> None:
        """Updates parameters and optimizer_state in place"""
        raise NotImplementedError()

    def compute_updates(
        self,
        parameters: List[torch.Tensor],
        gradients: List[torch.Tensor],
        optimizer_state: OptimizerState,
        lr: float,
    ) -> List[torch.Tensor]:
        """Updates optimizer_state in place, but returns update instead of updating parameters"""
        prev_parameters = [p.clone() for p in parameters]
        self.step(parameters, gradients, optimizer_state, lr)
        updates = [p - prev for p, prev in zip(parameters, prev_parameters)]
        for p, prev in zip(parameters, prev_parameters):
            p.data = prev
        return updates


class SGD(BaseOptimizer):
    def init(self, parameters: List[torch.Tensor]) -> OptimizerState:
        return [
            torch.zeros_like(p, memory_format=torch.preserve_format) for p in parameters
        ]

    def step(
        self,
        parameters: List[torch.Tensor],
        gradients: List[torch.Tensor],
        optimizer_state: OptimizerState,
        lr: float,
    ) -> None:
        """Updates parameters and optimizer_state in place"""
        torch.optim._functional.sgd(
            parameters,
            gradients,
            optimizer_state,
            weight_decay=0.0,  # already taken care of in the task
            momentum=self.config["momentum"],
            lr=lr,
            dampening=0.0,
            nesterov=True,
            maximize=False,
        )

class AdamState(NamedTuple):
    exp_avgs: List[torch.Tensor]
    exp_avg_sqs: List[torch.Tensor]
    max_exp_avg_sqs: List[torch.Tensor]
    step: List[torch.Tensor]  # Updated to store tensors instead of integers


class Adam(BaseOptimizer):
    def init(self, parameters: List[torch.Tensor]) -> OptimizerState:
        return AdamState(
            # Initialize exp_avgs on the same device as parameters
            [
                torch.zeros_like(p, memory_format=torch.preserve_format, device='cuda')
                for p in parameters
            ],
            # Initialize exp_avg_sqs on the same device as parameters
            [
                torch.zeros_like(p, memory_format=torch.preserve_format, device='cuda')
                for p in parameters
            ],
            # Initialize max_exp_avg_sqs if required (empty list for now)
            [],
            # Initialize step as tensors on the same device as parameters
            [torch.tensor(0, dtype=torch.float32, device='cuda') for p in parameters],
        )

    def step(
        self,
        parameters: List[torch.Tensor],
        gradients: List[torch.Tensor],
        optimizer_state: OptimizerState,
        lr: float,
    ) -> None:
        """Updates parameters and optimizer_state in place"""
        # Increment step counts (tensors)
        for i in range(len(optimizer_state.step)):
            optimizer_state.step[i] += 1

        for param, grad, exp_avg, exp_avg_sq in zip(parameters, gradients, optimizer_state.exp_avgs,
                                                    optimizer_state.exp_avg_sqs):
            print(
                f"Param device: {param.device}, Grad device: {grad.device}, Exp Avg device: {exp_avg.device}, Exp Avg Sq device: {exp_avg_sq.device}")

        # Call torch.optim._functional.adam with updated step (singleton tensors)
        torch.optim._functional.adam(
            params=parameters,
            grads=gradients,
            exp_avgs=optimizer_state.exp_avgs,
            exp_avg_sqs=optimizer_state.exp_avg_sqs,
            max_exp_avg_sqs=optimizer_state.max_exp_avg_sqs,
            state_steps=optimizer_state.step,  # Now contains singleton tensors
            amsgrad=False,
            beta1=0.9,
            beta2=0.999,
            lr=lr,
            weight_decay=0.0,  # Already handled elsewhere
            eps=1e-8,
            maximize=False,
        )