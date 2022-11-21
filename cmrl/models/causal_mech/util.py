from typing import Callable, Dict, List, Union, MutableMapping
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from cmrl.utils.variables import Variable, ContinuousVariable, DiscreteVariable, BinaryVariable


def variable_loss_func(
    outputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    output_variables: List[Variable],
    device: Union[str, torch.device] = "cpu",
):
    dims = list(outputs.values())[0].shape[:-1]
    total_loss = torch.zeros(*dims, len(outputs)).to(device)

    for i, var in enumerate(output_variables):
        output = outputs[var.name]
        target = targets[var.name].to(device)
        if isinstance(var, ContinuousVariable):
            dim = target.shape[-1]  # (xxx, ensemble-num, batch-size, dim)
            assert output.shape[-1] == 2 * dim
            mean, log_var = output[..., :dim], output[..., dim:]
            loss = F.gaussian_nll_loss(mean, target, log_var.exp(), reduction="none").mean(dim=-1)
            total_loss[..., i] = loss
        elif isinstance(var, DiscreteVariable):
            # TODO: onehot to int?
            raise NotImplementedError
            total_loss[..., i] = F.cross_entropy(output, target, reduction="none")
        elif isinstance(var, BinaryVariable):
            total_loss[..., i] = F.binary_cross_entropy(output, target, reduction="none")
        else:
            raise NotImplementedError
    return total_loss


def train_func(
    loader: DataLoader,
    forward: Callable[[MutableMapping[str, torch.Tensor]], Dict[str, torch.Tensor]],
    optimizer: Optimizer,
    loss_func: Callable[[MutableMapping[str, torch.Tensor], MutableMapping[str, torch.Tensor]], torch.Tensor],
):
    """train for data

    Args:
        forward: forward function.
        loader: train data-loader.
        optimizer: Optimizer
        loss_func: loss function

    Returns: tensor of train loss, with shape (xxx, ensemble-num, batch-size).

    """
    batch_loss_list = []
    for inputs, targets in loader:
        outputs = forward(inputs)
        loss = loss_func(outputs, targets)  # ensemble-num, batch-size, output-var-num

        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

        batch_loss_list.append(loss)
    return torch.cat(batch_loss_list, dim=-2).detach().cpu()


def eval_func(
    loader: DataLoader,
    forward: Callable[[MutableMapping[str, torch.Tensor]], Dict[str, torch.Tensor]],
    loss_func: Callable[[MutableMapping[str, torch.Tensor], MutableMapping[str, torch.Tensor]], torch.Tensor],
):
    """evaluate for data

    Args:
        forward: forward function.
        loader: train data-loader.
        loss_func: loss function

    Returns: tensor of train loss, with shape (xxx, ensemble-num, batch-size).

    """
    batch_loss_list = []
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = forward(inputs)
            loss = loss_func(outputs, targets)  # ensemble-num, batch-size, output-var-num

            batch_loss_list.append(loss)
    return torch.cat(batch_loss_list, dim=-2).detach().cpu()
