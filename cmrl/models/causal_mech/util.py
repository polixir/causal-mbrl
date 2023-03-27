from typing import Callable, Dict, List, Union, MutableMapping
from collections import defaultdict
import math
import time

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.distributions.von_mises import _log_modified_bessel_fn

from cmrl.utils.variables import Variable, ContinuousVariable, DiscreteVariable, BinaryVariable, RadianVariable


def von_mises_nll_loss(
        input: Tensor,
        target: Tensor,
        var: Tensor,
        full: bool = False,
        eps: float = 1e-6,
        reduction: str = "mean",
) -> Tensor:
    r"""Von Mises negative log likelihood loss.

    Args:
        input: loc of the Von Mises distribution.
        target: sample from the Von Mises distribution.
        var: tensor of positive var(s), one for each of the expectations
            in the input (heteroscedastic), or a single one (homoscedastic).
        full (bool, optional): include the constant term in the loss calculation. Default: ``False``.
        eps (float, optional): value added to var, for stability. Default: 1e-6.
        reduction (string, optional): specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the output is the average of all batch member losses,
            ``'sum'``: the output is the sum of all batch member losses.
            Default: ``'mean'``.
    """
    # Entries of var must be non-negative
    if torch.any(var < 0):
        raise ValueError("var has negative entry/entries")

    # Clamp for stability
    var = var.clone()
    with torch.no_grad():
        var.clamp_(min=eps)

    concentration = 1 / var
    loss = -concentration * torch.cos(input - target) + _log_modified_bessel_fn(concentration, order=0)
    if full:
        loss += math.log(2 * math.pi)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


def circular_gaussian_nll_loss(
        input: Tensor,
        target: Tensor,
        var: Tensor,
        full: bool = False,
        eps: float = 1e-6,
        reduction: str = "mean",
) -> Tensor:
    # Entries of var must be non-negative
    if torch.any(var < 0):
        raise ValueError("var has negative entry/entries")

    # Clamp for stability
    var = var.clone()
    with torch.no_grad():
        var.clamp_(min=eps)

    diff = torch.remainder(input - target, 2 * torch.pi)
    diff[diff > torch.pi] = 2 * torch.pi - diff[diff > torch.pi]
    loss = 0.5 * (torch.log(var) + diff ** 2 / var)
    if full:
        loss += 0.5 * math.log(2 * math.pi)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


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
            loss = F.gaussian_nll_loss(mean, target, log_var.exp(), reduction="none", full=True).mean(dim=-1)
            total_loss[..., i] = loss
        elif isinstance(var, RadianVariable):
            dim = target.shape[-1]  # (xxx, ensemble-num, batch-size, dim)
            assert output.shape[-1] == 2 * dim
            mean, log_var = output[..., :dim], output[..., dim:]
            loss = circular_gaussian_nll_loss(mean, target, log_var.exp(), reduction="none").mean(dim=-1)
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
