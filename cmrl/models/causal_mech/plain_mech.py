from typing import Optional, List, Dict, Union
import pathlib
import itertools
import copy

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import Adam
from omegaconf import DictConfig
from stable_baselines3.common.logger import Logger

from cmrl.types import Variable, ContinuousVariable, DiscreteVariable, BinaryVariable
from cmrl.models.networks.parallel_mlp import ParallelMLP
from cmrl.models.graphs.base_graph import BaseGraph
from cmrl.models.causal_mech.base_causal_mech import BaseCausalMech
from cmrl.models.networks.coder import VariableEncoder, VariableDecoder

from time import time


class PlainMech(BaseCausalMech):
    def __init__(
        self,
        name: str,
        # base causal-mech params
        input_variables: List[Variable],
        output_variables: List[Variable],
        node_dim: int,
        variable_encoders: Optional[Dict[str, VariableEncoder]],
        variable_decoders: Optional[Dict[str, VariableDecoder]],
        ensemble_num: int = 7,
        elite_num: int = 5,
        # network params
        deterministic: bool = False,
        hidden_dims: Optional[List[int]] = None,
        use_bias: bool = True,
        activation_fn_cfg: Optional[DictConfig] = None,
        # forward method
        residual: bool = True,
        multi_step: str = "none",
        # trainer
        optim_lr: float = 1e-4,
        optim_weight_decay: float = 1e-5,
        optim_eps: float = 1e-8,
        optim_encoder: bool = True,
        # logger
        logger: Optional[Logger] = None,
        # others
        device: Union[str, torch.device] = "cpu",
        **kwargs
    ):
        self.deterministic = deterministic
        self.hidden_dims = hidden_dims if hidden_dims is not None else [200] * 4
        self.use_bias = use_bias
        self.activation_fn_cfg = activation_fn_cfg

        if multi_step == "none":
            multi_step = "forward-euler 1"

        super(PlainMech, self).__init__(
            name=name,
            input_variables=input_variables,
            output_variables=output_variables,
            node_dim=node_dim,
            variable_encoders=variable_encoders,
            variable_decoders=variable_decoders,
            residual=residual,
            multi_step=multi_step,
            optim_lr=optim_lr,
            optim_weight_decay=optim_weight_decay,
            optim_eps=optim_eps,
            optim_encoder=optim_encoder,
            logger=logger,
            device=device,
            **kwargs
        )

    def build_network(self):
        self.network = ParallelMLP(
            input_dim=self.input_var_num * self.node_dim,
            output_dim=self.output_var_num * self.node_dim,
            hidden_dims=self.hidden_dims,
            use_bias=self.use_bias,
            extra_dims=[self.ensemble_num],
            activation_fn_cfg=self.activation_fn_cfg,
        ).to(self.device)

        parmas = [self.network.parameters()] + [decoder.parameters() for decoder in self.variable_decoders.values()]
        if self.optim_encoder:
            parmas.extend([encoder.parameters() for encoder in self.variable_encoders.values()])
        self.optim = Adam(itertools.chain(*parmas), lr=self.optim_lr, weight_decay=self.optim_weight_decay, eps=self.optim_eps)

    def build_graph(self):
        self.graph = None

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert list(inputs.keys()) == list(self.variable_encoders.keys())
        data_shape = list(inputs.values())[0].shape
        assert len(data_shape) == 3  # ensemble-num, batch-size, specific-dim
        ensemble, batch_size, specific_dim = data_shape
        assert ensemble == self.ensemble_num

        inputs_tensor = torch.zeros(ensemble, batch_size, self.input_var_num * self.node_dim).to(self.device)
        for i, var in enumerate(self.input_variables):
            out = self.variable_encoders[var.name](inputs[var.name].to(self.device))  # ensemble-num, batch-size, node-dim
            inputs_tensor[:, :, i * self.node_dim : (i + 1) * self.node_dim] = out

        if self.multi_step.startswith("forward-euler"):
            step_num = int(self.multi_step.split()[-1])
            output_tensor = None
            for step in range(step_num):
                if step > 0:
                    inputs_tensor = torch.concat(
                        [output_tensor, inputs_tensor[:, :, self.output_var_num * self.node_dim :]], dim=-1
                    )
                output_tensor = self.network(inputs_tensor)
                if self.residual:
                    output_tensor += inputs_tensor[:, :, : self.output_var_num * self.node_dim]
        else:
            raise NotImplementedError

        outputs = {}
        for i, var in enumerate(self.output_variables):
            hid = output_tensor[:, :, i * self.node_dim : (i + 1) * self.node_dim]
            out = self.variable_decoders[var.name](hid)
            outputs[var.name] = out

        return outputs

    def train(self, loader: DataLoader):
        """train for ensemble data

        Args:
            loader: train data-loader.

        Returns: tensor of train loss, with shape (ensemble-num, batch-size).

        """
        batch_loss_list = []
        for inputs, targets in loader:
            outputs = self.forward(inputs)
            loss = self.loss(outputs, targets)  # ensemble-num, batch-size, output-var-num

            self.optim.zero_grad()
            loss.mean().backward()
            self.optim.step()

            batch_loss_list.append(loss)
        return torch.cat(batch_loss_list, dim=-2).detach().cpu()

    def eval(self, loader: DataLoader):
        """evaluate for non-ensemble data

        Args:
            loader: valid data-loader.

        Returns: tensor of eval loss, with shape (batch-size).

        """
        batch_loss_list = []
        with torch.no_grad():
            for inputs, targets in loader:
                outputs = self.forward(inputs)
                loss = self.loss(outputs, targets)  # ensemble-num, batch-size, output-var-num

                batch_loss_list.append(loss)
        return torch.cat(batch_loss_list, dim=-2).detach().cpu()

    def learn(
        self,
        # loader
        train_loader: DataLoader,
        valid_loader: DataLoader,
        # model learning
        longest_epoch: int = -1,
        improvement_threshold: float = 0.01,
        patience: int = 5,
        work_dir: Optional[Union[str, pathlib.Path]] = None,
        **kwargs
    ):
        best_weights: Optional[Dict] = None
        epoch_iter = range(longest_epoch) if longest_epoch >= 0 else itertools.count()
        epochs_since_update = 0
        best_eval_loss = self.eval(valid_loader).mean(dim=(1, 2))

        for epoch in epoch_iter:
            train_loss = self.train(train_loader)
            eval_loss = self.eval(valid_loader).mean(dim=(1, 2))
            maybe_best_weights = self._maybe_get_best_weights(best_eval_loss, eval_loss, improvement_threshold)
            if maybe_best_weights:
                # best loss
                best_eval_loss = torch.minimum(best_eval_loss, eval_loss)
                best_weights = maybe_best_weights
                epochs_since_update = 0
            else:
                epochs_since_update += 1

            # log
            self.total_epoch += 1
            if self.logger is not None:
                self.logger.record("{}/epoch".format(self.name), epoch)
                self.logger.record("{}/train_dataset_size".format(self.name), len(train_loader.dataset))
                self.logger.record("{}/valid_dataset_size".format(self.name), len(valid_loader.dataset))
                self.logger.record("{}/train_loss".format(self.name), train_loss.mean().item())
                self.logger.record("{}/val_loss".format(self.name), eval_loss.mean().item())
                self.logger.record("{}/best_val_loss".format(self.name), best_eval_loss.mean().item())

                self.logger.dump(self.total_epoch)

            if patience and epochs_since_update >= patience:
                break

        # saving the best models:
        self._maybe_set_best_weights_and_elite(best_weights, best_eval_loss)

    def _maybe_get_best_weights(
        self,
        best_val_loss: torch.Tensor,
        val_loss: torch.Tensor,
        threshold: float = 0.01,
    ) -> Optional[Dict]:
        """Return the current model state dict  if the validation score improves.
        For ensembles, this checks the validation for each ensemble member separately.
        Copy from https://github.com/facebookresearch/mbrl-lib/blob/main/mbrl/models/model_trainer.py

        Args:
            best_val_score (tensor): the current best validation losses per model.
            val_score (tensor): the new validation loss per model.
            threshold (float): the threshold for relative improvement.
        Returns:
            (dict, optional): if the validation score's relative improvement over the
            best validation score is higher than the threshold, returns the state dictionary
            of the stored model, otherwise returns ``None``.
        """
        improvement = (best_val_loss - val_loss) / torch.abs(best_val_loss)
        if (improvement > threshold).any().item():
            best_weights = copy.deepcopy(self.network.state_dict())
        else:
            best_weights = None

        return best_weights

    def _maybe_set_best_weights_and_elite(self, best_weights: Optional[Dict], best_val_score: torch.Tensor):
        if best_weights is not None:
            self.network.load_state_dict(best_weights)

        sorted_indices = np.argsort(best_val_score.tolist())
        self.elite_indices = sorted_indices[: self.elite_num]
