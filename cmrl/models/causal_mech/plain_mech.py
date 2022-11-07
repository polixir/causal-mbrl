from typing import Optional, List, Dict, Union, MutableMapping
import pathlib
import itertools
import copy

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from omegaconf import DictConfig
from hydra.utils import instantiate
from stable_baselines3.common.logger import Logger

from cmrl.utils.types import Variable
from cmrl.models.networks.parallel_mlp import ParallelMLP
from cmrl.models.causal_mech.base_causal_mech import BaseCausalMech
from cmrl.models.networks.coder import VariableEncoder, VariableDecoder


class PlainMech(BaseCausalMech):
    def __init__(
        self,
        name: str,
        input_variables: List[Variable],
        output_variables: List[Variable],
        ensemble_num: int = 7,
        elite_num: int = 5,
        # cfgs
        network_cfg: Optional[DictConfig] = None,
        encoder_cfg: Optional[DictConfig] = None,
        decoder_cfg: Optional[DictConfig] = None,
        optimizer_cfg: Optional[DictConfig] = None,
        # forward method
        residual: bool = True,
        encoder_reduction: str = "sum",
        multi_step: str = "none",
        # logger
        logger: Optional[Logger] = None,
        # others
        device: Union[str, torch.device] = "cpu",
        **kwargs
    ):
        if multi_step == "none":
            multi_step = "forward-euler 1"

        super(PlainMech, self).__init__(
            name=name,
            input_variables=input_variables,
            output_variables=output_variables,
            ensemble_num=ensemble_num,
            elite_num=elite_num,
            network_cfg=network_cfg,
            encoder_cfg=encoder_cfg,
            decoder_cfg=decoder_cfg,
            optimizer_cfg=optimizer_cfg,
            residual=residual,
            encoder_reduction=encoder_reduction,
            multi_step=multi_step,
            logger=logger,
            device=device,
            **kwargs
        )

    def build_network(self):
        self.network = instantiate(self.network_cfg)(
            input_dim=self.encoder_output_dim,
            output_dim=self.output_var_num * self.decoder_input_dim,
            extra_dims=[self.ensemble_num],
        ).to(self.device)

    def build_graph(self):
        self.graph = None

    def single_step_forward(self, inputs: MutableMapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert len(set(inputs.keys()) & set(self.variable_encoders.keys())) == len(inputs)
        data_shape = list(inputs.values())[0].shape
        assert len(data_shape) == 3, "{}".format(data_shape)  # ensemble-num, batch-size, specific-dim
        ensemble, batch_size, specific_dim = data_shape
        assert ensemble == self.ensemble_num

        inputs_tensor = torch.zeros(ensemble, batch_size, self.input_var_num, self.encoder_output_dim).to(self.device)
        for i, var in enumerate(self.input_variables):
            out = self.variable_encoders[var.name](inputs[var.name].to(self.device))
            inputs_tensor[:, :, i] = out
        output_tensor = self.network(self.reduce_encoder_output(inputs_tensor))

        outputs = {}
        for i, var in enumerate(self.output_variables):
            hid = output_tensor[:, :, i * self.decoder_input_dim : (i + 1) * self.decoder_input_dim]
            outputs[var.name] = self.variable_decoders[var.name](hid)

        if self.residual:
            for name in filter(lambda s: s.startswith("obs"), inputs.keys()):
                assert inputs[name].shape[:2] == outputs["next_{}".format(name)].shape[:2]
                assert inputs[name].shape[2] * 2 == outputs["next_{}".format(name)].shape[2]
                outputs["next_{}".format(name)][:, :, : inputs[name].shape[2]] += inputs[name].to(self.device)
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

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

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
