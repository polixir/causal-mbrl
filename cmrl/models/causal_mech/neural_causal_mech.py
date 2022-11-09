from typing import Optional, List, Dict, Union, MutableMapping
from abc import abstractmethod, ABC
from itertools import chain, count
from functools import partial
import pathlib
import copy

import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import Optimizer
from stable_baselines3.common.logger import Logger
from omegaconf import DictConfig
from hydra.utils import instantiate

from cmrl.models.causal_mech.base_causal_mech import BaseCausalMech
from cmrl.models.networks.base_network import BaseNetwork
from cmrl.models.graphs.base_graph import BaseGraph
from cmrl.models.networks.coder import VariableEncoder, VariableDecoder
from cmrl.utils.variables import Variable, ContinuousVariable, DiscreteVariable, BinaryVariable
from cmrl.models.causal_mech.util import variable_loss_func, train_func, eval_func

default_network_cfg = DictConfig(
    dict(
        _target_="cmrl.models.networks.ParallelMLP",
        _partial_=True,
        _recursive_=False,
        hidden_dims=[200, 200],
        bias=True,
        activation_fn_cfg=dict(_target_="torch.nn.SiLU"),
    )
)

default_encoder_cfg = DictConfig(
    dict(
        _target_="cmrl.models.networks.VariableEncoder",
        _partial_=True,
        _recursive_=False,
        output_dim=100,
        hidden_dims=[100],
        bias=True,
        activation_fn_cfg=dict(_target_="torch.nn.SiLU"),
    )
)

default_decoder_cfg = DictConfig(
    dict(
        _target_="cmrl.models.networks.VariableDecoder",
        _partial_=True,
        _recursive_=False,
        input_dim=100,
        hidden_dims=[100],
        bias=True,
        identity=False,
        activation_fn_cfg=dict(_target_="torch.nn.SiLU"),
    )
)

default_optimizer_cfg = DictConfig(
    dict(
        _target_="torch.optim.Adam",
        _partial_=True,
        lr=1e-4,
        weight_decay=1e-5,
        eps=1e-8,
    )
)


class NeuralCausalMech(BaseCausalMech):
    def __init__(
        self,
        name: str,
        input_variables: List[Variable],
        output_variables: List[Variable],
        # ensemble
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
        super(NeuralCausalMech, self).__init__(
            name=name,
            input_variables=input_variables,
            output_variables=output_variables,
            device=device,
        )
        # ensemble
        self.ensemble_num = ensemble_num
        self.elite_num = elite_num
        # cfgs
        self.network_cfg = default_network_cfg if network_cfg is None else network_cfg
        self.encoder_cfg = default_encoder_cfg if encoder_cfg is None else encoder_cfg
        self.decoder_cfg = default_decoder_cfg if decoder_cfg is None else decoder_cfg
        self.optimizer_cfg = default_optimizer_cfg if optimizer_cfg is None else optimizer_cfg
        # forward method
        self.residual = residual
        self.encoder_reduction = encoder_reduction
        self.multi_step = multi_step
        # logger
        self.logger = logger

        # build member object
        self.variable_encoders: Optional[Dict[str, VariableEncoder]] = None
        self.variable_decoders: Optional[Dict[str, VariableEncoder]] = None
        self.network: Optional[BaseNetwork] = None
        self.graph: Optional[BaseGraph] = None
        self.optimizer: Optional[Optimizer] = None
        self.build_coder()
        self.build_network()
        self.build_graph()
        self.build_optimizer()

        self.total_epoch = 0
        self.elite_indices: List[int] = []

    def single_step_forward(self, inputs: MutableMapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size = self.get_inputs_batch_size(inputs)

        inputs_tensor = torch.zeros(self.ensemble_num, batch_size, self.input_var_num, self.encoder_output_dim).to(self.device)
        for i, var in enumerate(self.input_variables):
            out = self.variable_encoders[var.name](inputs[var.name].to(self.device))
            inputs_tensor[:, :, i] = out

        output_tensor = self.network(self.reduce_encoder_output(inputs_tensor))

        outputs = {}
        for i, var in enumerate(self.output_variables):
            hid = output_tensor[:, :, i * self.decoder_input_dim : (i + 1) * self.decoder_input_dim]
            outputs[var.name] = self.variable_decoders[var.name](hid)

        if self.residual:
            outputs = self.residual_outputs(inputs, outputs)
        return outputs

    def forward(self, inputs: MutableMapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.multi_step.startswith("forward-euler"):
            step_num = int(self.multi_step.split()[-1])

            outputs = {}
            for step in range(step_num):
                outputs = self.single_step_forward(inputs)
                inputs = {}
                if step < step_num - 1:
                    for name in filter(lambda s: s.startswith("obs"), inputs.keys()):
                        assert inputs[name].shape[:2] == outputs["next_{}".format(name)].shape[:2]
                        assert inputs[name].shape[2] * 2 == outputs["next_{}".format(name)].shape[2]
                        inputs[name] = outputs["next_{}".format(name)][:, :, : inputs[name].shape[2]]
        else:
            raise NotImplementedError("multi-step method {} is not supported".format(self.multi_step))

        return outputs

    @abstractmethod
    def build_network(self):
        raise NotImplementedError

    def build_optimizer(self):
        assert self.network is not None, "you must build network first"
        params = (
            [self.network.parameters()]
            + [encoder.parameters() for encoder in self.variable_encoders.values()]
            + [decoder.parameters() for decoder in self.variable_decoders.values()]
        )

        self.optimizer = instantiate(self.optimizer_cfg)(params=chain(*params))

    @abstractmethod
    def build_graph(self):
        raise NotImplementedError

    def build_coder(self):
        self.variable_encoders = {}
        for var in self.input_variables:
            assert var.name not in self.variable_encoders, "duplicate name in encoders: {}".format(var.name)
            self.variable_encoders[var.name] = instantiate(self.encoder_cfg)(variable=var).to(self.device)

        assert self.decoder_input_dim

        self.variable_decoders = {}
        for var in self.output_variables:
            assert var.name not in self.variable_decoders, "duplicate name in decoders: {}".format(var.name)
            self.variable_decoders[var.name] = instantiate(self.decoder_cfg)(variable=var).to(self.device)

    def get_inputs_batch_size(self, inputs: MutableMapping[str, torch.Tensor]) -> int:
        assert len(set(inputs.keys()) & set(self.variable_encoders.keys())) == len(inputs)
        data_shape = list(inputs.values())[0].shape
        # assert len(data_shape) == 3, "{}".format(data_shape)  # ensemble-num, batch-size, specific-dim
        ensemble, batch_size, specific_dim = data_shape[-3:]
        assert ensemble == self.ensemble_num

        return batch_size, data_shape[:-3]

    def residual_outputs(
        self,
        inputs: MutableMapping[str, torch.Tensor],
        outputs: MutableMapping[str, torch.Tensor],
    ) -> MutableMapping[str, torch.Tensor]:
        for name in filter(lambda s: s.startswith("obs"), inputs.keys()):
            # assert inputs[name].shape[:2] == outputs["next_{}".format(name)].shape[:2]
            # assert inputs[name].shape[2] * 2 == outputs["next_{}".format(name)].shape[2]
            var_dim = inputs[name].shape[-1]
            outputs["next_{}".format(name)][..., :var_dim] += inputs[name].to(self.device)
        return outputs

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
        epoch_iter = range(longest_epoch) if longest_epoch >= 0 else count()
        epochs_since_update = 0

        loss_func = partial(variable_loss_func, output_variables=self.output_variables, device=self.device)
        train = partial(train_func, forward=self.forward, optimizer=self.optimizer, loss_func=loss_func)
        eval = partial(eval_func, forward=self.forward, loss_func=loss_func)

        best_eval_loss = eval(valid_loader).mean(dim=(-2, -1))

        for epoch in epoch_iter:
            train_loss = train(train_loader)
            eval_loss = eval(valid_loader)
            maybe_best_weights = self._maybe_get_best_weights(
                best_eval_loss, eval_loss.mean(dim=(-2, -1)), improvement_threshold
            )
            if maybe_best_weights:
                # best loss
                best_eval_loss = torch.minimum(best_eval_loss, eval_loss.mean(dim=(-2, -1)))
                best_weights = maybe_best_weights
                epochs_since_update = 0
            else:
                epochs_since_update += 1

            # log
            self.total_epoch += 1
            if self.logger is not None:
                self.logger.record("{}/epoch".format(self.name), epoch)
                self.logger.record("{}/epochs_since_update".format(self.name), epochs_since_update)
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

    @property
    def encoder_output_dim(self):
        return self.encoder_cfg.output_dim

    @property
    def union_output_var_dim(self):
        # all output variables should be ContinuousVariable and have same variable.dim
        output_dim = []
        for var in self.output_variables:
            assert isinstance(var, ContinuousVariable), "all output variables should be ContinuousVariable"
            output_dim.append(var.dim)
        assert len(set(output_dim)) == 1, "all output variables should have same variable.dim"
        return output_dim[0]

    @property
    def decoder_input_dim(self):
        if self.decoder_cfg.identity:
            return self.union_output_var_dim * 2
        else:
            return self.decoder_cfg.input_dim

    def reduce_encoder_output(
        self,
        encoder_output: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert len(encoder_output.shape) == 4, (
            "shape of encoder_output should be (ensemble-num, batch-size, input-var-num, encoder-output-dim), "
            "rather than {}".format(encoder_output.shape)
        )

        if mask is None:
            mask = self.forward_mask

        assert mask.shape[-1] == self.input_var_num

        # [..., ensemble_num, batch_size, input_var_num, encoder_output_dim]
        mask = mask.unsqueeze(-1).unsqueeze(-3).unsqueeze(-4)
        repeat_shape = (*mask.shape[:-4], *encoder_output.shape[:-2], 1, encoder_output.shape[-1])
        # mask = mask.repeat((*mask.shape[:-4], *encoder_output.shape[:-2], 1, encoder_output.shape[-1]))

        mask = mask.repeat((1,) * len(mask.shape[:-4]) + (*encoder_output.shape[:-2], 1, encoder_output.shape[-1]))

        # [*mask_extra_dims, ensemble_num, batch_size, input_var_num, encoder_output_dim]
        masked_encoder_output = encoder_output.repeat(tuple(mask.shape[:-4]) + (1,) * 4)

        # choose mask value
        mask_value = 0
        if self.encoder_reduction == "max":
            mask_value = -float("inf")
        assert masked_encoder_output.shape == mask.shape
        # print(torch.unique(mask, return_counts=True))

        masked_encoder_output[mask == 0] = mask_value

        if self.encoder_reduction == "sum":
            return masked_encoder_output.sum(-2)
        elif self.encoder_reduction == "mean":
            return masked_encoder_output.mean(-2)
        elif self.encoder_reduction == "max":
            values, indices = masked_encoder_output.max(-2)
            return values
        else:
            raise NotImplementedError("not implemented encoder reduction method: {}".format(self.encoder_reduction))
