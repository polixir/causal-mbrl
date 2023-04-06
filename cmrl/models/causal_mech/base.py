from typing import Optional, List, Dict, Union, MutableMapping
from abc import abstractmethod, ABC
from itertools import chain, count
import pathlib
from functools import partial
import copy
from multiprocessing import cpu_count

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from omegaconf import DictConfig
from stable_baselines3.common.logger import Logger
from hydra.utils import instantiate

from cmrl.models.graphs.base_graph import BaseGraph
from cmrl.models.graphs.binary_graph import BinaryGraph
from cmrl.utils.variables import Variable
from cmrl.models.constant import NETWORK_CFG, ENCODER_CFG, DECODER_CFG, OPTIMIZER_CFG, SCHEDULER_CFG
from cmrl.models.networks.base_network import BaseNetwork
from cmrl.models.graphs.base_graph import BaseGraph
from cmrl.models.networks.coder import VariableEncoder, VariableDecoder
from cmrl.utils.variables import Variable, ContinuousVariable, DiscreteVariable, BinaryVariable
from cmrl.models.causal_mech.util import variable_loss_func, train_func, eval_func
from cmrl.models.data_loader import EnsembleBufferDataset, collate_fn


class BaseCausalMech(ABC):
    """The base class of causal-mech learned by neural networks.
    Pay attention that the causal discovery maybe not realized through a neural way.
    """

    def __init__(
        self,
        name: str,
        input_variables: List[Variable],
        output_variables: List[Variable],
        logger: Optional[Logger] = None,
    ):
        self.name = name
        self.input_variables = input_variables
        self.output_variables = output_variables
        self.logger = logger

        self.input_variables_dict = dict([(v.name, v) for v in self.input_variables])
        self.output_variables_dict = dict([(v.name, v) for v in self.output_variables])

        self.input_var_num = len(self.input_variables)
        self.output_var_num = len(self.output_variables)
        self.graph: Optional[BaseGraph] = None

    @abstractmethod
    def learn(
        self,
        inputs: MutableMapping[str, np.ndarray],
        outputs: MutableMapping[str, np.ndarray],
        work_dir: Optional[Union[str, pathlib.Path]] = None,
        **kwargs
    ):
        raise NotImplementedError

    @abstractmethod
    def forward(self, inputs: MutableMapping[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @property
    def causal_graph(self) -> torch.Tensor:
        """property causal graph"""
        if self.graph is None:
            return torch.ones(len(self.input_variables), len(self.output_variables), dtype=torch.int, device=self.device)
        else:
            return self.graph.get_binary_adj_matrix()

    def save(self, save_dir: Union[str, pathlib.Path]):
        pass

    def load(self, load_dir: Union[str, pathlib.Path]):
        pass


class EnsembleNeuralMech(BaseCausalMech):
    def __init__(
        self,
        # base
        name: str,
        input_variables: List[Variable],
        output_variables: List[Variable],
        logger: Optional[Logger] = None,
        # model learning
        longest_epoch: int = -1,
        improvement_threshold: float = 0.01,
        patience: int = 5,
        batch_size: int = 256,
        # ensemble
        ensemble_num: int = 7,
        elite_num: int = 5,
        # cfgs
        network_cfg: Optional[DictConfig] = None,
        encoder_cfg: Optional[DictConfig] = None,
        decoder_cfg: Optional[DictConfig] = None,
        optimizer_cfg: Optional[DictConfig] = None,
        scheduler_cfg: Optional[DictConfig] = None,
        # forward method
        residual: bool = True,
        encoder_reduction: str = "sum",
        # others
        device: Union[str, torch.device] = "cpu",
    ):
        BaseCausalMech.__init__(
            self, name=name, input_variables=input_variables, output_variables=output_variables, logger=logger
        )
        # model learning
        self.longest_epoch = longest_epoch
        self.improvement_threshold = improvement_threshold
        self.patience = patience
        self.batch_size = batch_size
        # ensemble
        self.ensemble_num = ensemble_num
        self.elite_num = elite_num
        # cfgs
        self.network_cfg = NETWORK_CFG if network_cfg is None else network_cfg
        self.encoder_cfg = ENCODER_CFG if encoder_cfg is None else encoder_cfg
        self.decoder_cfg = DECODER_CFG if decoder_cfg is None else decoder_cfg
        self.optimizer_cfg = OPTIMIZER_CFG if optimizer_cfg is None else optimizer_cfg
        self.scheduler_cfg = SCHEDULER_CFG if scheduler_cfg is None else scheduler_cfg
        # forward method
        self.residual = residual
        self.encoder_reduction = encoder_reduction
        # others
        self.device = device

        # build member object
        self.variable_encoders: Optional[Dict[str, VariableEncoder]] = None
        self.variable_decoders: Optional[Dict[str, VariableEncoder]] = None
        self.network: Optional[BaseNetwork] = None
        self.graph: Optional[BaseGraph] = None
        self.optimizer: Optional[Optimizer] = None
        self.scheduler: Optional[object] = None
        self.build_coders()
        self.build_network()
        self.build_graph()
        self.build_optimizer()

        self.total_epoch = 0
        self.elite_indices: List[int] = []

    @property
    def encoder_output_dim(self):
        return self.encoder_cfg.output_dim

    @property
    def decoder_input_dim(self):
        return self.decoder_cfg.input_dim

    def build_network(self):
        self.network = instantiate(self.network_cfg)(
            input_dim=self.encoder_output_dim,
            output_dim=self.decoder_input_dim,
            extra_dims=[self.output_var_num, self.ensemble_num],
        ).to(self.device)

    def build_optimizer(self):
        assert self.network, "you must build network first"
        assert self.variable_encoders and self.variable_decoders, "you must build coders first"
        params = (
            [self.network.parameters()]
            + [encoder.parameters() for encoder in self.variable_encoders.values()]
            + [decoder.parameters() for decoder in self.variable_decoders.values()]
        )

        self.optimizer = instantiate(self.optimizer_cfg)(params=chain(*params))
        self.scheduler = instantiate(self.scheduler_cfg)(optimizer=self.optimizer)

    def forward(self, inputs: MutableMapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size, _ = self.get_inputs_batch_size(inputs)

        inputs_tensor = torch.zeros(self.ensemble_num, batch_size, self.input_var_num, self.encoder_output_dim).to(self.device)
        for i, var in enumerate(self.input_variables):
            out = self.variable_encoders[var.name](inputs[var.name].to(self.device))
            inputs_tensor[:, :, i] = out

        output_tensor = self.network(self.reduce_encoder_output(inputs_tensor))

        outputs = {}
        for i, var in enumerate(self.output_variables):
            hid = output_tensor[i]
            outputs[var.name] = self.variable_decoders[var.name](hid)

        if self.residual:
            outputs = self.residual_outputs(inputs, outputs)
        return outputs

    def build_graph(self):
        pass

    def build_coders(self):
        self.variable_encoders = {}
        for var in self.input_variables:
            assert var.name not in self.variable_encoders, "duplicate name in encoders: {}".format(var.name)
            self.variable_encoders[var.name] = instantiate(self.encoder_cfg)(variable=var).to(self.device)

        self.variable_decoders = {}
        for var in self.output_variables:
            assert var.name not in self.variable_decoders, "duplicate name in decoders: {}".format(var.name)
            self.variable_decoders[var.name] = instantiate(self.decoder_cfg)(variable=var).to(self.device)

    def save(self, save_dir: Union[str, pathlib.Path]):
        if isinstance(save_dir, str):
            save_dir = pathlib.Path(save_dir)
        save_dir = save_dir / pathlib.Path(self.name)
        save_dir.mkdir(exist_ok=True)

        self.network.save(save_dir)
        if self.graph is not None:
            self.graph.save(save_dir)
        for coder in self.variable_encoders.values():
            coder.save(save_dir)
        for coder in self.variable_decoders.values():
            coder.save(save_dir)

    def load(self, load_dir: Union[str, pathlib.Path]):
        if isinstance(load_dir, str):
            load_dir = pathlib.Path(load_dir)
        assert load_dir.exists()

        self.network.load(load_dir)
        if self.graph is not None:
            self.graph.load(load_dir)
        for coder in self.variable_encoders.values():
            coder.load(load_dir)
        for coder in self.variable_decoders.values():
            coder.load(load_dir)

    def get_inputs_info(self, inputs: MutableMapping[str, torch.Tensor]):
        assert len(set(inputs.keys()) & set(self.input_variables_dict.keys())) == len(inputs)
        data_shape = next(iter(inputs.values())).shape
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

    def reduce_encoder_output(
        self,
        encoder_output: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert len(encoder_output.shape) == 4, (
            "shape of `encoder_output` should be (ensemble-num, batch-size, input-var-num, encoder-output-dim), "
            "rather than {}".format(encoder_output.shape)
        )

        if mask is None:
            # [..., input-var-num]
            mask = self.forward_mask
            # [..., ensemble-num, batch-size, input-var-num]
            mask = mask.unsqueeze(-2).unsqueeze(-2)
            mask = mask.repeat((1,) * len(mask.shape[:-3]) + (*encoder_output.shape[:2], 1))

        # mask shape [..., ensemble-num, batch-size, input-var-num]
        assert (
            mask.shape[-3:] == encoder_output.shape[:-1]
        ), "mask shape should be (..., ensemble-num, batch-size, input-var-num)"

        # [*mask-extra-dims, ensemble-num, batch-size, input-var-num, encoder-output-dim]
        mask = mask[..., None].repeat([1] * len(mask.shape) + [encoder_output.shape[-1]])
        masked_encoder_output = encoder_output.repeat(tuple(mask.shape[:-4]) + (1,) * 4)

        # choose mask value
        mask_value = 0
        if self.encoder_reduction == "max":
            mask_value = -float("inf")
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

    @property
    def forward_mask(self) -> torch.Tensor:
        """property input masks"""
        return self.causal_graph.T

    def get_data_loaders(
        self,
        inputs: MutableMapping[str, np.ndarray],
        outputs: MutableMapping[str, np.ndarray],
    ):
        train_set = EnsembleBufferDataset(
            inputs=inputs, outputs=outputs, training=True, train_ratio=0.8, ensemble_num=self.ensemble_num, seed=1
        )
        valid_set = EnsembleBufferDataset(
            inputs=inputs, outputs=outputs, training=False, train_ratio=0.8, ensemble_num=self.ensemble_num, seed=1
        )

        train_loader = DataLoader(train_set, batch_size=self.batch_size, collate_fn=collate_fn, num_workers=cpu_count())
        valid_loader = DataLoader(valid_set, batch_size=self.batch_size, collate_fn=collate_fn, num_workers=cpu_count())

        return train_loader, valid_loader

    def learn(
        self,
        inputs: MutableMapping[str, np.ndarray],
        outputs: MutableMapping[str, np.ndarray],
        work_dir: Optional[Union[str, pathlib.Path]] = None,
        **kwargs
    ):
        train_loader, valid_loader = self.get_data_loaders(inputs, outputs)

        best_weights: Optional[Dict] = None
        epoch_iter = range(self.longest_epoch) if self.longest_epoch >= 0 else count()
        epochs_since_update = 0

        loss_func = partial(variable_loss_func, output_variables=self.output_variables, device=self.device)
        train = partial(train_func, forward=self.forward, optimizer=self.optimizer, loss_func=loss_func)
        eval = partial(eval_func, forward=self.forward, loss_func=loss_func)

        best_eval_loss = eval(valid_loader).mean(dim=(-2, -1))

        for epoch in epoch_iter:
            train_loss = train(train_loader)
            eval_loss = eval(valid_loader)

            maybe_best_weights = self._maybe_get_best_weights(
                best_eval_loss, eval_loss.mean(dim=(-2, -1)), self.improvement_threshold
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
                self.logger.record("{}/lr".format(self.name), self.optimizer.param_groups[0]["lr"])

                self.logger.dump(self.total_epoch)

            if self.patience and epochs_since_update >= self.patience:
                break

            self.scheduler.step()

        # saving the best models:
        self._maybe_set_best_weights_and_elite(best_weights, best_eval_loss)

        self.save(save_dir=work_dir)

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

    def get_inputs_batch_size(self, inputs: MutableMapping[str, torch.Tensor]) -> int:
        assert len(set(inputs.keys()) & set(self.variable_encoders.keys())) == len(inputs)
        data_shape = list(inputs.values())[0].shape
        # assert len(data_shape) == 3, "{}".format(data_shape)  # ensemble-num, batch-size, specific-dim
        ensemble, batch_size, specific_dim = data_shape[-3:]
        assert ensemble == self.ensemble_num

        return batch_size, data_shape[:-3]
