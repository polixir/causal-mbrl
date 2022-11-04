from typing import Optional, List, Dict, Union
import pathlib
import itertools

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import Adam
from omegaconf import DictConfig
from stable_baselines3.common.logger import Logger

from cmrl.types import Variable, ContinuousVariable
from cmrl.models.networks.parallel_mlp import ParallelMLP
from cmrl.models.graphs.base_graph import BaseGraph
from cmrl.models.causal_mech.base_causal_mech import BaseCausalMech
from cmrl.models.networks.coder import VariableEncoder, VariableDecoder


class PlainMech(BaseCausalMech):
    def __init__(
        self,
        # base causal-mech params
        input_variables: List[Variable],
        output_variables: List[Variable],
        node_dim: int,
        variable_encoders: Dict[str, VariableEncoder],
        variable_decoders: Dict[str, VariableDecoder],
        # network params
        deterministic: bool = False,
        hidden_dims: Optional[List[int]] = None,
        ensemble_num: int = 7,
        use_bias: bool = True,
        activation_fn_cfg: Optional[DictConfig] = None,
        # forward method
        residual: bool = True,
        # trainer
        optim_lr: float = 1e-4,
        optim_weight_decay: float = 1e-5,
        optim_eps: float = 1e-8,
        optim_coder: bool = True,
        # logger
        logger: Optional[Logger] = None,
        # others
        device: Union[str, torch.device] = "cpu",
        **kwargs
    ):
        self.deterministic = deterministic
        self.hidden_dims = hidden_dims if hidden_dims is not None else [200] * 4
        self.ensemble_num = ensemble_num
        self.use_bias = use_bias
        self.activation_fn_cfg = activation_fn_cfg

        super(PlainMech, self).__init__(
            input_variables=input_variables,
            output_variables=output_variables,
            node_dim=node_dim,
            variable_encoders=variable_encoders,
            variable_decoders=variable_decoders,
            residual=residual,
            optim_lr=optim_lr,
            optim_weight_decay=optim_weight_decay,
            optim_eps=optim_eps,
            optim_coder=optim_coder,
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
        )

        if self.optim_coder:
            parmas = itertools.chain(
                self.network.parameters(),
                *[encoder.parameters() for encoder in self.variable_encoders.values()],
                *[decoder.parameters() for decoder in self.variable_decoders.values()]
            )
            self.optim = Adam(parmas, lr=self.optim_lr, weight_decay=self.optim_weight_decay, eps=self.optim_eps)

    def build_graph(self):
        self.graph = None

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert len(inputs) > 0, "inputs should not be null"
        ensemble, batch_size = list(inputs.values())[0].shape[:2]
        assert ensemble == self.ensemble_num
        assert list(inputs.keys()) == list(self.variable_encoders.keys())

        inputs_tensor = torch.zeros(ensemble, batch_size, self.input_var_num * self.node_dim)
        for i, (name, encoder) in enumerate(self.variable_encoders.items()):
            out = encoder(inputs[name])  # ensemble-num, batch-size, node-dim
            inputs_tensor[:, :, i * self.node_dim : (i + 1) * self.node_dim] = out

        hidden_tensor = self.network(inputs_tensor)

        outputs = {}
        for i, (name, decoder) in enumerate(self.variable_decoders.items()):
            hid = hidden_tensor[:, :, i * self.node_dim : (i + 1) * self.node_dim]
            out = decoder(hid)
            outputs[name] = out

        return outputs

    def loss(self, outputs, targets):
        loss = torch.tensor(0.0)
        for var in self.output_variables:
            output = outputs[var.name]
            target = targets[var.name]
            if isinstance(var, ContinuousVariable):
                loss += F.mse_loss(output, target)
        return loss

    def learn(
        self,
        # loader
        train_loader: DataLoader,
        valid_loader: DataLoader,
        # model learning
        longest_epoch: int = -1,
        improvement_threshold: float = 0.1,
        patience: int = 5,
        work_dir: Optional[Union[str, pathlib.Path]] = None,
        **kwargs
    ):
        best_weights: Optional[Dict] = None
        epoch_iter = range(longest_epoch) if longest_epoch > 0 else itertools.count()
        epochs_since_update = 0

        for inputs, targets in train_loader:
            outputs = self.forward(inputs)
            loss = self.loss(outputs, targets)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            break
