from typing import Optional, List, Dict, Union
import torch
from omegaconf import DictConfig

from cmrl.types import Variable
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

    def build_graph(self):
        self.graph = None

    def learn(self, train_loader, valid_loader):
        pass
