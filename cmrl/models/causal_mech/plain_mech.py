from typing import Optional, List, Dict, Union, MutableMapping

import torch
from omegaconf import DictConfig
from hydra.utils import instantiate
from stable_baselines3.common.logger import Logger

from cmrl.utils.variables import Variable
from cmrl.models.causal_mech.neural_causal_mech import NeuralCausalMech


class PlainMech(NeuralCausalMech):
    def __init__(
        self,
        name: str,
        input_variables: List[Variable],
        output_variables: List[Variable],
        # model learning
        longest_epoch: int = -1,
        improvement_threshold: float = 0.01,
        patience: int = 5,
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
        if multi_step == "none":
            multi_step = "forward-euler 1"

        super(PlainMech, self).__init__(
            name=name,
            input_variables=input_variables,
            output_variables=output_variables,
            longest_epoch=longest_epoch,
            improvement_threshold=improvement_threshold,
            patience=patience,
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

    @property
    def forward_mask(self):
        return torch.ones(self.input_var_num).to(self.device)
