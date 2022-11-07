from typing import Optional, List, Dict, Union, MutableMapping

import torch
from omegaconf import DictConfig
from hydra.utils import instantiate
from stable_baselines3.common.logger import Logger

from cmrl.utils.types import Variable
from cmrl.models.causal_mech.base_causal_mech import NeuralCausalMech


class CMItest(NeuralCausalMech):
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

        super(CMItest, self).__init__(
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
            output_dim=self.decoder_input_dim,
            extra_dims=[self.output_var_num, self.ensemble_num],
        ).to(self.device)

    def build_graph(self):
        self.graph = None

    def single_step_forward(self, inputs: MutableMapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size = self.get_inputs_batch_size(inputs)

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
