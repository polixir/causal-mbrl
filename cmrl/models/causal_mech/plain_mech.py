from typing import Optional, List, Dict, Union, MutableMapping

import torch
from torch.utils.data import DataLoader
import numpy as np
from omegaconf import DictConfig
from hydra.utils import instantiate
from stable_baselines3.common.logger import Logger

from cmrl.utils.variables import Variable
from cmrl.models.causal_mech.base import EnsembleNeuralMech
from cmrl.models.data_loader import EnsembleBufferDataset, collate_fn


class PlainMech(EnsembleNeuralMech):
    def __init__(
            self,
            name: str,
            input_variables: List[Variable],
            output_variables: List[Variable],
            logger: Optional[Logger] = None,
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
            scheduler_cfg: Optional[DictConfig] = None,
            # forward method
            residual: bool = True,
            encoder_reduction: str = "sum",
            # others
            device: Union[str, torch.device] = "cpu",
    ):
        EnsembleNeuralMech.__init__(
            self,
            name=name,
            input_variables=input_variables,
            output_variables=output_variables,
            logger=logger,
            longest_epoch=longest_epoch,
            improvement_threshold=improvement_threshold,
            patience=patience,
            ensemble_num=ensemble_num,
            elite_num=elite_num,
            network_cfg=network_cfg,
            encoder_cfg=encoder_cfg,
            decoder_cfg=decoder_cfg,
            optimizer_cfg=optimizer_cfg,
            scheduler_cfg=scheduler_cfg,
            residual=residual,
            encoder_reduction=encoder_reduction,
            device=device,
        )

    def forward(
            self,
            inputs: MutableMapping[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        batch_size, _ = self.get_inputs_info(inputs)

        inputs_tensor = torch.zeros(self.ensemble_num, batch_size, self.input_var_num, self.encoder_output_dim).to(
            self.device)
        for i, var in enumerate(self.input_variables):
            out = self.variable_encoders[var.name](inputs[var.name].to(self.device))
            inputs_tensor[:, :, i] = out

        # for name, param in self.network.named_parameters():
        #     if param.grad is not None and torch.isnan(param.grad).any():
        #         print("nan gradient found")
        #         print("name:", name)
        #         print("param:", param.grad)
        #         raise SystemExit

        output_tensor = self.network(self.reduce_encoder_output(inputs_tensor))

        outputs = {}
        for i, var in enumerate(self.output_variables):
            outputs[var.name] = self.variable_decoders[var.name](output_tensor)

        if self.residual:
            outputs = self.residual_outputs(inputs, outputs)
        return outputs

    @property
    def forward_mask(self):
        return torch.ones(self.input_var_num).to(self.device)


if __name__ == '__main__':
    import gym
    from stable_baselines3.common.buffers import ReplayBuffer
    from torch.utils.data import DataLoader

    from cmrl.models.causal_mech.reinforce import ReinforceCausalMech
    from cmrl.models.data_loader import EnsembleBufferDataset, collate_fn, buffer_to_dict
    from cmrl.utils.creator import parse_space
    from cmrl.utils.env import load_offline_data
    from cmrl.models.causal_mech.util import variable_loss_func
    from cmrl.sb3_extension.logger import configure as logger_configure

    env = gym.make("ContinuousCartPoleSwingUp-v0", real_time_scale=0.02)
    real_replay_buffer = ReplayBuffer(int(1e6), env.observation_space, env.action_space, "cpu",
                                      handle_timeout_termination=False)
    load_offline_data(env, real_replay_buffer, "SAC-expert-replay", use_ratio=1)

    input_variables = parse_space(env.observation_space, "obs") + parse_space(env.action_space, "act")
    output_variables = parse_space(env.observation_space, "next_obs")

    logger = logger_configure("plain-log", ["tensorboard", "stdout"])

    mech = PlainMech(
        "plain_mech",
        input_variables,
        output_variables,
        logger=logger,
        sample_num=2000,
        kci_times=20,
    )

    inputs, outputs = buffer_to_dict(
        env.observation_space,
        env.action_space,
        env.obs2state,
        real_replay_buffer,
        "transition"
    )

    mech.learn(inputs, outputs)
