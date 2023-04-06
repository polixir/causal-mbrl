from typing import Optional, List, Dict, Union, MutableMapping

import numpy
import torch
from torch.utils.data import DataLoader
import numpy as np
from omegaconf import DictConfig
from hydra.utils import instantiate
from stable_baselines3.common.logger import Logger

from cmrl.utils.variables import Variable
from cmrl.models.causal_mech.base import EnsembleNeuralMech
from cmrl.models.graphs.binary_graph import BinaryGraph
from cmrl.models.data_loader import EnsembleBufferDataset, collate_fn


class OracleMech(EnsembleNeuralMech):
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
        EnsembleNeuralMech.__init__(
            self,
            name=name,
            input_variables=input_variables,
            output_variables=output_variables,
            logger=logger,
            longest_epoch=longest_epoch,
            improvement_threshold=improvement_threshold,
            patience=patience,
            batch_size=batch_size,
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

    def set_oracle_graph(self, graph_data: Optional[numpy.ndarray]):
        self.graph = BinaryGraph(self.input_var_num, self.output_var_num, device=self.device)
        if graph_data is None:
            graph_data = np.ones([self.input_var_num, self.output_var_num])
        self.graph.set_data(graph_data=graph_data)
        print("set oracle causal graph successfully: \n{}".format(graph_data))


if __name__ == "__main__":
    from typing import cast

    import gym
    from stable_baselines3.common.buffers import ReplayBuffer
    from torch.utils.data import DataLoader
    from emei import EmeiEnv

    from cmrl.models.data_loader import EnsembleBufferDataset, collate_fn, buffer_to_dict
    from cmrl.utils.creator import parse_space
    from cmrl.utils.env import load_offline_data
    from cmrl.models.causal_mech.util import variable_loss_func
    from cmrl.sb3_extension.logger import configure as logger_configure

    env = cast(EmeiEnv, gym.make("ParallelContinuousCartPoleSwingUp-v0"))
    real_replay_buffer = ReplayBuffer(
        int(1e6), env.observation_space, env.action_space, "cpu", handle_timeout_termination=False
    )
    load_offline_data(env, real_replay_buffer, "SAC-expert", use_ratio=1)

    input_variables = parse_space(env.state_space, "obs") + parse_space(env.action_space, "act")
    output_variables = parse_space(env.state_space, "next_obs")

    logger = logger_configure("kci-log", ["tensorboard", "stdout"])

    mech = OracleMech("plain_mech", input_variables, output_variables, logger=logger, device="cuda:1")

    inputs, outputs = buffer_to_dict(env.observation_space, env.action_space, env.obs2state, real_replay_buffer, "transition")

    mech.learn(inputs, outputs)
