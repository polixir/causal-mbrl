from typing import Optional, List, Dict, Union, MutableMapping

import numpy
import numpy as np
import torch
from omegaconf import DictConfig
from stable_baselines3.common.logger import Logger
from hydra.utils import instantiate
from causallearn.utils.KCI.KCI import KCI_CInd
from tqdm import tqdm

from cmrl.models.causal_mech.base import EnsembleNeuralMech
from cmrl.utils.variables import Variable, ContinuousVariable, DiscreteVariable, BinaryVariable, RadianVariable


class KernelTestMech(EnsembleNeuralMech):
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
            # KCI
            sample_num=2000,
            kci_times=10
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
        self.sample_num = sample_num
        self.kci_times = kci_times

    def kci_compute_graph(
            self,
            inputs: MutableMapping[str, numpy.ndarray],
            outputs: MutableMapping[str, numpy.ndarray],
            **kwargs
    ):

        # [[0, 0, 0, 0],
        #  [0, 0, 1, 1],
        #  [1, 0, 0, 0],
        #  [0, 1, 1, 1],
        #  [0, 0, 1, 1]]

        length = next(iter(inputs.values())).shape[0]
        sample_length = min(length, self.sample_num)

        def deal_with_radian_input(name, data):
            if isinstance(self.input_variables_dict[name], RadianVariable):
                return (data + np.pi) % (2 * np.pi) - np.pi
            else:
                return data

        p_values_array = np.empty((self.kci_times, self.input_var_num, self.output_var_num))
        with tqdm(total=self.kci_times * self.input_var_num * self.output_var_num, desc="kci") as pbar:
            for time in range(self.kci_times):
                sample_indices = np.random.permutation(length)[:sample_length]
                for out_idx, out_name in enumerate(outputs):
                    for in_idx, in_name in enumerate(inputs):
                        if self.residual:
                            data_x = outputs[out_name][sample_indices] - inputs[out_name.replace("next_", "")][
                                sample_indices]
                            data_x = data_x
                        else:
                            data_x = outputs[out_name][sample_indices]
                        data_y = deal_with_radian_input(in_name, inputs[in_name])[sample_indices]
                        data_z = [deal_with_radian_input(other_in_name, in_data)[sample_indices]
                                  for other_in_name, in_data in inputs.items() if other_in_name != in_name]
                        data_z = np.concatenate(data_z, axis=1)

                        # data_x = (data_x - data_x.mean(axis=0)) / data_x.std(axis=0)
                        # data_y = (data_y - data_y.mean(axis=0)) / data_y.std(axis=0)
                        # data_z = (data_z - data_z.mean(axis=0)) / data_z.std(axis=0)

                        kci = KCI_CInd()
                        p_value, test_stat = kci.compute_pvalue(data_x, data_y, data_z)
                        p_values_array[time][in_idx][out_idx] = p_value

                        pbar.update(1)
        final_p_values = (p_values_array < 0.05).mean(axis=0)
        print(final_p_values)
        pass

    def build_network(self):
        self.network = instantiate(self.network_cfg)(
            input_dim=self.encoder_output_dim,
            output_dim=self.decoder_input_dim,
            extra_dims=[self.ensemble_num],
        ).to(self.device)

    def forward(
            self,
            inputs: MutableMapping[str, numpy.ndarray]
    ) -> Dict[str, torch.Tensor]:
        pass


if __name__ == '__main__':
    import gym
    from stable_baselines3.common.buffers import ReplayBuffer
    from torch.utils.data import DataLoader

    from cmrl.models.causal_mech.reinforce import ReinforceCausalMech
    from cmrl.models.data_loader import EnsembleBufferDataset, collate_fn, buffer_to_dict
    from cmrl.utils.creator import parse_space
    from cmrl.utils.env import load_offline_data
    from cmrl.models.causal_mech.util import variable_loss_func

    env = gym.make("ContinuousCartPoleSwingUp-v0", real_time_scale=0.02)
    real_replay_buffer = ReplayBuffer(int(1e6), env.observation_space, env.action_space, "cpu",
                                      handle_timeout_termination=False)
    load_offline_data(env, real_replay_buffer, "SAC-expert", use_ratio=1)

    input_variables = parse_space(env.observation_space, "obs") + parse_space(env.action_space, "act")
    output_variables = parse_space(env.observation_space, "next_obs")

    mech = KernelTestMech(
        "kernel_test_mech",
        input_variables,
        output_variables,
        sample_num=1000,
        kci_times=20
    )

    inputs, outputs = buffer_to_dict(
        env.observation_space,
        env.action_space,
        env.obs2state,
        real_replay_buffer,
        "transition"
    )

    mech.kci_compute_graph(inputs, outputs)
