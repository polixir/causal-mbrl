from typing import Optional, List, Dict, Union, MutableMapping
from functools import partial
from collections import defaultdict

import pathlib
import numpy
import numpy as np
import torch
from omegaconf import DictConfig
from stable_baselines3.common.logger import Logger
from hydra.utils import instantiate

# from cmrl.utils.RCIT import KCI_CInd
from causallearn.utils.KCI.KCI import KCI_CInd
from tqdm import tqdm

from cmrl.models.causal_mech.base import EnsembleNeuralMech
from cmrl.utils.variables import Variable, ContinuousVariable, DiscreteVariable, BinaryVariable, RadianVariable
from cmrl.models.graphs.binary_graph import BinaryGraph


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
        # KCI
        sample_num: int = 2000,
        kci_times: int = 10,
        not_confident_bound: float = 0.25,
        longest_sample: int = 5000,
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
        self.sample_num = sample_num
        self.kci_times = kci_times
        self.not_confident_bound = not_confident_bound
        self.longest_sample = longest_sample

    def kci(
        self,
        input_idx: int,
        output_idx: int,
        inputs: MutableMapping[str, numpy.ndarray],
        outputs: MutableMapping[str, numpy.ndarray],
        sample_indices: np.ndarray,
    ):
        in_name, out_name = list(inputs.keys())[input_idx], list(outputs.keys())[output_idx]

        if self.residual:
            data_x = outputs[out_name][sample_indices] - inputs[out_name.replace("next_", "")][sample_indices]
        else:
            data_x = outputs[out_name][sample_indices]

        def deal_with_radian_input(name, data):
            if isinstance(self.input_variables_dict[name], RadianVariable):
                return (data + np.pi) % (2 * np.pi) - np.pi
            else:
                return data

        data_y = deal_with_radian_input(in_name, inputs[in_name])[sample_indices]
        data_z = [
            deal_with_radian_input(other_in_name, in_data)[sample_indices]
            for other_in_name, in_data in inputs.items()
            if other_in_name != in_name
        ]
        data_z = np.concatenate(data_z, axis=1)

        kci = KCI_CInd()
        p_value, test_stat = kci.compute_pvalue(data_x, data_y, data_z)
        return p_value

    def kci_compute_graph(
        self,
        inputs: MutableMapping[str, numpy.ndarray],
        outputs: MutableMapping[str, numpy.ndarray],
        work_dir: Optional[pathlib.Path] = None,
        **kwargs
    ):

        open(work_dir / "history_vote.txt", "w")

        length = next(iter(inputs.values())).shape[0]
        sample_length = min(length, self.sample_num) if self.sample_num > 0 else length

        init_pvalues_array = np.empty((self.kci_times, self.input_var_num, self.output_var_num))
        with tqdm(
            total=self.kci_times * self.input_var_num * self.output_var_num,
            desc="init kci of {} samples".format(sample_length),
        ) as pbar:
            for time in range(self.kci_times):
                sample_indices = np.random.permutation(length)[:sample_length]
                kci = partial(self.kci, inputs=inputs, outputs=outputs, sample_indices=sample_indices)
                for out_idx in range(len(outputs)):
                    for in_idx in range(len(inputs)):
                        init_pvalues_array[time][in_idx][out_idx] = kci(in_idx, out_idx)
                        pbar.update(1)

        votes = (init_pvalues_array < 0.05).mean(axis=0)
        is_not_confident = np.logical_and(votes > self.not_confident_bound, votes < 1 - self.not_confident_bound)
        not_confident_list = np.array(np.where(is_not_confident)).T

        recompute_times = 1
        while len(not_confident_list) != 0:
            with open(work_dir / "history_vote.txt", "a") as f:
                f.write(str(votes) + "\n")
            print(votes)

            new_sample_length = int(sample_length * 1.5**recompute_times)
            if new_sample_length > min(self.longest_sample, length):
                break

            pvalues_dict = defaultdict(list)
            with tqdm(
                total=self.kci_times * len(not_confident_list),
                desc="{}th re-compute kci of {} samples".format(recompute_times, new_sample_length),
            ) as pbar:
                for time in range(self.kci_times):
                    sample_indices = np.random.permutation(length)[:new_sample_length]
                    kci = partial(self.kci, inputs=inputs, outputs=outputs, sample_indices=sample_indices)
                    for in_idx, out_idx in not_confident_list:
                        pvalues_dict[(in_idx, out_idx)].append(kci(in_idx, out_idx))
                        pbar.update(1)

            not_confident_list = []
            for key, value in pvalues_dict.items():
                vote = (np.array(value) < 0.05).mean()
                if self.not_confident_bound < vote < 1 - self.not_confident_bound:
                    not_confident_list.append(key)
                else:
                    votes[key] = vote
            recompute_times += 1

        return votes > 0.5

    def build_network(self):
        self.network = instantiate(self.network_cfg)(
            input_dim=self.encoder_output_dim,
            output_dim=self.decoder_input_dim,
            extra_dims=[self.ensemble_num],
        ).to(self.device)

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
        self.graph = BinaryGraph(self.input_var_num, self.output_var_num, device=self.device)

    def learn(
        self,
        inputs: MutableMapping[str, np.ndarray],
        outputs: MutableMapping[str, np.ndarray],
        work_dir: Optional[pathlib.Path] = None,
        **kwargs
    ):
        work_dir = pathlib.Path(".") if work_dir is None else work_dir
        graph = self.kci_compute_graph(inputs, outputs, work_dir)
        self.graph.set_data(graph)

        super(KernelTestMech, self).learn(inputs, outputs, work_dir=work_dir, **kwargs)


if __name__ == "__main__":
    import gym
    from emei import EmeiEnv
    from stable_baselines3.common.buffers import ReplayBuffer
    from torch.utils.data import DataLoader
    from typing import cast

    from cmrl.models.data_loader import EnsembleBufferDataset, collate_fn, buffer_to_dict
    from cmrl.utils.creator import parse_space
    from cmrl.utils.env import load_offline_data
    from cmrl.sb3_extension.logger import configure as logger_configure
    from cmrl.models.causal_mech.util import variable_loss_func

    def unwrap_env(env):
        while isinstance(env, gym.Wrapper):
            env = env.env
        return env

    env = unwrap_env(gym.make("ParallelContinuousCartPoleSwingUp-v0"))
    real_replay_buffer = ReplayBuffer(
        int(1e6), env.observation_space, env.action_space, "cpu", handle_timeout_termination=False
    )
    load_offline_data(env, real_replay_buffer, "SAC-expert", use_ratio=1)

    extra_info = {"Radian": ["obs_1", "obs_5", "obs_9"]}
    # extra_info = {"Radian": ["obs_1"]}

    input_variables = parse_space(env.state_space, "obs", extra_info=extra_info) + parse_space(env.action_space, "act")
    output_variables = parse_space(env.state_space, "next_obs", extra_info=extra_info)

    logger = logger_configure("kci-log", ["tensorboard", "stdout"])

    mech = KernelTestMech("kernel_test_mech", input_variables, output_variables, sample_num=100, kci_times=20, logger=logger)

    inputs, outputs = buffer_to_dict(env.state_space, env.action_space, env.obs2state, real_replay_buffer, "transition")

    mech.learn(inputs, outputs)
