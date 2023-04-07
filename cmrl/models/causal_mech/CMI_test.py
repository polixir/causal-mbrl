from typing import Optional, List, Dict, Union, MutableMapping
import pathlib
from functools import partial
from itertools import count

import torch
import numpy as np
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from hydra.utils import instantiate
from stable_baselines3.common.logger import Logger

from cmrl.utils.variables import Variable
from cmrl.models.causal_mech.base import EnsembleNeuralMech
from cmrl.models.graphs.binary_graph import BinaryGraph
from cmrl.models.causal_mech.util import variable_loss_func, train_func, eval_func


class CMITestMech(EnsembleNeuralMech):
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
            residual=residual,
            encoder_reduction=encoder_reduction,
            device=device,
        )

        self.total_CMI_epoch = 0

    def build_network(self):
        self.network = instantiate(self.network_cfg)(
            input_dim=self.encoder_output_dim,
            output_dim=self.decoder_input_dim,
            extra_dims=[self.output_var_num, self.ensemble_num],
        ).to(self.device)

    def build_graph(self):
        self.graph = BinaryGraph(self.input_var_num, self.output_var_num, device=self.device)

    @property
    def CMI_mask(self) -> torch.Tensor:
        mask = torch.zeros(self.input_var_num + 1, self.output_var_num, self.input_var_num, dtype=torch.long)
        for i in range(self.input_var_num + 1):
            m = torch.ones(self.output_var_num, self.input_var_num)
            if i != self.input_var_num:
                m[:, i] = 0
            mask[i] = m
        return mask.to(self.device)

    def multi_graph_forward(self, inputs: MutableMapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """when first step, inputs should be dict of str and Tensor with (ensemble-num, batch-size, specific-dim) shape,
        since twice step, the shape of Tensor becomes (input-var-num + 1, ensemble-num, batch-size, specific-dim)

        Args:
            inputs:

        Returns:

        """
        batch_size, extra_dim = self.get_inputs_info(inputs)

        inputs_tensor = torch.empty(*extra_dim, self.ensemble_num, batch_size, self.input_var_num, self.encoder_output_dim).to(
            self.device
        )
        for i, var in enumerate(self.input_variables):
            out = self.variable_encoders[var.name](inputs[var.name].to(self.device))
            inputs_tensor[..., i, :] = out

        # if len(extra_dim) == 0:
        #     # [..., output-var-num, input-var-num]
        #     mask = self.CMI_mask
        #     # [..., output-var-num, ensemble-num, batch-size, input-var-num]
        #     mask = mask.unsqueeze(-2).unsqueeze(-2)
        #     mask = mask.repeat((1,) * len(mask.shape[:-3]) + (self.ensemble_num, batch_size, 1))
        #     reduced_inputs_tensor = self.reduce_encoder_output(inputs_tensor, mask)
        #     assert (
        #         not torch.isinf(reduced_inputs_tensor).any() and not torch.isnan(reduced_inputs_tensor).any()
        #     ), "tensor must not be inf or nan"
        #     output_tensor = self.network(reduced_inputs_tensor)
        # else:
        #     output_tensor = torch.empty(
        #         *extra_dim, self.output_var_num, self.ensemble_num, batch_size, self.decoder_input_dim
        #     ).to(self.device)
        #
        #     CMI_mask = self.CMI_mask
        #     for i in range(self.input_var_num + 1):
        #         # [..., output-var-num, input-var-num]
        #         mask = CMI_mask[i]
        #         # [..., output-var-num, ensemble-num, batch-size, input-var-num]
        #         mask = mask.unsqueeze(-2).unsqueeze(-2)
        #         mask = mask.repeat((1,) * len(mask.shape[:-3]) + (self.ensemble_num, batch_size, 1))
        #         if i == len(inputs_tensor) - 1:
        #             reduced_inputs_tensor = self.reduce_encoder_output(inputs_tensor[i], mask)
        #             outs = self.network(reduced_inputs_tensor)
        #             output_tensor[i] = outs
        #         else:
        #             for j in range(self.output_var_num):
        #                 ins = inputs_tensor[-1]
        #                 ins[:, :, j] = inputs_tensor[i, :, :, j, :]
        #                 reduced_inputs_tensor = self.reduce_encoder_output(inputs_tensor[i], mask)
        #                 outs = self.network(reduced_inputs_tensor)
        #                 output_tensor[i, j] = outs[j]

        mask = self.CMI_mask
        # [..., output-var-num, ensemble-num, batch-size, input-var-num]
        mask = mask.unsqueeze(-2).unsqueeze(-2)
        mask = mask.repeat((1,) * len(mask.shape[:-3]) + (self.ensemble_num, batch_size, 1))
        reduced_inputs_tensor = self.reduce_encoder_output(inputs_tensor, mask)
        assert (
            not torch.isinf(reduced_inputs_tensor).any() and not torch.isnan(reduced_inputs_tensor).any()
        ), "tensor must not be inf or nan"
        output_tensor = self.network(reduced_inputs_tensor)

        outputs = {}
        for i, var in enumerate(self.output_variables):
            hid = output_tensor[:, i]
            outputs[var.name] = self.variable_decoders[var.name](hid)

        if self.residual:
            outputs = self.residual_outputs(inputs, outputs)
        return outputs

    def calculate_CMI(self, nll_loss: torch.Tensor, threshold=1):
        nll_loss_diff = nll_loss[:-1] - nll_loss[-1]
        graph_data = (nll_loss_diff.mean(dim=(1, 2)) > threshold).to(torch.long)
        return graph_data, nll_loss_diff.mean(dim=(1, 2))

    def learn(
        self,
        inputs: MutableMapping[str, np.ndarray],
        outputs: MutableMapping[str, np.ndarray],
        work_dir: Optional[pathlib.Path] = None,
        **kwargs
    ):
        work_dir = pathlib.Path(".") if work_dir is None else work_dir

        open(work_dir / "history_mask.txt", "w")
        open(work_dir / "history_cmi.txt", "w")
        train_loader, valid_loader = self.get_data_loaders(inputs, outputs)

        final_graph_data = None

        epoch_iter = range(self.longest_epoch) if self.longest_epoch >= 0 else count()
        epochs_since_update = 0

        loss_func = partial(variable_loss_func, output_variables=self.output_variables, device=self.device)
        train = partial(train_func, forward=self.multi_graph_forward, optimizer=self.optimizer, loss_func=loss_func)
        eval = partial(eval_func, forward=self.multi_graph_forward, loss_func=loss_func)

        best_eval_loss = eval(valid_loader).mean(dim=(0, 2, 3))

        for epoch in epoch_iter:
            train_loss = train(train_loader)
            eval_loss = eval(valid_loader)

            improvement = (best_eval_loss - eval_loss.mean(dim=(0, 2, 3))) / torch.abs(best_eval_loss)
            if (improvement > self.improvement_threshold).any().item():
                best_eval_loss = torch.minimum(best_eval_loss, eval_loss.mean(dim=(0, 2, 3)))
                epochs_since_update = 0

                final_graph_data, mean_nll_loss_diff = self.calculate_CMI(eval_loss)
                with open(work_dir / "history_mask.txt", "a") as f:
                    f.write(str(final_graph_data) + "\n")
                with open(work_dir / "history_cmi.txt", "a") as f:
                    f.write(str(mean_nll_loss_diff) + "\n")
                print(
                    "new best valid, CMI test result:\n{}\nwith mean nll loss diff:\n{}".format(
                        final_graph_data, mean_nll_loss_diff
                    )
                )
            else:
                epochs_since_update += 1

            # log
            self.total_CMI_epoch += 1
            if self.logger is not None:
                self.logger.record("{}-CMI-test/epoch".format(self.name), epoch)
                self.logger.record("{}-CMI-test/epochs_since_update".format(self.name), epochs_since_update)
                self.logger.record("{}-CMI-test/train_dataset_size".format(self.name), len(train_loader.dataset))
                self.logger.record("{}-CMI-test/valid_dataset_size".format(self.name), len(valid_loader.dataset))
                self.logger.record("{}-CMI-test/train_loss".format(self.name), train_loss.mean().item())
                self.logger.record("{}-CMI-test/val_loss".format(self.name), eval_loss.mean().item())
                self.logger.record("{}-CMI-test/best_val_loss".format(self.name), best_eval_loss.mean().item())
                self.logger.record("{}-CMI-test/lr".format(self.name), self.optimizer.param_groups[0]["lr"])

                self.logger.dump(self.total_CMI_epoch)

            if self.patience and epochs_since_update >= self.patience:
                break

            self.scheduler.step()
            print(self.optimizer)

        assert final_graph_data is not None
        self.graph.set_data(final_graph_data)
        self.build_optimizer()

        super(CMITestMech, self).learn(inputs, outputs, work_dir=work_dir, **kwargs)


if __name__ == "__main__":
    import gym
    from stable_baselines3.common.buffers import ReplayBuffer
    from torch.utils.data import DataLoader

    from cmrl.models.data_loader import EnsembleBufferDataset, collate_fn, buffer_to_dict
    from cmrl.utils.creator import parse_space
    from cmrl.sb3_extension.logger import configure as logger_configure

    from cmrl.utils.env import load_offline_data
    from cmrl.models.causal_mech.util import variable_loss_func

    def unwrap_env(env):
        while isinstance(env, gym.Wrapper):
            env = env.env
        return env

    env = unwrap_env(gym.make("ParallelContinuousCartPoleSwingUp-v0"))
    real_replay_buffer = ReplayBuffer(
        int(1e6), env.observation_space, env.action_space, "cpu", handle_timeout_termination=False
    )
    load_offline_data(env, real_replay_buffer, "SAC-expert", use_ratio=0.01)

    extra_info = {"Radian": ["obs_1", "obs_5", "obs_9"]}
    # extra_info = {"Radian": ["obs_1"]}

    input_variables = parse_space(env.state_space, "obs", extra_info=extra_info) + parse_space(env.action_space, "act")
    output_variables = parse_space(env.state_space, "next_obs", extra_info=extra_info)

    logger = logger_configure("cmi-log", ["tensorboard", "stdout"])

    mech = CMITestMech("kernel_test_mech", input_variables, output_variables)

    inputs, outputs = buffer_to_dict(env.state_space, env.action_space, env.obs2state, real_replay_buffer, "transition")

    mech.learn(inputs, outputs)
