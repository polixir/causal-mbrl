import abc

import torch
import numpy as np
import pathlib
import collections
from typing import Optional, Union, Tuple, Dict, List, cast
from cmrl.models.transition.base_transition import BaseEnsembleTransition
from cmrl.models.reward_and_termination import BaseRewardMech, BaseTerminationMech
from cmrl.util.logger import Logger
from cmrl.util.replay_buffer import ReplayBuffer, TransitionIterator, BootstrapIterator
from cmrl.types import InteractionBatch


def split_dict(old_dict: Dict,
               need_keys: List[str]):
    return dict([(key, old_dict[key]) for key in need_keys])


class BaseDynamics:
    _MECH_TO_VARIABLE = {"transition": "batch_next_obs",
                         "reward_mech": "batch_reward",
                         "termination_mech": "batch_terminal", }
    _VARIABLE_TO_MECH = dict([(value, key) for key, value in _MECH_TO_VARIABLE.items()])
    _MODEL_LOG_FORMAT = [
        ("epoch", "E", "int"),
        ("train_dataset_size", "TD", "int"),
        ("val_dataset_size", "VD", "int"),
        ("train_loss", "TLOSS", "float"),
        ("val_loss", "VLOSS", "float"),
        ("best_val_loss", "BVLOSS", "float"),
    ]

    def __init__(self,
                 transition: BaseEnsembleTransition,
                 learned_reward: bool = True,
                 reward_mech: Optional[BaseRewardMech] = None,
                 learned_termination: bool = False,
                 termination_mech: Optional[BaseTerminationMech] = None,
                 optim_lr: float = 1e-4,
                 weight_decay: float = 1e-5,
                 optim_eps: float = 1e-8,
                 logger: Optional[Logger] = None,
                 ):
        super(BaseDynamics, self).__init__()
        self.transition = transition
        self.learned_reward = learned_reward
        self.reward_mech = reward_mech
        self.learned_termination = learned_termination
        self.termination_mech = termination_mech

        self.optim_lr = optim_lr
        self.weight_decay = weight_decay
        self.optim_eps = optim_eps
        self.logger = logger

        self.device = self.transition.device
        self.ensemble_num = self.transition.ensemble_num

        self.learn_mech = ["transition"]
        self.transition_optimizer = torch.optim.Adam(
            self.transition.parameters(), lr=optim_lr, weight_decay=weight_decay, eps=optim_eps, )
        if self.learned_reward:
            self.reward_mech_optimizer = torch.optim.Adam(
                self.reward_mech.parameters(), lr=optim_lr, weight_decay=weight_decay, eps=optim_eps, )
            self.learn_mech.append("reward_mech")
        if self.learned_termination:
            self.termination_mech_optimizer = torch.optim.Adam(
                self.termination_mech.parameters(), lr=optim_lr, weight_decay=weight_decay, eps=optim_eps, )
            self.learn_mech.append("termination_mech")

        if self.logger is not None:
            for mech in self.learn_mech:
                self.logger.register_group(
                    mech,
                    self._MODEL_LOG_FORMAT,
                    color="blue",
                    dump_frequency=1,
                )

    @abc.abstractmethod
    def learn(self,
              replay_buffer: ReplayBuffer,
              **kwargs):
        pass

    # auxiliary method for "single batch data"
    def get_3d_tensor(self,
                      data: Union[np.ndarray, torch.Tensor],
                      is_ensemble: bool):
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        if is_ensemble:
            if data.ndim == 2:  # reward or terminal
                data = data.unsqueeze(data.ndim)
            return data.to(self.device)
        else:
            if data.ndim == 1:  # reward or terminal
                data = data.unsqueeze(data.ndim)
            return data.repeat([self.ensemble_num, 1, 1]).to(self.device)

    # auxiliary method for "interaction batch data"
    def get_mech_loss(self,
                      batch: InteractionBatch,
                      mech: str = "transition",
                      loss_type: str = "mse",
                      is_ensemble: bool = False):
        data = {}
        for attr in batch.attrs:
            data[attr] = self.get_3d_tensor(getattr(batch, attr).copy(), is_ensemble=is_ensemble)
        model_in = split_dict(data, ["batch_obs", "batch_action"])

        variable = self._MECH_TO_VARIABLE[mech]
        return getattr(getattr(self, mech), "get_{}_loss".format(loss_type))(model_in, data[variable])

    # auxiliary method for "replay buffer"
    def dataset_split(self,
                      replay_buffer: ReplayBuffer,
                      validation_ratio: float = 0.2,
                      batch_size: int = 256,
                      shuffle_each_epoch: bool = True,
                      bootstrap_permutes: bool = False,
                      ) -> Tuple[TransitionIterator, Optional[TransitionIterator]]:
        data = replay_buffer.get_all(shuffle=True)

        val_size = int(len(data) * validation_ratio)
        train_size = len(data) - val_size
        train_data = data[:train_size]
        train_iter = BootstrapIterator(
            train_data,
            batch_size,
            self.ensemble_num,
            shuffle_each_epoch=shuffle_each_epoch,
            permute_indices=bootstrap_permutes,
            rng=replay_buffer.rng,
        )

        val_iter = None
        if val_size > 0:
            val_data = data[train_size:]
            val_iter = TransitionIterator(
                val_data, batch_size, shuffle_each_epoch=False, rng=replay_buffer.rng
            )

        return train_iter, val_iter

    # auxiliary method for "dataset"
    def evaluate(self,
                 dataset: TransitionIterator,
                 mech: str = "transition", ):
        assert not isinstance(dataset, BootstrapIterator)

        batch_loss_list = []
        with torch.no_grad():
            for batch in dataset:
                val_loss = self.get_mech_loss(batch, mech=mech, loss_type="mse", is_ensemble=False)
                batch_loss_list.append(val_loss)
        return torch.cat(batch_loss_list, dim=batch_loss_list[0].ndim - 2).cpu()

    def train(self, dataset: TransitionIterator,
              mech: str = "transition", ):
        assert isinstance(dataset, BootstrapIterator)

        batch_loss_list = []
        for batch in dataset:
            train_loss = self.get_mech_loss(batch, mech=mech, loss_type="nll", is_ensemble=True)
            optim = getattr(self, "{}_optimizer".format(mech))
            optim.zero_grad()
            train_loss.mean().backward()
            optim.step()
            batch_loss_list.append(train_loss)
        return torch.cat(batch_loss_list, dim=batch_loss_list[0].ndim - 2).detach().cpu()

    def query(self, obs, action,
              return_as_np=True):
        result = collections.defaultdict(dict)
        obs = self.get_3d_tensor(obs, is_ensemble=False)
        action = self.get_3d_tensor(action, is_ensemble=False)
        for mech in self.learn_mech:
            with torch.no_grad():
                mean, logvar = getattr(self, "{}".format(mech)).forward(obs, action)
            variable = self.get_variable_by_mech(mech)
            if return_as_np:
                result[variable]["mean"] = mean.cpu().numpy()
                result[variable]["logvar"] = logvar.cpu().numpy()
            else:
                result[variable]["mean"] = mean.cpu()
                result[variable]["logvar"] = logvar.cpu()
        return result

    # other auxiliary method
    def save(self, save_dir: Union[str, pathlib.Path]):
        for mech in self.learn_mech:
            getattr(self, mech).save(save_dir=save_dir)

    def load(self, load_dir: Union[str, pathlib.Path]):
        for mech in self.learn_mech:
            getattr(self, mech).load(load_dir=load_dir)

    def get_variable_by_mech(self,
                             mech: str) -> str:
        assert mech in self._MECH_TO_VARIABLE
        return self._MECH_TO_VARIABLE[mech]

    def get_mach_by_variable(self,
                             variable: str) -> str:
        assert variable in self._VARIABLE_TO_MECH
        return self._VARIABLE_TO_MECH[variable]
