import collections

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import pathlib
import abc
import omegaconf
import itertools
from typing import Optional, Union, Tuple, Callable, Dict, List, cast
import copy
from cmrl.models.nns import EnsembleMLP
from cmrl.models.transition.base_transition import BaseEnsembleTransition
from cmrl.models.reward_and_termination import BaseRewardMech, BaseTerminationMech
from cmrl.models.dynamics import BaseDynamics
from cmrl.util.logger import Logger
from cmrl.util.replay_buffer import ReplayBuffer, TransitionIterator, BootstrapIterator
from cmrl.types import InteractionBatch


class PlainEnsembleDynamics(BaseDynamics):
    def __init__(self,
                 transition: BaseEnsembleTransition,
                 learned_reward: bool = True,
                 reward_mech: Optional[BaseRewardMech] = None,
                 learned_termination: bool = False,
                 termination_mech: Optional[BaseTerminationMech] = None,
                 # trainer
                 optim_lr: float = 1e-4,
                 weight_decay: float = 1e-5,
                 optim_eps: float = 1e-8,
                 logger: Optional[Logger] = None, ):
        super(PlainEnsembleDynamics, self).__init__(transition=transition,
                                                    learned_reward=learned_reward,
                                                    reward_mech=reward_mech,
                                                    learned_termination=learned_termination,
                                                    termination_mech=termination_mech,
                                                    optim_lr=optim_lr,
                                                    weight_decay=weight_decay,
                                                    optim_eps=optim_eps,
                                                    logger=logger)

    def learn(self,
              # data
              replay_buffer: ReplayBuffer,
              # dataset split
              validation_ratio: float = 0.2,
              batch_size: int = 256,
              shuffle_each_epoch: bool = True,
              bootstrap_permutes: bool = False,
              # model learning
              longest_epoch: Optional[int] = None,
              improvement_threshold: float = 0.1,
              patience: int = 5,
              work_dir: Optional[Union[str, pathlib.Path]] = None,
              # other
              **kwargs):
        train_dataset, val_dataset = self.dataset_split(replay_buffer,
                                                        validation_ratio,
                                                        batch_size,
                                                        shuffle_each_epoch,
                                                        bootstrap_permutes)
        longest_epoch = longest_epoch

        for mech in self.learn_mech:
            train_losses, val_losses = [], []
            best_weights: Optional[Dict] = None
            epoch_iter = range(longest_epoch) if longest_epoch > 0 else itertools.count()
            epochs_since_update = 0

            best_val_loss = self.evaluate(val_dataset, mech=mech)

            for epoch in epoch_iter:
                train_loss = self.train(train_dataset, mech=mech)
                train_losses.append(train_loss)

                val_loss = self.evaluate(val_dataset, mech=mech)
                val_losses.append(val_loss)

                maybe_best_weights = self.maybe_get_best_weights(
                    best_val_loss, val_loss, mech, improvement_threshold,
                )
                if maybe_best_weights:
                    # best loss
                    best_val_loss = val_loss.clone()
                    best_weights = maybe_best_weights
                    epochs_since_update = 0
                else:
                    epochs_since_update += 1

                # log
                if self.logger is not None:
                    self.logger.log_data(
                        mech,
                        {
                            "epoch": epoch,
                            "train_dataset_size": train_dataset.num_stored,
                            "val_dataset_size": val_dataset.num_stored,
                            "train_loss": train_loss.mean(),
                            "val_loss": val_loss.mean(),
                            "best_val_score": best_val_loss.mean()
                        }, )

                if patience and epochs_since_update >= patience:
                    break

            # saving the best models:
            self.maybe_set_best_weights_and_elite(best_weights, best_val_loss, mech=mech)
        self.save(work_dir)

    def maybe_get_best_weights(
            self,
            best_val_loss: torch.Tensor,
            val_loss: torch.Tensor,
            mech: str = "transition",
            threshold: float = 0.01, ):
        best_every_member_loss = best_val_loss.mean(dim=(1, 2))
        every_member_loss = val_loss.mean(dim=(1, 2))
        improvement = (best_every_member_loss - every_member_loss) / torch.abs(best_every_member_loss)
        if (improvement > threshold).any().item():
            model = getattr(self, mech)
            best_weights = copy.deepcopy(model.state_dict())
        else:
            best_weights = None

        return best_weights

    def maybe_set_best_weights_and_elite(
            self,
            best_weights: Optional[Dict],
            best_val_loss: torch.Tensor,
            mech: str = "transition",
    ):
        model = getattr(self, mech)
        assert isinstance(model, EnsembleMLP)
        best_every_member_loss = best_val_loss.mean(dim=(1, 2))

        if best_weights is not None:
            model.load_state_dict(best_weights)
        sorted_indices = np.argsort(best_every_member_loss.tolist())
        elite_models = sorted_indices[: model.elite_num]
        model.set_elite(elite_models)
