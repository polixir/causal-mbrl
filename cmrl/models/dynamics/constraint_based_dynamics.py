import copy
import itertools
import pathlib
from typing import Callable, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import torch
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.buffers import ReplayBuffer

from cmrl.models.dynamics.base_dynamics import BaseDynamics
from cmrl.models.nns import EnsembleMLP
from cmrl.models.reward_and_termination import BaseRewardMech, BaseTerminationMech
from cmrl.models.transition.base_transition import BaseEnsembleTransition
from cmrl.models.causal_discovery.CMI_test import TransitionConditionalMutualInformationTest
from cmrl.util.transition_iterator import BootstrapIterator, TransitionIterator
from cmrl.models.util import to_tensor
from cmrl.types import TensorType


class ConstraintBasedDynamics(BaseDynamics):
    def __init__(
        self,
        transition: BaseEnsembleTransition,
        learned_reward: bool = True,
        reward_mech: Optional[BaseRewardMech] = None,
        learned_termination: bool = False,
        termination_mech: Optional[BaseTerminationMech] = None,
        # trainer
        optim_lr: float = 1e-4,
        weight_decay: float = 1e-5,
        optim_eps: float = 1e-8,
        logger: Optional[Logger] = None,
    ):
        super(ConstraintBasedDynamics, self).__init__(
            transition=transition,
            learned_reward=learned_reward,
            reward_mech=reward_mech,
            learned_termination=learned_termination,
            termination_mech=termination_mech,
            optim_lr=optim_lr,
            weight_decay=weight_decay,
            optim_eps=optim_eps,
            logger=logger,
        )
        # self.cmi_test: Optional[EnsembleMLP] = None
        # self.build_cmi_test()
        #
        # self.cmi_test_optimizer = torch.optim.Adam(
        #     self.cmi_test.parameters(),
        #     lr=optim_lr,
        #     weight_decay=weight_decay,
        #     eps=optim_eps,
        # )
        # self.learn_mech.append("cmi_test")
        # self.total_epoch["cmi_test"] = 0
        # self._MECH_TO_VARIABLE["cmi_test"] = self._MECH_TO_VARIABLE["transition"]

        for mech in self.learn_mech:
            if hasattr(getattr(self, mech), "input_mask"):
                setattr(self, "{}_oracle_mask".format(mech), None)
                setattr(self, "{}_history_mask".format(mech), torch.ones(getattr(self, mech).input_mask.shape).to(self.device))

    def build_cmi_test(self):
        self.cmi_test = TransitionConditionalMutualInformationTest(
            obs_size=self.transition.obs_size,
            action_size=self.transition.action_size,
            ensemble_num=1,
            elite_num=1,
            residual=self.transition.residual,
            learn_logvar_bounds=self.transition.learn_logvar_bounds,
            num_layers=4,
            hid_size=200,
            activation_fn_cfg=self.transition.activation_fn_cfg,
            device=self.transition.device,
        )

    def set_oracle_mask(self, mech: str, mask: TensorType):
        assert hasattr(self, "{}_oracle_mask".format(mech))
        setattr(self, "{}_oracle_mask".format(mech), to_tensor(mask))

    def learn(
        self,
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
        **kwargs
    ):
        train_dataset, val_dataset = self.dataset_split(
            replay_buffer,
            validation_ratio,
            batch_size,
            shuffle_each_epoch,
            bootstrap_permutes,
        )

        for mech in self.learn_mech:
            if hasattr(self, "{}_oracle_mask".format(mech)):
                getattr(self, mech).set_input_mask(getattr(self, "{}_oracle_mask".format(mech)))

            best_weights: Optional[Dict] = None
            epoch_iter = range(longest_epoch) if longest_epoch > 0 else itertools.count()
            epochs_since_update = 0

            best_val_loss = self.evaluate(val_dataset, mech=mech).mean(dim=(1, 2))

            for epoch in epoch_iter:
                train_loss = self.train(train_dataset, mech=mech)
                val_loss = self.evaluate(val_dataset, mech=mech).mean(dim=(1, 2))

                maybe_best_weights = self.maybe_get_best_weights(
                    best_val_loss,
                    val_loss,
                    mech,
                    improvement_threshold,
                )
                if maybe_best_weights:
                    # best loss
                    best_val_loss = torch.minimum(best_val_loss, val_loss)
                    best_weights = maybe_best_weights
                    epochs_since_update = 0
                else:
                    epochs_since_update += 1

                # log
                self.total_epoch[mech] += 1
                if self.logger is not None:
                    self.logger.record("{}/epoch".format(mech), epoch)
                    self.logger.record("{}/train_dataset_size".format(mech), train_dataset.num_stored)
                    self.logger.record("{}/val_dataset_size".format(mech), val_dataset.num_stored)
                    self.logger.record("{}/train_loss".format(mech), train_loss.mean().item())
                    self.logger.record("{}/val_loss".format(mech), val_loss.mean().item())
                    self.logger.record("{}/best_val_loss".format(mech), best_val_loss.mean().item())
                    self.logger.dump(self.total_epoch[mech])
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
        threshold: float = 0.01,
    ):
        improvement = (best_val_loss - val_loss) / torch.abs(best_val_loss)
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

        if best_weights is not None:
            model.load_state_dict(best_weights)
        sorted_indices = np.argsort(best_val_loss.tolist())
        elite_models = sorted_indices[: model.elite_num]
        model.set_elite_members(elite_models)
