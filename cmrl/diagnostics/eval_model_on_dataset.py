import argparse
import collections
import math
import pathlib
from typing import List

import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np

import cmrl.utils.creator
import cmrl.utils.env
from cmrl.utils.config import load_hydra_cfg
from cmrl.utils.transition_iterator import TransitionIterator


class DatasetEvaluator:
    def __init__(self, model_dir: str, dataset: str, batch_size: int = 4096, device="cuda:0"):
        self.model_path = pathlib.Path(model_dir)
        self.batch_size = batch_size

        self.cfg = load_hydra_cfg(self.model_path)
        self.cfg.device = device
        self.env, self.term_fn, self.reward_fn = cmrl.util.env.make_env(self.cfg)

        self.dynamics = cmrl.util.creator.create_dynamics(
            self.cfg.dynamics,
            self.env.state_space.shape,
            self.env.action_space.shape,
            load_dir=self.model_path,
            load_device=device,
        )

        self.replay_buffer = cmrl.util.creator.create_replay_buffer(
            self.cfg,
            self.env.state_space.shape,
            self.env.action_space.shape,
        )

        if hasattr(self.env, "get_dataset"):
            (
                universe,
                basic_env_name,
                params,
                origin_dataset_type,
            ) = self.cfg.task.env.split("___")
            if dataset is None:
                dataset = origin_dataset_type

            self.output_path = self.model_path / "diagnostics" / "eval_on_{}".format(dataset)
            pathlib.Path.mkdir(self.output_path, parents=True, exist_ok=True)

            data_dict = self.env.get_dataset("{}-{}".format(params, dataset))
            self.replay_buffer.add_batch(
                data_dict["observations"],
                data_dict["actions"],
                data_dict["next_observations"],
                data_dict["rewards"],
                data_dict["terminals"].astype(bool) | data_dict["timeouts"].astype(bool),
            )
        else:
            raise NotImplementedError

    def plot_dataset_results(
        self,
        dataset: TransitionIterator,
        hist_bins: int = 20,
        hist_log: bool = True,
    ):
        target_list = collections.defaultdict(list)
        predict_list = collections.defaultdict(list)

        # collect predict and target
        for batch in dataset:
            dynamics_result = self.dynamics.query(batch.batch_obs, batch.batch_action, return_as_np=True)
            for variable in dynamics_result:
                target_list[variable].extend(getattr(batch, variable))
                predict_list[variable].extend(dynamics_result[variable]["mean"].mean(axis=0))
            # add the difference between next-obs and obs
            diff_target = getattr(batch, "batch_next_obs") - getattr(batch, "batch_obs")
            diff_predict = dynamics_result["batch_next_obs"]["mean"].mean(axis=0) - getattr(batch, "batch_obs")
            target_list["batch_diff_next_obs"].extend(diff_target)
            predict_list["batch_diff_next_obs"].extend(diff_predict)

            # large_error_index = np.argwhere(np.abs(diff_target - diff_predict) > 1)
            # for batch_idx, dim in large_error_index:
            #     print("dim:{}\nobs:{}\naction:{}\npredict:{}\ntarget:{}\n".format(dim,
            #                                                                       batch.batch_obs[batch_idx],
            #                                                                       batch.batch_action[batch_idx],
            #                                                                       diff_predict[batch_idx],
            #                                                                       diff_target[batch_idx]))

        target_np, predict_np = {}, {}
        for variable in target_list:
            target_np[variable] = np.array(target_list[variable])
            if len(target_np[variable].shape) == 1:
                target_np[variable] = target_np[variable][:, None]
            predict_np[variable] = np.array(predict_list[variable])

        for variable in target_np:
            # draw predict-and-target plot
            dim_num = target_np[variable].shape[1]
            row_num = math.ceil(math.sqrt(dim_num))
            fig, axs = plt.subplots(row_num, row_num, figsize=(row_num * 8, row_num * 8))
            if isinstance(axs, np.ndarray):
                axis_list = [e for row in axs for e in row]
            else:
                axis_list = [axs]
            for dim in range(dim_num):
                axis = axis_list[dim]
                sort_idx = target_np[variable][:, dim].argsort()
                sorted_predict = predict_np[variable][sort_idx, dim]
                sorted_target = target_np[variable][sort_idx, dim]

                axis.plot(sorted_target, sorted_predict, ".")
                axis.plot(
                    [sorted_target.min(), sorted_target.max()],
                    [sorted_target.min(), sorted_target.max()],
                    linewidth=2,
                    color="k",
                )
            fname = self.output_path / f"target_and_pred-{variable}.png"
            plt.savefig(fname)
            plt.close()

            # draw target distribution plot
            fig, axs = plt.subplots(row_num, row_num, figsize=(row_num * 8, row_num * 8))
            if isinstance(axs, np.ndarray):
                axis_list = [e for row in axs for e in row]
            else:
                axis_list = [axs]
            for dim in range(dim_num):
                axis = axis_list[dim]

                target = target_np[variable][:, dim]
                axis.hist(target, bins=hist_bins, log=hist_log)
            plt.savefig(self.output_path / f"target_dist-{variable}.png")
            plt.close()

    def run(self):
        _, dataset = self.dynamics.dataset_split(self.replay_buffer, validation_ratio=1.0, batch_size=self.batch_size)

        self.plot_dataset_results(dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args()

    evaluator = DatasetEvaluator(args.model_dir, args.dataset)

    mpl.rcParams["figure.facecolor"] = "white"
    mpl.rcParams["font.size"] = 14

    evaluator.run()
