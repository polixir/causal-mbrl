import argparse
import collections
import math
import pathlib
from typing import List, Optional

import matplotlib as mpl
import matplotlib.axes as maxes
import matplotlib.lines as mlines
import matplotlib.pylab as plt
import numpy as np
from matplotlib.widgets import Button, RadioButtons, Slider

import cmrl.utils.creator
import cmrl.utils.env
from cmrl.utils.config import load_hydra_cfg
from cmrl.utils.creator import create_dynamics, create_agent
from cmrl.models.fake_env import get_penalty

mpl.use("TKAgg")
SIN_COS_BINDINGS = {"BoundaryInvertedPendulumSwingUp-v0": [1]}


def set_ylim(y_min, y_max, ax):
    if y_max - y_min > 0.1:
        obs_y_lim = [y_min - 0.05, y_max + 0.05]
    else:
        obs_y_lim = [y_min - 0.025, y_max + 0.025]
    ax.set_ylim(*obs_y_lim)


class DatasetEvaluator:
    WIDGET_LEFT = 0.05
    WIDGET_RIGHT = 0.25
    WIDGET_TOP = 0.95
    WIDGET_BOTTOM = 0.05
    PLOT_LEFT = 0.3

    IN_DIM_BUTTON_WIDTH = 0.15
    IN_DIM_BUTTON_ALL_HEIGHT = 0.2

    OUT_DIM_BUTTON_WIDTH = 0.15
    OUT_DIM_BUTTON_ALL_HEIGHT = 0.2

    SLIDER_WIDTH = 0.15
    SLIDER_ALL_HEIGHT = 0.4

    DRAW_BUTTON_WIDTH = 0.1
    DRAW_BUTTON_HEIGHT = 0.1

    def __init__(
        self,
        model_dir: str,
        batch_size: int = 256,
        plot_dot_num=1024,
        draw_diff=True,
        range_quantile=0,
        device="cuda:0",
        penalty_coeff=None,
    ):
        self.model_path = pathlib.Path(model_dir)
        self.batch_size = batch_size
        self.plot_dot_num = plot_dot_num
        self.draw_diff = draw_diff
        assert 0 <= range_quantile <= 50
        self.range_quantile = range_quantile

        self.cfg = load_hydra_cfg(self.model_path)
        self.cfg.device = device
        self.env, *_ = cmrl.utils.env.make_env(self.cfg)
        if penalty_coeff is None:
            self.penalty_coeff = self.cfg.task.penalty_coeff
        else:
            self.penalty_coeff = penalty_coeff

        self.dynamics = create_dynamics(
            self.cfg,
            self.env.state_space,
            self.env.action_space,
        )
        if not self.cfg.transition.discovery:
            self.dynamics.transition.set_oracle_graph(self.env.get_transition_graph())
        if self.cfg.reward_mech.learn and not self.cfg.reward_mech.discovery:
            self.dynamics.reward_mech.set_oracle_graph(self.env.get_reward_mech_graph())
        if self.cfg.termination_mech.learn and not self.cfg.termination_mech.discovery:
            self.dynamics.termination_mech.set_oracle_graph(self.env.get_termination_mech_graph())
        self.dynamics.transition.load(self.model_path / "transition")

        self.bindings = []
        self.obs_range, self.action_range = self.get_range()

        self.range = np.concatenate([self.obs_range, self.action_range], axis=0)
        self.real_obs_dim_num = self.env.state_space.shape[0]
        self.compact_obs_dim_num, self.action_dim_num = (
            self.obs_range.shape[0],
            self.action_range.shape[0],
        )
        self.compact_in_dim_num = self.compact_obs_dim_num + self.action_dim_num
        self.real_in_dim_num = self.real_obs_dim_num + self.action_dim_num
        self.out_dim_num = self.real_obs_dim_num

        self.real_compact_obs_mapping = {}
        self.real_obs_meaning = {}
        self.in_labels, self.out_labels = self.get_labels()

        self.current_in_dim = 0
        self.current_out_dim = 0
        self.current_fixed_value = []

        # create plot
        self.obs_ax: Optional[maxes.Axes] = None
        self.reward_ax: Optional[maxes.Axes] = None
        self.predict_line: Optional[mlines.Line2D] = None
        self.ground_truth_line: Optional[mlines.Line2D] = None
        self.reward_line: Optional[mlines.Line2D] = None
        self.penalized_reward_line: Optional[mlines.Line2D] = None
        self.in_dim_button: Optional[RadioButtons] = None
        self.out_dim_button: Optional[RadioButtons] = None
        self.dim_value_sliders: List[Slider] = []
        self.draw_button: Optional[Button] = None
        self.create_plot()

    def get_labels(self):
        obs_labels = []
        for dim in range(self.real_obs_dim_num):
            if dim in self.bindings:
                obs_labels.append("obs:{}+{}".format(dim + 1, dim + 2))
                self.real_compact_obs_mapping[dim] = len(obs_labels) - 1
                self.real_obs_meaning[dim] = "sin"
            elif dim - 1 in self.bindings:
                self.real_compact_obs_mapping[dim] = len(obs_labels) - 1
                self.real_obs_meaning[dim] = "cos"
                continue
            else:
                obs_labels.append("obs:{}".format(dim + 1))
                self.real_compact_obs_mapping[dim] = len(obs_labels) - 1
                self.real_obs_meaning[dim] = "normal"
        assert len(obs_labels) == self.compact_obs_dim_num

        in_labels = obs_labels + [f"action:{dim + 1}" for dim in range(self.action_dim_num)]
        out_labels = [f"obs:{dim + 1}" for dim in range(self.real_obs_dim_num)]
        return in_labels, out_labels

    def create_plot(self):
        current_height = self.WIDGET_TOP

        fig, self.obs_ax = plt.subplots(1, 1)
        self.reward_ax = self.obs_ax.twinx()
        plt.subplots_adjust(left=self.PLOT_LEFT)
        # in and out dim button
        current_height -= self.IN_DIM_BUTTON_ALL_HEIGHT
        ax = plt.axes(
            [
                self.WIDGET_LEFT,
                current_height,
                self.IN_DIM_BUTTON_WIDTH,
                self.IN_DIM_BUTTON_ALL_HEIGHT,
            ]
        )
        self.in_dim_button = RadioButtons(ax, self.in_labels)
        current_height -= self.OUT_DIM_BUTTON_ALL_HEIGHT
        ax = plt.axes(
            [
                self.WIDGET_LEFT,
                current_height,
                self.OUT_DIM_BUTTON_WIDTH,
                self.OUT_DIM_BUTTON_ALL_HEIGHT,
            ]
        )
        self.out_dim_button = RadioButtons(ax, self.out_labels)

        def in_dim_button_click(label):
            self.current_in_dim = self.in_labels.index(label)

        def out_dim_button_click(label):
            self.current_out_dim = self.out_labels.index(label)

        self.in_dim_button.on_clicked(in_dim_button_click)
        self.out_dim_button.on_clicked(out_dim_button_click)
        # slider
        for dim in range(self.compact_in_dim_num):
            slider_height = self.SLIDER_ALL_HEIGHT / self.compact_in_dim_num
            current_height -= slider_height
            ax = plt.axes([self.WIDGET_LEFT, current_height, self.SLIDER_WIDTH, slider_height])
            slider = Slider(
                ax,
                self.in_labels[dim],
                self.range[dim][0],
                self.range[dim][1],
                (self.range[dim][0] + self.range[dim][1]) / 2,
            )
            self.dim_value_sliders.append(slider)
            self.current_fixed_value.append((self.range[dim][0] + self.range[dim][1]) / 2)

            # cause by late binding, see:
            # https://stackoverflow.com/questions/3431676/creating-functions-in-a-loop
            def slider_changed(value, dim=dim):
                self.current_fixed_value[dim] = value

            slider.on_changed(slider_changed)
        # draw
        current_height -= self.DRAW_BUTTON_HEIGHT
        ax = plt.axes(
            [
                self.WIDGET_LEFT,
                current_height,
                self.DRAW_BUTTON_WIDTH,
                self.DRAW_BUTTON_HEIGHT,
            ]
        )
        self.draw_button = Button(ax, "draw")
        self.draw_button.on_clicked(self.draw)

    def get_range(self, dataset_type="SAC-expert-replay"):
        data_dict = self.env.get_dataset(dataset_type)
        obs_min = np.percentile(data_dict["observations"], self.range_quantile, axis=0)
        obs_max = np.percentile(data_dict["observations"], 100 - self.range_quantile, axis=0)
        action_min = np.percentile(data_dict["actions"], self.range_quantile, axis=0)
        action_max = np.percentile(data_dict["actions"], 100 - self.range_quantile, axis=0)
        obs_range, action_range = np.array(list(zip(obs_min, obs_max))), np.array(list(zip(action_min, action_max)))

        # if basic_env_name in SIN_COS_BINDINGS:
        #     self.bindings = SIN_COS_BINDINGS[basic_env_name]
        #     for idx, binding_idx in enumerate(self.bindings):
        #         theta_idx = binding_idx - idx
        #         obs_range = np.delete(obs_range, [binding_idx, binding_idx + 1], axis=0)
        #         obs_range = np.insert(obs_range, theta_idx, np.array([0, 2 * np.pi]), axis=0)
        return obs_range, action_range

    def build_model_in(self):
        x = np.linspace(*self.range[self.current_in_dim], self.plot_dot_num, dtype=np.float32)
        compact_model_in = np.empty([self.plot_dot_num, self.compact_in_dim_num], dtype=np.float32)
        for dim in range(self.compact_in_dim_num):
            if dim == self.current_in_dim:
                compact_model_in[:, dim] = x.copy()
            else:
                compact_model_in[:, dim] = np.full(self.plot_dot_num, self.current_fixed_value[dim], dtype=np.float32)

        real_model_in = np.empty([self.plot_dot_num, self.real_in_dim_num], dtype=np.float32)
        for dim in range(self.real_in_dim_num):
            if dim < self.real_obs_dim_num:  # is an obs
                compact_dim = self.real_compact_obs_mapping[dim]
                if self.real_obs_meaning[dim] == "normal":
                    real_model_in[:, dim] = compact_model_in[:, compact_dim].copy()
                elif self.real_obs_meaning[dim] == "sin":
                    real_model_in[:, dim] = np.sin(compact_model_in[:, compact_dim].copy())
                elif self.real_obs_meaning[dim] == "cos":
                    real_model_in[:, dim] = np.cos(compact_model_in[:, compact_dim].copy())
            else:  # is an action
                compact_dim = dim - (self.real_obs_dim_num - self.compact_obs_dim_num)
                real_model_in[:, dim] = compact_model_in[:, compact_dim].copy()
        return x, real_model_in

    def draw(self, event):
        x, model_in = self.build_model_in()
        predict, ground_truth, reward, penalized_reward = self.get_model_out(model_in)

        self.predict_line.set_data(x, predict)
        self.ground_truth_line.set_data(x, ground_truth)
        self.reward_line.set_data(x, reward)
        self.penalized_reward_line.set_data(x, penalized_reward)

        x_lim = [np.min(x), np.max(x)]
        self.obs_ax.set_xlim(*x_lim)

        set_ylim(
            np.min([predict, ground_truth]),
            np.max([predict, ground_truth]),
            self.obs_ax,
        )
        set_ylim(
            np.min([reward, penalized_reward]),
            np.max([reward, penalized_reward]),
            self.reward_ax,
        )
        plt.draw()

    def get_model_out(self, model_in):
        batch_num = math.ceil(self.plot_dot_num / self.batch_size)
        predict = np.empty(self.plot_dot_num)
        ground_truth = np.empty(self.plot_dot_num)
        reward = np.empty(self.plot_dot_num)
        penalized_reward = np.empty(self.plot_dot_num)

        for batch_idx in range(batch_num):
            f, t = self.batch_size * batch_idx, self.batch_size * (batch_idx + 1)

            batch_input = model_in[self.batch_size * batch_idx : self.batch_size * (batch_idx + 1)]
            batch_obs, batch_action = (
                batch_input[:, : self.real_obs_dim_num],
                batch_input[:, self.real_obs_dim_num :],
            )

            predict_next_obs, predict_reward, terminal, info = self.dynamics.step(batch_obs, batch_action)
            gt_next_obs = self.env.get_batch_next_obs(batch_obs, batch_action)
            gt_reward = self.env.get_batch_reward(gt_next_obs)

            if self.draw_diff:
                predict_next_obs -= batch_obs
                gt_next_obs -= batch_obs

            batch_penalty = get_penalty(info["origin-next_obs"])

            predict[f:t] = predict_next_obs[:, self.current_out_dim]
            ground_truth[f:t] = gt_next_obs[:, self.current_out_dim]
            reward[f:t] = gt_reward[:, 0]
            penalized_reward[f:t] = gt_reward[:, 0] - batch_penalty * self.penalty_coeff

        return predict, ground_truth, reward, penalized_reward

    def run(self):
        (self.predict_line,) = self.obs_ax.plot(
            np.linspace(0, 1, 100),
            np.linspace(0, 1, 100),
            color="blue",
            lw=2,
            label="predict",
        )
        (self.ground_truth_line,) = self.obs_ax.plot(
            np.linspace(0, 1, 100),
            np.linspace(0, 1, 100),
            color="black",
            lw=2,
            label="ground truth",
        )
        (self.reward_line,) = self.reward_ax.plot(
            np.linspace(0, 1, 100),
            np.linspace(0, 0.1, 100),
            color="green",
            lw=2,
            label="reward",
        )
        (self.penalized_reward_line,) = self.reward_ax.plot(
            np.linspace(0, 1, 100),
            np.linspace(0, 0.1, 100),
            color="red",
            lw=2,
            label="penalized reward",
        )

        lines = [
            self.predict_line,
            self.ground_truth_line,
            self.reward_line,
            self.penalized_reward_line,
        ]
        labels = [line.get_label() for line in lines]
        self.obs_ax.legend(lines, labels)
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=str, default=None)
    parser.add_argument("--penalty_coeff", type=float, default=None)
    parser.add_argument("--not_draw_diff", action="store_true", default=False)
    args = parser.parse_args()

    evaluator = DatasetEvaluator(args.model_dir, penalty_coeff=args.penalty_coeff, draw_diff=not args.not_draw_diff)

    mpl.rcParams["figure.facecolor"] = "white"
    mpl.rcParams["font.size"] = 14

    evaluator.run()
