import argparse
import pathlib
from typing import List
import collections
from typing import Optional
import matplotlib as mpl
import matplotlib.pylab as plt
import matplotlib.lines as mlines
import matplotlib.axes as maxes
import numpy as np
import math
from cmrl.util.config import load_hydra_cfg
import cmrl.util.creator
import cmrl.util.env
from matplotlib.widgets import RadioButtons, Slider, Button

mpl.use("Qt5Agg")


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

    def __init__(self,
                 model_dir: str,
                 batch_size: int = 256,
                 plot_dot_num=1024,
                 draw_diff=True):
        self.model_path = pathlib.Path(model_dir)
        self.batch_size = batch_size
        self.plot_dot_num = plot_dot_num
        self.draw_diff = draw_diff

        self.cfg = load_hydra_cfg(self.model_path)
        self.env, self.term_fn, self.reward_fn = cmrl.util.env.make_env(self.cfg)

        self.dynamics = cmrl.util.creator.create_dynamics(self.cfg.dynamics,
                                                          self.env.observation_space.shape,
                                                          self.env.action_space.shape,
                                                          model_dir=self.model_path, )

        self.obs_range, self.action_range = self.get_range()
        self.obs_dim_num, self.action_dim_num = self.obs_range.shape[0], self.action_range.shape[0]

        self.in_dim_num = self.obs_dim_num + self.action_dim_num
        self.in_labels = [f"obs:{dim + 1}" for dim in range(self.obs_dim_num)] + \
                         [f"action:{dim + 1}" for dim in range(self.action_dim_num)]
        self.range = np.concatenate([self.obs_range, self.action_range], axis=0)
        self.out_labels = [f"obs:{dim + 1}" for dim in range(self.obs_dim_num)]
        if self.dynamics.learned_reward:
            self.out_labels.append("reward")
        if self.dynamics.learned_termination:
            raise NotImplementedError
        self.out_dim_num = len(self.out_labels)

        self.current_in_dim = 0
        self.current_out_dim = 0
        self.current_fixed_value = []

        # create plot
        self.ax: Optional[maxes.Axes] = None
        self.predict_line: Optional[mlines.Line2D] = None
        self.ground_truth_line: Optional[mlines.Line2D] = None
        self.in_dim_button: Optional[RadioButtons] = None
        self.out_dim_button: Optional[RadioButtons] = None
        self.dim_value_sliders: List[Slider] = []
        self.draw_button: Optional[Button] = None
        self.create_plot()

    def create_plot(self):
        current_height = self.WIDGET_TOP

        fig, self.ax = plt.subplots(1, 1)
        plt.subplots_adjust(left=self.PLOT_LEFT)
        # in and out dim button
        current_height -= self.IN_DIM_BUTTON_ALL_HEIGHT
        ax = plt.axes([self.WIDGET_LEFT, current_height, self.IN_DIM_BUTTON_WIDTH, self.IN_DIM_BUTTON_ALL_HEIGHT])
        self.in_dim_button = RadioButtons(ax, self.in_labels)
        current_height -= self.OUT_DIM_BUTTON_ALL_HEIGHT
        ax = plt.axes([self.WIDGET_LEFT, current_height, self.OUT_DIM_BUTTON_WIDTH, self.OUT_DIM_BUTTON_ALL_HEIGHT])
        self.out_dim_button = RadioButtons(ax, self.out_labels)

        def in_dim_button_click(label):
            if label.startswith("obs"):
                self.current_in_dim = int(label[4:]) - 1
            elif label.startswith("action"):
                self.current_in_dim = self.obs_dim_num + int(label[7:]) - 1
            else:
                raise NotImplementedError

        def out_dim_button_click(label):
            if label.startswith("obs"):
                self.current_out_dim = int(label[4:]) - 1
            elif label.startswith("reward"):
                self.current_out_dim = self.obs_dim_num
            else:
                raise NotImplementedError

        self.in_dim_button.on_clicked(in_dim_button_click)
        self.out_dim_button.on_clicked(out_dim_button_click)
        # slider
        for dim in range(self.in_dim_num):
            slider_height = self.SLIDER_ALL_HEIGHT / self.in_dim_num
            current_height -= slider_height
            ax = plt.axes([self.WIDGET_LEFT, current_height, self.SLIDER_WIDTH, slider_height])
            slider = Slider(ax, self.in_labels[dim], self.range[dim][0], self.range[dim][1],
                            (self.range[dim][0] + self.range[dim][1]) / 2)
            self.dim_value_sliders.append(slider)
            self.current_fixed_value.append((self.range[dim][0] + self.range[dim][1]) / 2)

            # cause by late binding, see:
            # https://stackoverflow.com/questions/3431676/creating-functions-in-a-loop
            def slider_changed(value, dim=dim):
                self.current_fixed_value[dim] = value

            slider.on_changed(slider_changed)
        # draw
        current_height -= self.DRAW_BUTTON_HEIGHT
        ax = plt.axes([self.WIDGET_LEFT, current_height, self.DRAW_BUTTON_WIDTH, self.DRAW_BUTTON_HEIGHT])
        self.draw_button = Button(ax, "draw")
        self.draw_button.on_clicked(self.draw)

    def get_range(self, dataset_type="expert-replay", q=5):
        assert 0 < q < 50
        universe, basic_env_name, params, origin_dataset_type = self.cfg.task.env.split("___")
        data_dict = self.env.get_dataset("{}-{}".format(params, dataset_type))
        obs_min = np.percentile(data_dict["observations"], q, axis=0)
        obs_max = np.percentile(data_dict["observations"], 100 - q, axis=0)
        action_min = np.percentile(data_dict["actions"], q, axis=0)
        action_max = np.percentile(data_dict["actions"], 100 - q, axis=0)
        return np.array(list(zip(obs_min, obs_max))), np.array(list(zip(action_min, action_max)))

    def draw(self, event):
        # build model input
        x = np.linspace(*self.range[self.current_in_dim], self.plot_dot_num, dtype=np.float32)
        model_in = np.empty([self.plot_dot_num, self.in_dim_num], dtype=np.float32)
        for dim in range(self.in_dim_num):
            if dim == self.current_in_dim:
                continue
            model_in[:, dim] = np.full(self.plot_dot_num, self.current_fixed_value[dim], dtype=np.float32)
        model_in[:, self.current_in_dim] = x.copy()

        predict, ground_truth = self.get_model_output(model_in)
        self.predict_line.set_data(x, predict)
        self.ground_truth_line.set_data(x, ground_truth)
        x_lim = [np.min(x), np.max(x)]
        y_min, y_max = np.min([predict, ground_truth]), np.max([predict, ground_truth])
        if y_max - y_min > 0.1:
            y_lim = [y_min, y_max]
        else:
            y_lim = [y_min - 0.025, y_max + 0.025]
        self.ax.set_xlim(*x_lim)
        self.ax.set_ylim(*y_lim)
        plt.draw()

    def get_model_output(self, model_in):
        batch_num = math.ceil(self.plot_dot_num / self.batch_size)
        predict = np.empty(self.plot_dot_num)
        ground_truth = np.empty(self.plot_dot_num)

        for batch_idx in range(batch_num):
            batch_input = model_in[self.batch_size * batch_idx: self.batch_size * (batch_idx + 1)]
            batch_obs, batch_action = batch_input[:, :self.obs_dim_num], batch_input[:, self.obs_dim_num:]
            dynamics_result = self.dynamics.query(batch_obs, batch_action, return_as_np=True)
            gt_next_obs, gt_reward, gt_done, _ = self.env.query(batch_obs, batch_action)

            if self.current_out_dim < self.obs_dim_num:  # obs
                batch_predict_obs = dynamics_result["batch_next_obs"]["mean"].mean(0)
                batch_gt_obs = gt_next_obs
                if self.draw_diff:
                    batch_predict_obs -= batch_obs
                    batch_gt_obs -= batch_obs
                batch_predict = batch_predict_obs[:, self.current_out_dim]
                batch_ground_truth = batch_gt_obs[:, self.current_out_dim]
            elif self.current_out_dim == self.obs_dim_num:  # reward
                batch_predict = dynamics_result["batch_reward"]["mean"].mean(0)
                batch_ground_truth = gt_reward
            else:
                raise NotImplementedError
            predict[self.batch_size * batch_idx: self.batch_size * (batch_idx + 1)] = batch_predict
            ground_truth[self.batch_size * batch_idx: self.batch_size * (batch_idx + 1)] = batch_ground_truth
        return predict, ground_truth

    def run(self):
        self.predict_line, = self.ax.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), color="red", lw=2)
        self.ground_truth_line, = self.ax.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), color="blue", lw=2)
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=str, default=None)
    args = parser.parse_args()

    evaluator = DatasetEvaluator(args.model_dir)

    mpl.rcParams["figure.facecolor"] = "white"
    mpl.rcParams["font.size"] = 14

    evaluator.run()
