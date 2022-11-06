from stable_baselines3.common.buffers import ReplayBuffer

from cmrl.utils.env import make_env
from cmrl.algorithms.util import load_offline_data

from tests.constants import cfg


def test_load_offline_data():
    env, term_fn, reward_fn, init_obs_fn = make_env(cfg)
    replay_buffer = ReplayBuffer(
        cfg.task.num_steps, env.observation_space, env.action_space, cfg.device, handle_timeout_termination=False
    )

    load_offline_data(cfg, env, replay_buffer)
