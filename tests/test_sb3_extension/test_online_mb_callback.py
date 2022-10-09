from typing import cast

from stable_baselines3 import SAC
from omegaconf import DictConfig
import gym
import emei

from cmrl.util.env import make_env
from cmrl.util.creator import create_dynamics, create_replay_buffer
from cmrl.sb3_extension.online_mb_callback import OnlineModelBasedCallback
from cmrl.models.dynamics import ConstraintBasedDynamics, PlainEnsembleDynamics
from cmrl.models.transition.one_step.plain_ensemble import PlainEnsembleGaussianTransition
from cmrl.models.fake_env import VecFakeEnv


def test_callback():
    env = cast(emei.EmeiEnv, gym.make("BoundaryInvertedPendulumSwingUp-v0", freq_rate=1, time_step=0.02))
    term_fn = env.get_terminal
    reward_fn = env.get_reward
    init_obs_fn = env.get_batch_init_obs

    transition = PlainEnsembleGaussianTransition(obs_size=5, action_size=1)

    dynamics = PlainEnsembleDynamics(
        transition=transition,
        learned_reward=False,
        reward_mech=reward_fn,
        learned_termination=False,
        termination_mech=term_fn,
    )

    fake_env = VecFakeEnv(16, env.observation_space, env.action_space)
    fake_env.set_up(dynamics, reward_fn, term_fn, init_obs_fn)

    callback = OnlineModelBasedCallback(env, dynamics)

    model = SAC("MlpPolicy", fake_env, verbose=1)
    model.learn(total_timesteps=100, log_interval=4, callback=callback)
