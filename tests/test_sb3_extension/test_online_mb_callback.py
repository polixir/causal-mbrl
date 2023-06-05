from typing import cast

from stable_baselines3 import SAC
import gym
import emei

from stable_baselines3.common.buffers import ReplayBuffer

from cmrl.sb3_extension.online_mb_callback import OnlineModelBasedCallback
from cmrl.utils.creator import parse_space
from cmrl.models.causal_mech.oracle_mech import OracleMech
from cmrl.models.dynamics import Dynamics
from cmrl.models.fake_env import VecFakeEnv


def test_callback():
    env = cast(emei.EmeiEnv, gym.make("BoundaryInvertedPendulumSwingUp-v0", freq_rate=1, time_step=0.02))
    reward_fn = env.get_reward
    termination_fn = env.get_terminal
    get_init_obs_fn = env.get_batch_init_obs

    obs_variables = parse_space(env.state_space, "obs")
    act_variables = parse_space(env.action_space, "act")
    next_obs_variables = parse_space(env.state_space, "next_obs")

    transition = OracleMech(
        name="transition",
        input_variables=obs_variables + act_variables,
        output_variables=next_obs_variables,
    )

    dynamics = Dynamics(transition, env.state_space, env.action_space)
    real_replay_buffer = ReplayBuffer(
        100, env.state_space, env.action_space, device="cpu", handle_timeout_termination=False
    )

    fake_env = VecFakeEnv(
        num_envs=1,
        observation_space=env.state_space,
        action_space=env.action_space,
        dynamics=dynamics,
        reward_fn=reward_fn,
        termination_fn=termination_fn,
        get_init_obs_fn=get_init_obs_fn,
    )

    callback = OnlineModelBasedCallback(env, dynamics, real_replay_buffer, freq_train_model=20, longest_epoch=1)

    model = SAC("MlpPolicy", fake_env, verbose=1)
    model.learn(total_timesteps=100, log_interval=4, callback=callback)
