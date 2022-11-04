import gym
import emei
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.data import DataLoader

from cmrl.models.causal_mech.plain_mech import PlainMech
from cmrl.types import Variable, ContinuousVariable, DiscreteVariable
from cmrl.models.data_loader import BufferDataset, EnsembleBufferDataset
from cmrl.algorithms.util import load_offline_data


def test_single_dim_continuous():
    env = gym.make("BoundaryInvertedPendulumSwingUp-v0", freq_rate=1, time_step=0.02)
    # assert isinstance(env, emei.EmeiEnv)

    real_replay_buffer = ReplayBuffer(
        int(1e5), env.observation_space, env.action_space, "cpu", handle_timeout_termination=False
    )
    load_offline_data(env, real_replay_buffer, "SAC-expert-replay", use_ratio=0.01)

    # test for transition
    dataset = EnsembleBufferDataset(real_replay_buffer, env.observation_space, env.action_space, mech="transition")
    loader = DataLoader(dataset, batch_size=8)

    node_dim = 1

    input_variables = [ContinuousVariable(name="obs_0", dim=node_dim), ContinuousVariable(name="act_0", dim=node_dim)]
    output_variables = [ContinuousVariable(name="obs_0", dim=node_dim)]

    mech = PlainMech(
        input_variables=input_variables,
        output_variables=output_variables,
        node_dim=node_dim,
        variable_encoders={"state0": None, "action0": None},
        variable_decoders={"state0": None},
    )
