import gym
import emei
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.data import DataLoader
from torch.utils.data import default_collate

from cmrl.models.causal_mech.plain_mech import PlainMech
from cmrl.types import Variable, ContinuousVariable, DiscreteVariable
from cmrl.models.data_loader import BufferDataset, EnsembleBufferDataset, collate_fn
from cmrl.algorithms.util import load_offline_data
from cmrl.models.util import parse_space, create_decoders, create_encoders


def test_single_dim_continuous():
    env = gym.make("BoundaryInvertedPendulumSwingUp-v0", freq_rate=1, time_step=0.02)
    # assert isinstance(env, emei.EmeiEnv)

    real_replay_buffer = ReplayBuffer(
        int(1e5), env.observation_space, env.action_space, "cpu", handle_timeout_termination=False
    )
    load_offline_data(env, real_replay_buffer, "SAC-expert-replay", use_ratio=0.01)

    # test for transition
    tran_dataset = EnsembleBufferDataset(
        real_replay_buffer, env.observation_space, env.action_space, is_valid=False, mech="transition"
    )
    train_loader = DataLoader(tran_dataset, batch_size=8, collate_fn=collate_fn)
    valid_dataset = EnsembleBufferDataset(
        real_replay_buffer, env.observation_space, env.action_space, is_valid=True, mech="transition"
    )
    valid_loader = DataLoader(valid_dataset, batch_size=8, collate_fn=collate_fn)

    node_dim = 16

    input_variables = parse_space(env.observation_space, "obs") + parse_space(env.action_space, "act")
    variable_encoders = create_encoders(input_variables, node_dim=node_dim)
    output_variables = parse_space(env.observation_space, "next_obs")
    variable_decoders = create_decoders(output_variables, node_dim=node_dim)

    mech = PlainMech(
        input_variables=input_variables,
        output_variables=output_variables,
        node_dim=node_dim,
        variable_encoders=variable_encoders,
        variable_decoders=variable_decoders,
    )

    mech.learn(train_loader, valid_loader)
