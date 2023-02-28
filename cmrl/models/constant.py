from omegaconf import DictConfig

NETWORK_CFG = DictConfig(
    dict(
        _target_="cmrl.models.networks.ParallelMLP",
        _partial_=True,
        _recursive_=False,
        hidden_dims=[200, 200],
        bias=True,
        activation_fn_cfg=dict(_target_="torch.nn.SiLU"),
    )
)

ENCODER_CFG = DictConfig(
    dict(
        _target_="cmrl.models.networks.VariableEncoder",
        _partial_=True,
        _recursive_=False,
        output_dim=100,
        hidden_dims=[100],
        bias=True,
        activation_fn_cfg=dict(_target_="torch.nn.SiLU"),
    )
)

DECODER_CFG = DictConfig(
    dict(
        _target_="cmrl.models.networks.VariableDecoder",
        _partial_=True,
        _recursive_=False,
        input_dim=100,
        hidden_dims=[100],
        bias=True,
        activation_fn_cfg=dict(_target_="torch.nn.SiLU"),
    )
)

OPTIMIZER_CFG = DictConfig(
    dict(
        _target_="torch.optim.Adam",
        _partial_=True,
        lr=1e-4,
        weight_decay=1e-5,
        eps=1e-8,
    )
)

SCHEDULER_CFG = DictConfig(
    dict(
        _target_="torch.optim.lr_scheduler.StepLR",
        _partial_=True,
        step_size=1,
        gamma=1,
    )
)
