import hydra
from hydra.utils import instantiate
import wandb
from omegaconf import DictConfig, OmegaConf
from emei.core import get_params_str


@hydra.main(version_base=None, config_path="conf", config_name="main")
def run(cfg: DictConfig):
    if cfg.wandb:
        wandb.init(
            project="causal-mbrl",
            group=cfg.exp_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            sync_tensorboard=True,
        )

    algo = instantiate(cfg.algorithm.algo)(cfg=cfg)
    algo.learn()


if __name__ == "__main__":
    OmegaConf.register_new_resolver("to_str", get_params_str)
    run()
