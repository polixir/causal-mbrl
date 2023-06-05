import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from emei.core import get_params_str


@hydra.main(version_base=None, config_path="conf", config_name="main")
def run(cfg: DictConfig):
    algo = instantiate(cfg.algorithm.algo)(cfg=cfg)
    algo.learn()


if __name__ == "__main__":
    OmegaConf.register_new_resolver("to_str", get_params_str)
    run()
