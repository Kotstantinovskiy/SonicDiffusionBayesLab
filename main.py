import argparse
import os

from omegaconf import OmegaConf

from src.registry import methods_registry
from src.utils.model_utils import setup_seed


def main():
    parser = argparse.ArgumentParser(description="Sonic Diffusion")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to the config file"
    )
    args = parser.parse_args()

    config = OmegaConf.load(os.path.join("./configs", args.config))
    setup_seed(config.experiment.get("seed", 29))

    methods_registry[config.experiment.method](config).run_experiment()


if __name__ == "__main__":
    main()
