import pyrootutils
import shutil

from omegaconf import DictConfig
import hydra
from hydra.utils import call

import util


@hydra.main(version_base = None, config_path = "../config", config_name = "config")
def main(config: DictConfig):
    util.preprocess_config(config)
    util.setup_wandb(config)
    call(config.exp.run_func, config)
    if config.exp.cleanup:
        shutil.rmtree(config.exp.log_dir)

if __name__ == "__main__":
    pyrootutils.setup_root(
        search_from=__file__,
        indicator="requirements.txt",
        project_root_env_var=True,
        dotenv=True,
        pythonpath=True,
        cwd=True,
    )
    main()