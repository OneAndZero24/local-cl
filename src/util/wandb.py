import omegaconf
import wandb


def setup_wandb(config: omegaconf.DictConfig):
    """
    Sets up W&B run based on config.
    """

    group, name = config.exp.log_dir.parts[-2:]
    wandb_config = omegaconf.OmegaConf.to_container(
        config, resolve = True, throw_on_missing = True
    )
    wandb.init(
        entity = config.wandb.entity,
        project = config.wandb.project,
        dir = config.exp.log_dir,
        group = group,
        name = name,
        config = wandb_config,
        sync_tensorboard = False
    )
    wandb.define_metric("avg_acc", summary="last")
