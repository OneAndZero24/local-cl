import json

import omegaconf
import wandb


def _parse(x, r=[], prefix=""): # dirty solution
    tmp = [f"{prefix}{k}: {v}" for k, v in x.items() if k not in ("log_dir")]
    for i in tmp:
        if (": {" in i):
            a, b = i.split(': {', 1)
            b = b.replace("'", '"')
            _parse(json.loads('{'+b), r, a+'_')
        else:
            r.append(i[:64])
    return r

def setup_wandb(config: omegaconf.DictConfig):
    '''
    Sets up W&B run based on config.
    '''

    group, name = config.exp.log_dir.parts[-2:]
    wandb_config = omegaconf.OmegaConf.to_container(
        config, resolve = True, throw_on_missing = True
    )
    tags = _parse(config.exp)
    wandb.init(
        entity = config.wandb.entity,
        project = config.wandb.project,
        dir = config.exp.log_dir,
        group = group,
        name = name,
        config = wandb_config,
        sync_tensorboard = False,
        tags=tags
    )
