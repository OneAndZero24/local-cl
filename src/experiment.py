import logging

from omegaconf import DictConfig
from hydra.utils import instantiate

import torch
from torch.utils.data import DataLoader

import wandb

from tqdm import tqdm

from util.fabric import setup_fabric
 

log = logging.getLogger(__name__)


def get_scenarios(config: DictConfig):
    dataset_partial = instantiate(config.dataset)
    train_dataset = dataset_partial(train=True)
    test_dataset = dataset_partial(train=False)
    
    scenario_partial = instantiate(config.scenario)
    train_scenario = scenario_partial(train_dataset)
    test_scenario = scenario_partial(test_dataset)

    return train_scenario, test_scenario


def experiment(config: DictConfig):
    """
    Full training and testing on given scenario.
    """

    log.info(f'Initializing scenarios')
    train_scenario, test_scenario = get_scenarios(config)

    log.info(f'Launching Fabric')
    fabric = setup_fabric(config)

    log.info(f'Building model')
    fabric.setup(instantiate(config.model))

    for task_id, (train_task, test_task) in enumerate(zip(train_scenario, test_scenario)):
        log.info(f'Task {task_id + 1}/{len(train_scenario)}')

        log.info(f'Setting up dataloaders')
        train_loader = fabric.setup_dataloaders(DataLoader(
            train_task, 
            batch_size=config.exp.batch_size, 
            shuffle=True, 
            generator=torch.Generator(device=fabric.device)
        ))
        test_loader = fabric.setup_dataloaders(DataLoader(
            test_task, 
            batch_size=1, 
            shuffle=False, 
            generator=torch.Generator(device=fabric.device)
        ))

        with fabric.init_tensor():
            for epoch in range(config.exp.epochs):
                log.info(f'Epoch {epoch + 1}/{config.exp.epochs}')
                train(train_loader)
                test(test_loader)

            # TODO proper grad flow in LwF
            # TODO increment classifier
            # TODO fabric increment classifier


def train():
    """
    Train one epoch.
    """
    
    pass


def test():
    """
    Test one epoch.
    """
    
    pass