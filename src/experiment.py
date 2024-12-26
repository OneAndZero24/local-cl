import logging

from omegaconf import DictConfig
from hydra.utils import instantiate

import torch
from torch.utils.data import DataLoader

import wandb

from tqdm import tqdm

from util.fabric import setup_fabric
from model import IncrementalClassifier
from method.method_abc import MethodABC
 

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
    model = fabric.setup(instantiate(config.model))

    log.info(f'Setting up method')
    method = instantiate(config.method)(model)

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

        if isinstance(method.module.head, IncrementalClassifier):
            log.info(f'Incrementing model head')
            method.module.head.increment(train_task.get_classes())

        log.info(f'Setting up task')
        method.setup_task(task_id)

        with fabric.init_tensor():
            for epoch in range(config.exp.epochs):
                log.info(f'Epoch {epoch + 1}/{config.exp.epochs}')
                train(method, train_loader, task_id, epoch)
                test(method, test_loader, task_id, epoch)
                if task_id > 0:
                    for j in range(task_id-1, -1, -1):
                        test(method, test_loader, j, epoch, cm_suffix=f' after {task_id}')


def train(method: MethodABC, dataloader: DataLoader, task_id: int, epoch: int):
    """
    Train one epoch.
    """

    method.module.train()
    avg_loss = 0.0
    for batch_idx, (X, y, _) in enumerate(tqdm(dataloader)):
        loss, preds = method.forward(X, y)

        method.backward(loss)

        avg_loss += loss
        wandb.log({f'Loss/train/{task_id}/per_batch': loss}, (epoch+1)*len(dataloader) + batch_idx)

    avg_loss /= len(dataloader)
    wandb.log({f'Loss/{task_id}/train': avg_loss}, epoch+1)


def test(method: MethodABC, dataloader: DataLoader, task_id: int, epoch: int, cm_suffix: str = ''):
    """
    Test one epoch.
    """

    method.module.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        avg_loss = 0.0
        y_total = []
        preds_total = []
        for batch_idx, (X, y, _) in enumerate(tqdm(dataloader)):
            loss, preds = method.forward(X, y)
            avg_loss += loss

            preds, _ = torch.max(preds.data, 1)
            total += y.size(0)
            correct += (preds == y).sum().item()

            y_total.extend(y.cpu().numpy())
            preds_total.extend(preds.cpu().numpy())
            wandb.log({f'Loss/test/{task_id}/per_batch': loss}, (epoch+1)*len(dataloader) + batch_idx)

        avg_loss /= len(dataloader)
        log.info(f'Accuracy of the model on the test images: {100 * correct / total:.2f}%')
        wandb.log({f'Confusion matrix {str(task_id)+cm_suffix}' : 
                   wandb.plot.confusion_matrix(probs=None,y_true=y_total, preds=preds_total)})
        wandb.log({f'Loss/{task_id}/test': avg_loss}, epoch+1)
        wandb.log({f'Accuracy/{task_id}/test': 100 * correct / total}, epoch+1)
