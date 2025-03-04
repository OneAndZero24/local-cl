import logging

from omegaconf import DictConfig
from hydra.utils import instantiate

import torch
from torch.utils.data import DataLoader

import wandb

from tqdm import tqdm

from util.fabric import setup_fabric
from model import IncrementalClassifier
from src.method.method_plugin_abc import MethodPluginABC
 

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


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

    if config.exp.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    stop_task = None
    if 'stop_after_task' in config.exp:
        stop_task = config.exp.stop_after_task

    save_model = False
    if 'model_path' in config.exp:
        save_model = True
        model_path = config.exp.model_path

    log.info(f'Initializing scenarios')
    train_scenario, test_scenario = get_scenarios(config)

    log.info(f'Launching Fabric')
    fabric = setup_fabric(config)

    log.info(f'Building model')
    model = fabric.setup(instantiate(config.model))

    log.info(f'Setting up method')
    method = instantiate(config.method)(model)

    gen_cm = config.exp.gen_cm
    log_per_batch = config.exp.log_per_batch

    log.info(f'Setting up dataloaders')
    train_tasks = []
    test_tasks = []
    for train_task, test_task in zip(train_scenario, test_scenario):
        train_tasks.append(fabric.setup_dataloaders(DataLoader(
            train_task, 
            batch_size=config.exp.batch_size, 
            shuffle=True, 
            generator=torch.Generator(device=fabric.device)
        )))
        test_tasks.append(fabric.setup_dataloaders(DataLoader(
            test_task, 
            batch_size=1, 
            shuffle=False, 
            generator=torch.Generator(device=fabric.device)
        )))

    avg_acc = 0.0
    for task_id, (train_task, test_task) in enumerate(zip(train_tasks, test_tasks)):
        log.info(f'Task {task_id + 1}/{len(train_scenario)}')

        if isinstance(method.module.head, IncrementalClassifier):
            log.info(f'Incrementing model head')
            method.module.head.increment(train_task.dataset.get_classes())

        log.info(f'Setting up task')
        method.setup_task(task_id)

        with fabric.init_tensor():
            for epoch in range(config.exp.epochs):
                lastepoch = (epoch == config.exp.epochs-1)
                log.info(f'Epoch {epoch + 1}/{config.exp.epochs}')
                train(method, train_task, task_id, log_per_batch)
                acc = test(method, test_task, task_id, gen_cm, log_per_batch)
                if lastepoch:
                    avg_acc = 0.0
                    avg_acc += acc
                if task_id > 0:
                    for j in range(task_id-1, -1, -1):
                        acc = test(method, test_tasks[j], j, gen_cm, log_per_batch, cm_suffix=f' after {task_id}')
                        if lastepoch:
                            avg_acc += acc
        avg_acc /= task_id+1
        wandb.log({f'avg_acc': avg_acc})

        if stop_task is not None and task_id == stop_task:
            break
    
    if save_model:
        log.info(f'Saving model')
        torch.save(model.state_dict(), config.exp.model_path)


def train(method: MethodPluginABC, dataloader: DataLoader, task_id: int, log_per_batch: bool):
    """
    Train one epoch.
    """

    method.module.train()
    avg_loss = 0.0
    for batch_idx, (X, y, _) in enumerate(tqdm(dataloader)):
        loss, preds = method.forward(X, y, task_id)

        method.backward(loss)

        avg_loss += loss
        if log_per_batch:
            wandb.log({f'Loss/train/{task_id}/per_batch': loss})

    avg_loss /= len(dataloader)
    wandb.log({f'Loss/train/{task_id}': avg_loss})


def test(method: MethodPluginABC, dataloader: DataLoader, task_id: int, gen_cm: bool, log_per_batch: bool, cm_suffix: str = '') -> float:
    """
    Test one epoch.
    """

    method.module.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        avg_loss = 0.0
        if gen_cm:
            y_total = []
            preds_total = []
        for batch_idx, (X, y, _) in enumerate(tqdm(dataloader)):
            loss, preds = method.forward(X, y, task_id)
            avg_loss += loss

            _, preds = torch.max(preds.data, 1)
            total += y.size(0)
            correct += (preds == y).sum().item()
            if log_per_batch:
                wandb.log({f'Loss/test/{task_id}/per_batch': loss})

            if gen_cm:
                y_total.extend(y.cpu().numpy())
                preds_total.extend(preds.cpu().numpy())

        avg_loss /= len(dataloader)
        log.info(f'Accuracy of the model on the test images (task {task_id}): {100 * correct / total:.2f}%')
        wandb.log({f'Loss/test/{task_id}': avg_loss})
        wandb.log({f'Accuracy/test/{task_id}': 100 * correct / total})
        if gen_cm:
            title = f'Confusion matrix {str(task_id)+cm_suffix}'
            wandb.log({title: 
                wandb.plot.confusion_matrix(probs=None, y_true=y_total, preds=preds_total, title=title)}
            )
        return 100 * correct / total