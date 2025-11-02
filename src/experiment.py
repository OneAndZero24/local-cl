import logging

from copy import deepcopy
import numpy as np

from omegaconf import DictConfig
from hydra.utils import instantiate

import torch
from torch.utils.data import DataLoader

import wandb

from tqdm import tqdm

from util.fabric import setup_fabric
from model import IncrementalClassifier
from src.method.composer import Composer
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

    calc_bwt = False
    if 'calc_bwt' in config.exp:
        calc_bwt = config.exp.calc_bwt

    calc_fwt = False
    if 'calc_fwt' in config.exp:
        calc_fwt = config.exp.calc_fwt

    acc_table = False
    if 'acc_table' in config.exp:
        acc_table = config.exp.acc_table

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

    N = len(train_scenario)
    R = np.zeros((N, N))
    if calc_fwt:
        b = np.zeros(N)
    for task_id, (train_task, test_task) in enumerate(zip(train_tasks, test_tasks)):
        log.info(f'Task {task_id + 1}/{N}')
        if hasattr(method.module, 'head') and isinstance(method.module.head, IncrementalClassifier) \
            and not config.exp.dil:
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
                if calc_fwt:
                    method_tmp = Composer(
                        deepcopy(method.module), 
                        config.method.criterion, 
                        method.lr,
                        method.criterion_scale,
                        method.reg_type,
                        method.gamma,
                        method.clipgrad,
                        method.retaingraph,
                        method.log_reg
                    )
                    log.info('FWT training pass')
                    method_tmp.setup_task(task_id)
                    train(method_tmp, train_task, task_id, log_per_batch, quiet=True)
                    b[task_id] = test(method_tmp, test_task, task_id, gen_cm, log_per_batch, quiet=True)
                if lastepoch:
                    R[task_id, task_id] = acc
                if task_id > 0:
                    for j in range(task_id-1, -1, -1):
                        acc = test(method, test_tasks[j], j, gen_cm, log_per_batch, cm_suffix=f' after {task_id}')
                        if lastepoch:
                            R[task_id, j] = acc
        wandb.log({f'avg_acc': R[task_id, :task_id+1].mean()})

        if stop_task is not None and task_id == stop_task:
            break
    
    if calc_bwt:
        wandb.log({'bwt': (R[task_id, :task_id]-R.diagonal()[:-1]).mean()})

    if calc_fwt:
        fwt = []
        for i in range(1, task_id+1):
            fwt.append(R[i-1, i]-b[i])
        wandb.log({'fwt': np.array(fwt).mean()})

    if save_model:
        log.info(f'Saving model')
        torch.save(model.state_dict(), config.exp.model_path)

    if acc_table:
        log.info(f'Logging accuracy table')
        wandb.log({"acc_table": wandb.Table(data=R.tolist(), columns=[f"task_{i}" for i in range(N)])})


def train(method: MethodPluginABC, dataloader: DataLoader, task_id: int, log_per_batch: bool, quiet: bool = False):
    """
    Train one epoch.
    """

    method.module.train()
    avg_loss = 0.0
    for batch_idx, (X, y, _) in enumerate(tqdm(dataloader)):
        loss, preds = method.forward(X, y, task_id)

        loss = loss.mean()
        method.backward(loss)

        avg_loss += loss
        if log_per_batch and not quiet:
            wandb.log({f'Loss/train/{task_id}/per_batch': loss})

    avg_loss /= len(dataloader)
    if not quiet:
        wandb.log({f'Loss/train/{task_id}': avg_loss})


def test(method: MethodPluginABC, dataloader: DataLoader, task_id: int, gen_cm: bool, log_per_batch: bool, quiet: bool = False, cm_suffix: str = '') -> float:
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
            if log_per_batch and not quiet:
                wandb.log({f'Loss/test/{task_id}/per_batch': loss})

            if gen_cm:
                y_total.extend(y.cpu().numpy())
                preds_total.extend(preds.cpu().numpy())

        avg_loss /= len(dataloader)
        if not quiet:
            log.info(f'Accuracy of the model on the test images (task {task_id}): {100 * correct / total:.2f}%')
            wandb.log({f'Loss/test/{task_id}': avg_loss})
            wandb.log({f'Accuracy/test/{task_id}': 100 * correct / total})
            if gen_cm:
                title = f'Confusion matrix {str(task_id)+cm_suffix}'
                wandb.log({title: 
                    wandb.plot.confusion_matrix(probs=None, y_true=y_total, preds=preds_total, title=title)}
                )
        return 100 * correct / total