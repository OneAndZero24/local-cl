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
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from pathlib import Path

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

def activation_visualization(config: DictConfig):
    """
    Visualization for tracking activation changes across continual learning tasks.
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
    log.info('Selecting representative images from each task')
    selected_images = []
    selected_labels = []
    for task_id, test_task in enumerate(test_tasks):
        for X, y, _ in test_task:
            selected_images.append(X)
            selected_labels.append(y.item())
            break
    log.info(f'Selected {len(selected_images)} images (one per task)')
    activation_history = [
        [None for _ in range(N)] for _ in range(N)
    ]
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
        log.info(f'Recording activations for all images after task {task_id}')
        method.module.eval()
        for img_id in range(N):
            img = selected_images[img_id]
            avg_activations = get_layer_activations(method.module, img)
            activation_history[task_id][img_id] = avg_activations
            log.info(f' Task {task_id}, Image {img_id}: {len(avg_activations)} layers recorded')
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
    log.info('Creating activation drift visualization...')
    output_dir = Path(config.exp.log_dir) / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)
    activation_differences = []
    for img_id in range(N):
        baseline_activations = activation_history[img_id][img_id]
        differences = []
        for task_id in range(img_id, N):
            current_activations = activation_history[task_id][img_id]
            diff = np.mean([abs(curr - base) for curr, base in zip(current_activations, baseline_activations)])
            differences.append(diff)
        activation_differences.append(differences)
    plt.style.use('seaborn-v0_8-paper' if 'seaborn-v0_8-paper' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle('Activation Drift Over Tasks', fontsize=16)
    colors = plt.cm.tab10(np.linspace(0, 1, N))
    for img_id in range(N - 1):  # skip final task (no next regularization)
        x_values = list(range(img_id, N - 1))  # stop one before last
        y_values = activation_differences[img_id][:len(x_values)]
        # Plot main line with markers at points except possibly the last
        ax.plot(x_values, y_values,
                linewidth=2.5,
                marker='o',
                markersize=8,
                color=colors[img_id],
                label=f'Task {img_id+1} (class {selected_labels[img_id]})')
        # Place only one image at the beginning
        x = x_values[0]
        y = y_values[0]
        img_np = selected_images[img_id].cpu().squeeze().numpy()
        if img_np.ndim == 3:
            if img_np.shape[0] == 1:
                img_np = img_np[0]
            elif img_np.shape[0] == 3:
                img_np = np.transpose(img_np, (1, 2, 0))
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        imagebox = OffsetImage(img_np, zoom=1.8, cmap='gray' if img_np.ndim == 2 else None)
        ab = AnnotationBbox(imagebox, (x, y),
                            frameon=True,
                            pad=0.3,
                            bboxprops=dict(edgecolor=colors[img_id],
                                           linewidth=2,
                                           facecolor='white'))
        ax.add_artist(ab)
    max_diff = max([max(diffs) if len(diffs) > 0 else 0 for diffs in activation_differences])
    if max_diff == 0:
        max_diff = 1.0
    y_positions = np.linspace(0, max_diff, N)
    for img_id in range(N):
        img_np = selected_images[img_id].cpu().squeeze().numpy()
        if img_np.ndim == 3:
            if img_np.shape[0] == 1:
                img_np = img_np[0]
            elif img_np.shape[0] == 3:
                img_np = np.transpose(img_np, (1, 2, 0))
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        imagebox = OffsetImage(img_np, zoom=2.0, cmap='gray' if img_np.ndim == 2 else None)
        x_pos = N - 0.5
        y_pos = y_positions[img_id]
        ab = AnnotationBbox(imagebox, (x_pos, y_pos),
                            frameon=True,
                            pad=0.3,
                            bboxprops=dict(
                                edgecolor=colors[img_id],
                                linewidth=3,
                                facecolor='white'
                            ))
        ax.add_artist(ab)
    ax.set_xlabel('Task ID', fontsize=14)
    ax.set_ylabel('Average Activation Difference from Baseline', fontsize=14)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', fontsize=14, framealpha=0.95)
    ax.set_xticks(range(N))
    ax.set_xticklabels([f'{i}' for i in range(N)])
    plt.tight_layout()
    output_path = output_dir / 'activation_drift_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    log.info(f'Saved visualization to {output_path}')
    plt.close()
    log.info('Creating detailed per-layer activation plot...')
    num_layers = len(activation_history[0][0]) if activation_history[0][0] else 0
    if num_layers == 0:
        log.warning('No layer activations recorded. Skipping detailed per-layer plot.')
        log.info(f'Main visualization saved to {output_dir}')
        return
    fig, axes = plt.subplots(num_layers, 1, figsize=(14, 4 * num_layers), sharex=True)
    fig.suptitle('Per-Layer Activation Over Tasks', fontsize=16)
    if num_layers == 1:
        axes = [axes]
    for layer_idx in range(num_layers):
        ax = axes[layer_idx]
        for img_id in range(N):
            if img_id == N-1:
                break
            x_values = list(range(img_id, N))
            y_values = [activation_history[task_id][img_id][layer_idx]
                        for task_id in range(img_id, N)]
            ax.plot(x_values, y_values,
                    marker='o',
                    markersize=8,
                    linewidth=2.5,
                    color=colors[img_id],
                    label=f'Sample from Task {img_id+1}')
            # Place only one image at the beginning
            x = x_values[0]
            y = y_values[0]
            img_np = selected_images[img_id].cpu().squeeze().numpy()
            if img_np.ndim == 3:
                if img_np.shape[0] == 1:
                    img_np = img_np[0]
                elif img_np.shape[0] == 3:
                    img_np = np.transpose(img_np, (1, 2, 0))
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
            imagebox = OffsetImage(img_np, zoom=1.2, cmap='gray' if img_np.ndim == 2 else None)
            ab = AnnotationBbox(imagebox, (x, y),
                                frameon=True,
                                pad=0.3,
                                bboxprops=dict(edgecolor=colors[img_id],
                                               linewidth=2,
                                               facecolor='white'))
            ax.add_artist(ab)
        ax.set_title(f'Idx of Interval Activation Layer: {layer_idx+1}', fontsize=14, pad=10)
        ax.set_ylabel('Average Absolute Activation Value', fontsize=14)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=14)
        ax.set_xticks(range(N))
        ax.set_xticklabels([f'{i}' for i in range(N)])
    axes[-1].set_xlabel('Task ID', fontsize=14)
    plt.tight_layout()
    output_path_detailed = output_dir / 'activation_per_layer_detailed.png'
    plt.savefig(output_path_detailed, dpi=300, bbox_inches='tight')
    log.info(f'Saved detailed visualization to {output_path_detailed}')
    plt.close()
    log.info(f'All visualizations saved to {output_dir}')
    exit(0)

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

def get_layer_activations(model, x):
    """
    Get average of absolute values of activations for each layer.
    """
    model.eval()
    activations = []
    with torch.no_grad():
        current = torch.flatten(x, start_dim=1)
        actual_model = model._forward_module if hasattr(model, '_forward_module') else model
        if hasattr(actual_model, 'layers'):
            for layer in actual_model.layers:
                current = layer(current)
                if type(layer).__name__ == "IntervalActivation":
                    avg_act = current.abs().mean().item()
                    activations.append(avg_act)
    return activations