import logging
from copy import deepcopy
import numpy as np
from omegaconf import DictConfig
from hydra.utils import instantiate
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from pathlib import Path
import pickle
import wandb
from tqdm import tqdm

from src.util.fabric import setup_fabric
import torch
from torch.utils.data import DataLoader
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
    log.info('Selecting representative images from each task with the lowest label')
    selected_images = []
    selected_labels = []
    for task_id, test_task in enumerate(test_tasks):
        min_label = float('inf')
        selected_image = None
        for X, y, _ in test_task:
            label = y.item()
            if label < min_label:
                min_label = label
                selected_image = X
        if selected_image is not None:
            selected_images.append(selected_image)
            selected_labels.append(min_label)
        else:
            log.warning(f'No samples found for task {task_id + 1}')
    log.info(f'Selected {len(selected_images)} images (one per task)')
    activation_history = []
    x_points = []
    task_end_snapshots = []
    num_samples_per_task = 20
    for task_id, (train_task, test_task) in enumerate(zip(train_tasks, test_tasks)):
        log.info(f'Task {task_id + 1}/{N}')
        if hasattr(method.module, 'head') and isinstance(method.module.head, IncrementalClassifier) \
                and not config.exp.dil:
            log.info(f'Incrementing model head')
            method.module.head.increment(train_task.dataset.get_classes())
        log.info(f'Setting up task')
        method.setup_task(task_id)
        step = max(1, config.exp.epochs // num_samples_per_task)
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
                if (epoch + 1) % step == 0 or epoch == config.exp.epochs - 1:
                    log.info(f'Recording activations after task {task_id} epoch {epoch}')
                    method.module.eval()
                    avg_activations_list = []
                    for img_id in range(N):
                        img = selected_images[img_id]
                        avg_activations = get_layer_activations(method.module, img)
                        avg_activations_list.append(avg_activations)
                    log.info(f' Task {task_id}, Epoch {epoch}, Image {img_id}: {len(avg_activations)} layers recorded')
                    activation_history.append(avg_activations_list)
                    current_x = task_id + (epoch + 1) / config.exp.epochs
                    x_points.append(current_x)
            task_end_snapshots.append(len(activation_history) - 1)
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
    log.info('Saving essential data for plot regeneration...')
    output_dir = Path(config.exp.log_dir) / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'activation_history.pkl', 'wb') as f:
        pickle.dump(activation_history, f)
    with open(output_dir / 'x_points.pkl', 'wb') as f:
        pickle.dump(x_points, f)
    with open(output_dir / 'task_end_snapshots.pkl', 'wb') as f:
        pickle.dump(task_end_snapshots, f)
    torch.save(selected_images, output_dir / 'selected_images.pt')
    with open(output_dir / 'selected_labels.pkl', 'wb') as f:
        pickle.dump(selected_labels, f)
    generate_plots(activation_history, x_points, task_end_snapshots, selected_images, selected_labels, output_dir)
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
                if type(layer).__name__ == "IntervalActivation" or isinstance(layer, torch.nn.ReLU):
                    avg_act = current.abs().mean().item()
                    activations.append(avg_act)
    return activations


def compute_differences(activation_history, task_end_snapshots, x_points, signed=False):
    N = len(task_end_snapshots)
    differences_lists = []
    x_values_lists = []
    for img_id in range(N):
        baseline_index = task_end_snapshots[img_id]
        baseline_activations = activation_history[baseline_index][img_id]
        differences = []
        x_values_this = []
        for s in range(baseline_index, len(activation_history)):
            current_activations = activation_history[s][img_id]
            if signed:
                diff = np.mean([
                    curr - base for curr, base in zip(current_activations, baseline_activations)
                ])
            else:
                diff = np.mean([
                    abs(curr - base) for curr, base in zip(current_activations, baseline_activations)
                ])
            differences.append(diff)
            x_values_this.append(x_points[s])
        differences_lists.append(differences)
        x_values_lists.append(x_values_this)
    return differences_lists, x_values_lists


def plot_drift(fig, ax, differences, x_lists, colors, selected_images, selected_labels, N, label_prefix='', linestyle='-', add_images=True, y_label='Average Activation Difference from Baseline'):
    for img_id in range(N - 1):
        x_values = x_lists[img_id]
        y_values = differences[img_id]
        ax.plot(x_values, y_values,
                linewidth=2.5,
                color=colors[img_id],
                linestyle=linestyle,
                label=f'{label_prefix}Task {img_id+1}')
        marker_x = [x for x in x_values if abs(x - round(x)) < 1e-6]
        marker_y = [y_values[i] for i, x in enumerate(x_values) if abs(x - round(x)) < 1e-6]
        if marker_x:
            ax.plot(marker_x, marker_y,
                    marker='o',
                    markersize=8,
                    color=colors[img_id],
                    linestyle='None')
        if add_images:
            x = x_values[0]
            y = y_values[0] + 0.01  # Adjusted to be a little bit higher
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
    ax.set_xlabel('Task ID', fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', fontsize=14, framealpha=0.95)
    ax.set_xticks(range(1, N+1))
    ax.set_xticklabels([f'{i}' for i in range(1, N+1)])
    ax.tick_params(labelsize=14)
    ax.set_xlim(0.8, N)


def plot_per_layer(axes, activation_history, task_end_snapshots, x_lists, colors, selected_images, selected_labels, N, num_layers, label_prefix='', linestyle='-', add_images=True):
    for layer_idx in range(num_layers):
        ax = axes[layer_idx]
        for img_id in range(N - 1):
            baseline_index = task_end_snapshots[img_id]
            x_values = x_lists[img_id]
            y_values = [activation_history[baseline_index + i][img_id][layer_idx]
                        for i in range(len(x_values))]
            ax.plot(x_values, y_values,
                    linewidth=2.5,
                    color=colors[img_id],
                    linestyle=linestyle,
                    label=f'{label_prefix}Sample from Task {img_id+1}')
            marker_x = [x for x in x_values if abs(x - round(x)) < 1e-6]
            marker_y = [y_values[i] for i, x in enumerate(x_values) if abs(x - round(x)) < 1e-6]
            if marker_x:
                ax.plot(marker_x, marker_y,
                        marker='o',
                        markersize=8,
                        color=colors[img_id],
                        linestyle='None')
            if add_images:
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
        ax.set_title(f'Idx of Activation Layer: {layer_idx+1}', fontsize=14, pad=10)
        ax.set_ylabel('Average Absolute Activation Value', fontsize=14)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=14)
        ax.set_xticks(range(1, N+1))
        ax.set_xticklabels([f'{i}' for i in range(1, N+1)])
        ax.tick_params(labelsize=14)
        ax.set_xlim(0.8, N)


def generate_plots(activation_history, x_points, task_end_snapshots, selected_images, selected_labels, output_dir):
    N = len(selected_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, N))

    abs_diffs, x_lists = compute_differences(
        activation_history, task_end_snapshots, x_points, signed=False
    )

    log.info('Creating activation drift visualization...')
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle('Activation Drift Over Tasks', fontsize=16)

    plot_drift(
        fig, ax, abs_diffs, x_lists, colors,
        selected_images, selected_labels, N,
        label_prefix='', linestyle='-', add_images=True,
        y_label='Average Activation Difference'
    )

    plt.tight_layout()
    output_path = output_dir / 'activation_drift_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    log.info(f'Saved visualization to {output_path}')
    plt.close()

    # New plot for signed differences
    log.info('Creating signed activation drift visualization...')
    signed_diffs, _ = compute_differences(
        activation_history, task_end_snapshots, x_points, signed=True
    )

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle('Activation Drift Over Tasks', fontsize=16)

    plot_drift(
        fig, ax, signed_diffs, x_lists, colors,
        selected_images, selected_labels, N,
        label_prefix='', linestyle='-', add_images=True,
        y_label='Average Activation Difference from Baseline'
    )

    plt.tight_layout()
    output_path_signed = output_dir / 'activation_signed_drift_visualization.png'
    plt.savefig(output_path_signed, dpi=300, bbox_inches='tight')
    log.info(f'Saved signed visualization to {output_path_signed}')
    plt.close()

    log.info('Creating detailed per-layer activation plot...')
    num_layers = (
        len(activation_history[0][0])
        if activation_history and activation_history[0] and activation_history[0][0]
        else 0
    )

    if num_layers == 0:
        log.warning('No layer activations recorded. Skipping detailed per-layer plot.')
        log.info(f'Main visualization saved to {output_dir}')
        return

    fig, axes = plt.subplots(num_layers, 1, figsize=(14, 4 * num_layers), sharex=True)
    fig.suptitle('Per-Layer Activation Over Tasks', fontsize=16)

    if num_layers == 1:
        axes = [axes]

    plot_per_layer(
        axes, activation_history, task_end_snapshots, x_lists, colors,
        selected_images, selected_labels, N, num_layers,
        label_prefix='', linestyle='-', add_images=True
    )

    axes[-1].set_xlabel('Task ID', fontsize=14)
    plt.tight_layout()

    output_path_detailed = output_dir / 'activation_per_layer_detailed.png'
    plt.savefig(output_path_detailed, dpi=300, bbox_inches='tight')
    log.info(f'Saved detailed visualization to {output_path_detailed}')
    plt.close()

    log.info(f'All visualizations saved to {output_dir}')


def load_data(visualizations_dir: Path):
    data = {}

    with open(visualizations_dir / 'activation_history.pkl', 'rb') as f:
        data['activation_history'] = pickle.load(f)

    with open(visualizations_dir / 'x_points.pkl', 'rb') as f:
        data['x_points'] = pickle.load(f)

    with open(visualizations_dir / 'task_end_snapshots.pkl', 'rb') as f:
        data['task_end_snapshots'] = pickle.load(f)

    data['selected_images'] = torch.load(visualizations_dir / 'selected_images.pt')

    with open(visualizations_dir / 'selected_labels.pkl', 'rb') as f:
        data['selected_labels'] = pickle.load(f)

    return data


def load_and_plot_visualizations(visualizations_dir: str):
    """
    Load saved data from the visualizations directory and regenerate the plots.
    """
    output_dir = Path(visualizations_dir)
    data = load_data(output_dir)

    generate_plots(
        data['activation_history'], data['x_points'],
        data['task_end_snapshots'], data['selected_images'],
        data['selected_labels'], output_dir
    )


def compare_and_plot_visualizations(
    folder1: str, folder2: str, output_folder: str,
    method_name: str = 'InTAct', baseline_name: str = 'LwF'
):
    """
    Load data from two folders and generate comparison plots.
    """
    data1 = load_data(Path(folder1))
    data2 = load_data(Path(folder2))

    N = len(data1['selected_labels'])
    colors = plt.cm.tab10(np.linspace(0, 1, N))

    # Use images and labels from first dataset
    selected_images = data1['selected_images']
    selected_labels = data1['selected_labels']

    abs_diffs1, x_lists1 = compute_differences(
        data1['activation_history'], data1['task_end_snapshots'],
        data1['x_points'], signed=False
    )

    abs_diffs2, x_lists2 = compute_differences(
        data2['activation_history'], data2['task_end_snapshots'],
        data2['x_points'], signed=False
    )

    print('Creating activation drift comparison visualization...')
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle('Activation Drift Over Tasks Comparison', fontsize=16)

    plot_drift(
        fig, ax, abs_diffs1, x_lists1, colors,
        selected_images, selected_labels, N,
        label_prefix=f'{method_name} ', linestyle='-', add_images=True,
        y_label='Average Activation Difference'
    )

    plot_drift(
        fig, ax, abs_diffs2, x_lists2, colors,
        selected_images, selected_labels, N,
        label_prefix=f'{baseline_name} ', linestyle='--', add_images=False,
        y_label='Average Activation Difference'
    )

    plt.tight_layout()
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / 'activation_drift_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Saved visualization to {output_path}')
    plt.close()

    print('Creating signed activation drift comparison visualization...')
    signed_diffs1, _ = compute_differences(
        data1['activation_history'], data1['task_end_snapshots'],
        data1['x_points'], signed=True
    )

    signed_diffs2, _ = compute_differences(
        data2['activation_history'], data2['task_end_snapshots'],
        data2['x_points'], signed=True
    )

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle('Signed Activation Drift Over Tasks Comparison', fontsize=16)

    plot_drift(
        fig, ax, signed_diffs1, x_lists1, colors,
        selected_images, selected_labels, N,
        label_prefix=f'{method_name} ', linestyle='-', add_images=True,
        y_label='Average Activation Difference from Baseline'
    )

    plot_drift(
        fig, ax, signed_diffs2, x_lists2, colors,
        selected_images, selected_labels, N,
        label_prefix=f'{baseline_name} ', linestyle='--', add_images=False,
        y_label='Average Activation Difference from Baseline'
    )

    plt.tight_layout()
    output_path_signed = output_dir / 'activation_signed_drift_visualization.png'
    plt.savefig(output_path_signed, dpi=300, bbox_inches='tight')
    print(f'Saved signed visualization to {output_path_signed}')
    plt.close()

    print('Creating detailed per-layer activation comparison plot...')
    num_layers = (
        len(data1['activation_history'][0][0])
        if data1['activation_history'] and data1['activation_history'][0]
        and data1['activation_history'][0][0]
        else 0
    )

    if num_layers == 0:
        print('No layer activations recorded. Skipping detailed per-layer plot.')
        return

    fig, axes = plt.subplots(num_layers, 1, figsize=(14, 4 * num_layers), sharex=True)
    fig.suptitle('Per-Layer Activation Over Tasks Comparison', fontsize=16)

    if num_layers == 1:
        axes = [axes]

    plot_per_layer(
        axes, data1['activation_history'], data1['task_end_snapshots'],
        x_lists1, colors, selected_images, selected_labels, N, num_layers,
        label_prefix=f'{method_name} ', linestyle='-', add_images=True
    )

    plot_per_layer(
        axes, data2['activation_history'], data2['task_end_snapshots'],
        x_lists2, colors, selected_images, selected_labels, N, num_layers,
        label_prefix=f'{baseline_name} ', linestyle='--', add_images=False
    )

    axes[-1].set_xlabel('Task ID', fontsize=14)
    plt.tight_layout()

    output_path_detailed = output_dir / 'activation_per_layer_detailed.png'
    plt.savefig(output_path_detailed, dpi=300, bbox_inches='tight')
    print(f'Saved detailed visualization to {output_path_detailed}')
    plt.close()

    print(f'All visualizations regenerated in {output_dir}')
