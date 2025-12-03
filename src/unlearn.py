import logging
from typing import List, Optional

import numpy as np
from omegaconf import DictConfig
from hydra.utils import instantiate

import torch
from torch.utils.data import DataLoader, Subset

import wandb
from tqdm import tqdm

from util.fabric import setup_fabric
from src.classification_loss_functions import LossCriterion
from src.method.unlearn_interval_protection import UnlearnIntervalProtection

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def unlearn_experiment(config: DictConfig):
    """
    Unlearning experiment: Train a pretrained model with negative loss on selected classes
    to make it forget them, then test per-class accuracy.
    
    Config should include:
    - model: pretrained model configuration
    - dataset: dataset configuration
    - exp.unlearn_classes: list of class indices to unlearn
    - exp.batch_size: batch size for training
    - exp.epochs: number of unlearning epochs
    - exp.lr: learning rate for unlearning
    - exp.model_path: (optional) path to load pretrained model
    - exp.criterion: loss function to use (default: 'ce')
    """
    
    if config.exp.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
    
    # Get unlearn classes
    unlearn_classes = config.exp.unlearn_classes
    
    # Check if using interval protection
    use_interval_protection = config.exp.get('use_interval_protection', False)
    lambda_interval = config.exp.get('lambda_interval', 1.0)
    
    # Initialize dataset
    log.info('Initializing datasets')
    dataset_partial = instantiate(config.dataset)
    train_dataset = dataset_partial(train=True)
    test_dataset = dataset_partial(train=False)
    
    # Initialize Fabric
    log.info('Launching Fabric')
    fabric = setup_fabric(config)
    
    # Build model
    log.info('Building model')
    model = instantiate(config.model)
    
    # Always pretrain if pretrain_epochs > 0
    should_pretrain = config.exp.get('pretrain_epochs', 0) > 0
    
    # Setup model with fabric (no task setup, no incremental handling)
    model = fabric.setup(model)
    
    # Setup optimizer and loss
    lr = config.exp.lr if 'lr' in config.exp else 0.001
    criterion_name = config.exp.criterion if 'criterion' in config.exp else 'ce'
    criterion = LossCriterion(criterion_name)
    
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr
    )
    
    # Create dataloaders
    log.info('Setting up dataloaders')
    
    # Get all unique classes from PyTorch dataset
    all_classes = sorted(set(label for _, label in test_dataset))
    retain_classes = [cls for cls in all_classes if cls not in unlearn_classes]
    log.info(f'All classes: {all_classes}')
    log.info(f'Retain classes: {retain_classes}')
    log.info(f'Unlearn classes: {unlearn_classes}')
    
    # Filter training data for unlearn classes only
    unlearn_indices = []
    for i in range(len(train_dataset)):
        _, label = train_dataset[i]
        if label in unlearn_classes:
            unlearn_indices.append(i)
    unlearn_subset = Subset(train_dataset, unlearn_indices)
    
    train_loader = fabric.setup_dataloaders(DataLoader(
        unlearn_subset,
        batch_size=config.exp.batch_size,
        shuffle=True,
        generator=torch.Generator(device=fabric.device)
    ))
    
    # For interval protection, we also need retain class data
    retain_loader = None
    if use_interval_protection:
        retain_indices = []
        for i in range(len(train_dataset)):
            _, label = train_dataset[i]
            if label in retain_classes:
                retain_indices.append(i)
        retain_subset = Subset(train_dataset, retain_indices)
        
        retain_loader = fabric.setup_dataloaders(DataLoader(
            retain_subset,
            batch_size=config.exp.batch_size,
            shuffle=False,
            generator=torch.Generator(device=fabric.device)
        ))
    
    test_loader = fabric.setup_dataloaders(DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        generator=torch.Generator(device=fabric.device)
    ))
    
    # Pretrain if needed
    if should_pretrain:
        pretrain_epochs = config.exp.pretrain_epochs
        pretrain_lr = config.exp.get('pretrain_lr', 0.001)
        log.info(f'Pretraining model for {pretrain_epochs} epochs')
        
        # Create full training dataloader
        full_train_loader = fabric.setup_dataloaders(DataLoader(
            train_dataset,
            batch_size=config.exp.batch_size,
            shuffle=True,
            generator=torch.Generator(device=fabric.device)
        ))
        
        # Pretrain optimizer
        pretrain_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=pretrain_lr
        )
        
        for epoch in range(pretrain_epochs):
            log.info(f'Pretrain epoch {epoch + 1}/{pretrain_epochs}')
            pretrain_epoch(model, pretrain_optimizer, criterion, full_train_loader, epoch, fabric)
    
    # Initialize interval protection if enabled (AFTER pretraining)
    interval_protection = None
    if use_interval_protection:
        log.info('Initializing interval protection')
        interval_protection = UnlearnIntervalProtection(
            lambda_interval=lambda_interval,
            compute_intervals_from_data=True
        )
        
        # Setup protection after model is pretrained
        interval_protection.setup_protection(
            model, retain_loader, train_loader, fabric.device
        )
    
    # Test before unlearning
    log.info('Testing before unlearning')
    per_class_acc_before = test_per_class(
        model, test_loader, all_classes, 
        log_prefix='before_unlearn'
    )
    
    # Unlearning phase
    epochs = config.exp.epochs if 'epochs' in config.exp else 5
    log.info(f'Starting unlearning for {epochs} epochs')
    
    for epoch in range(epochs):
        log.info(f'Unlearning epoch {epoch + 1}/{epochs}')
        unlearn_train(
            model, optimizer, criterion, train_loader, 
            epoch, fabric, interval_protection
        )
        
        # Test after each epoch
        log.info(f'Testing after unlearning epoch {epoch + 1}')
        per_class_acc_epoch = test_per_class(
            model, test_loader, all_classes,
            log_prefix=f'unlearn_epoch_{epoch}'
        )
    
    # Final test after all unlearning
    log.info('Testing after unlearning (final)')
    per_class_acc_after = test_per_class(
        model, test_loader, all_classes,
        log_prefix='after_unlearn'
    )
    
    # Log comparison metrics
    log.info('=== Unlearning Results ===')
    for cls in all_classes:
        acc_before = per_class_acc_before[cls]
        acc_after = per_class_acc_after[cls]
        is_unlearn = cls in unlearn_classes
        status = '[UNLEARN]' if is_unlearn else '[RETAIN]'
        log.info(f'Class {cls} {status}: {acc_before:.2f}% -> {acc_after:.2f}% (Δ {acc_after - acc_before:+.2f}%)')
        
        wandb.log({
            f'comparison/class_{cls}_before': acc_before,
            f'comparison/class_{cls}_after': acc_after,
            f'comparison/class_{cls}_delta': acc_after - acc_before,
        })
    
    # Calculate aggregate metrics
    unlearn_acc_before = np.mean([per_class_acc_before[cls] for cls in unlearn_classes])
    unlearn_acc_after = np.mean([per_class_acc_after[cls] for cls in unlearn_classes])
    
    retain_acc_before = np.mean([per_class_acc_before[cls] for cls in retain_classes]) if retain_classes else 0
    retain_acc_after = np.mean([per_class_acc_after[cls] for cls in retain_classes]) if retain_classes else 0
    
    log.info(f'\nUnlearn classes avg: {unlearn_acc_before:.2f}% -> {unlearn_acc_after:.2f}%')
    log.info(f'Retain classes avg: {retain_acc_before:.2f}% -> {retain_acc_after:.2f}%')
    
    wandb.log({
        'summary/unlearn_classes_before': unlearn_acc_before,
        'summary/unlearn_classes_after': unlearn_acc_after,
        'summary/unlearn_classes_delta': unlearn_acc_after - unlearn_acc_before,
        'summary/retain_classes_before': retain_acc_before,
        'summary/retain_classes_after': retain_acc_after,
        'summary/retain_classes_delta': retain_acc_after - retain_acc_before,
    })


def pretrain_epoch(
    model,
    optimizer,
    criterion,
    dataloader: DataLoader,
    epoch: int,
    fabric
):
    """
    Train one epoch with standard supervised learning.
    """
    model.train()
    avg_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (X, y) in enumerate(tqdm(dataloader, desc='Pretraining')):
        optimizer.zero_grad()
        
        # Forward pass
        preds = model(X)
        loss = criterion(preds, y).mean()
        
        # Backward pass
        fabric.backward(loss)
        optimizer.step()
        
        avg_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(preds.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
        
        if batch_idx % 10 == 0:
            wandb.log({
                'pretrain/batch_loss': loss.item(),
                'pretrain/epoch': epoch,
            })
    
    avg_loss /= len(dataloader)
    accuracy = 100.0 * correct / total
    
    wandb.log({
        'pretrain/epoch_loss': avg_loss,
        'pretrain/epoch_accuracy': accuracy,
        'pretrain/epoch': epoch,
    })
    log.info(f'Pretrain epoch {epoch}: loss={avg_loss:.4f}, acc={accuracy:.2f}%')


def unlearn_train(
    model, 
    optimizer, 
    criterion, 
    dataloader: DataLoader, 
    epoch: int,
    fabric,
    interval_protection: Optional[UnlearnIntervalProtection] = None
):
    """
    Train one epoch with NEGATIVE loss to forget the specified classes.
    Optionally applies interval protection to preserve retain class knowledge.
    """
    model.train()
    avg_loss = 0.0
    avg_unlearn_loss = 0.0
    avg_protection_loss = 0.0
    
    for batch_idx, (X, y) in enumerate(tqdm(dataloader, desc='Unlearning')):
        optimizer.zero_grad()
        
        # Forward pass
        preds = model(X)
        unlearn_loss = criterion(preds, y)
        
        # NEGATIVE loss to unlearn
        unlearn_loss = -unlearn_loss.mean()
        
        # Add interval protection loss if enabled
        protection_loss = torch.tensor(0.0, device=X.device)
        if interval_protection is not None:
            protection_loss = interval_protection.compute_protection_loss(model, X.device)
        
        # Total loss
        loss = unlearn_loss + protection_loss
        
        # Backward pass
        fabric.backward(loss)
        optimizer.step()
        
        avg_loss += loss.item()
        avg_unlearn_loss += unlearn_loss.item()
        avg_protection_loss += protection_loss.item()
        
        if batch_idx % 10 == 0:
            wandb.log({
                'unlearn_train/batch_loss': loss.item(),
                'unlearn_train/batch_unlearn_loss': unlearn_loss.item(),
                'unlearn_train/batch_protection_loss': protection_loss.item(),
                'unlearn_train/epoch': epoch,
            })
    
    avg_loss /= len(dataloader)
    avg_unlearn_loss /= len(dataloader)
    avg_protection_loss /= len(dataloader)
    
    wandb.log({
        'unlearn_train/epoch_loss': avg_loss,
        'unlearn_train/epoch_unlearn_loss': avg_unlearn_loss,
        'unlearn_train/epoch_protection_loss': avg_protection_loss,
        'unlearn_train/epoch': epoch,
    })
    log.info(f'Unlearning epoch {epoch}: total={avg_loss:.4f}, unlearn={avg_unlearn_loss:.4f}, protection={avg_protection_loss:.4f}')


def test_per_class(
    model, 
    dataloader: DataLoader, 
    all_classes: List[int],
    log_prefix: str = 'test'
) -> dict:
    """
    Test the model and return per-class accuracy.
    
    Returns:
        dict: Mapping from class index to accuracy percentage
    """
    model.eval()
    
    # Initialize counters for each class
    class_correct = {cls: 0 for cls in all_classes}
    class_total = {cls: 0 for cls in all_classes}
    
    with torch.no_grad():
        for X, y in tqdm(dataloader, desc='Testing'):
            preds = model(X)
            _, predicted = torch.max(preds.data, 1)
            
            for i in range(y.size(0)):
                label = y[i].item()
                pred = predicted[i].item()
                
                if label in class_correct:
                    class_total[label] += 1
                    if pred == label:
                        class_correct[label] += 1
    
    # Calculate per-class accuracy
    per_class_acc = {}
    for cls in all_classes:
        if class_total[cls] > 0:
            acc = 100.0 * class_correct[cls] / class_total[cls]
        else:
            acc = 0.0
        per_class_acc[cls] = acc
        
        log.info(f'{log_prefix}/class_{cls}: {acc:.2f}% ({class_correct[cls]}/{class_total[cls]})')
        wandb.log({
            f'{log_prefix}/class_{cls}_acc': acc,
            f'{log_prefix}/class_{cls}_correct': class_correct[cls],
            f'{log_prefix}/class_{cls}_total': class_total[cls],
        })
    
    # Overall accuracy
    total_correct = sum(class_correct.values())
    total_samples = sum(class_total.values())
    overall_acc = 100.0 * total_correct / total_samples if total_samples > 0 else 0.0
    
    log.info(f'{log_prefix}/overall: {overall_acc:.2f}% ({total_correct}/{total_samples})')
    wandb.log({
        f'{log_prefix}/overall_acc': overall_acc,
    })
    
    return per_class_acc
