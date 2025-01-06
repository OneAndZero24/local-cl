import torch
import torch.nn.functional as F


def distillation_loss(outputs_new, outputs_old, T=2):
    size = outputs_old.size(dim=1)
    prob_new = F.softmax(outputs_new[:,:size]/T,dim=1)
    prob_old = F.softmax(outputs_old/T,dim=1)
    return prob_old.mul(-1*torch.log(prob_new)).sum(1).mean()*T*T


def activation_loss(activations, loss_type='entropy', gamma=1e-4):
    def entropy_loss(activation_sum):
        entropy = F.softmax(activation_sum, dim=0) * F.log_softmax(activation_sum, dim=0)
        return -1*entropy.sum()

    def l1_loss(activation_sum):
        return activation_sum.mean()

    def l0_loss(activation_sum):
        return (activation_sum > 0).float().mean()

    # Loss function lookup
    loss_fn = {
        'entropy': entropy_loss,
        'l1': l1_loss,
        'l0': l0_loss
    }.get(loss_type)

    if loss_fn is None:
        raise ValueError(f"Invalid loss_type '{loss_type}'. Choose from 'entropy', 'l1', or 'l0'.")

    loss = sum(loss_fn(activation.sum(dim=0)) for activation in activations)
    return gamma * loss