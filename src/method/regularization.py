import torch
import torch.nn.functional as F


def distillation_loss(outputs_new, outputs_old, T=2):
    size = outputs_old.size(dim=1)
    prob_new = F.softmax(outputs_new[:,:size]/T,dim=1)
    prob_old = F.softmax(outputs_old/T,dim=1)
    return prob_old.mul(-1*torch.log(prob_new)).sum(1).mean()*T*T


def param_change_loss(model, multiplier, params_buffer):
    loss = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            loss += (multiplier[name] * (p - params_buffer[name]) ** 2).sum()
    return loss


def entropy_loss(activation_sum):
    entropy = F.softmax(activation_sum, dim=0) * F.log_softmax(activation_sum, dim=0)
    return -1*entropy.sum()


def l1_loss(activation_sum):
    return activation_sum.mean()


def l0_loss(activation_sum):
    return (activation_sum > 0).float().mean()


def regularization(activations, loss_type='entropy'):
    loss_fn = {
        'entropy': entropy_loss,
        'l1': l1_loss,
        'l0': l0_loss
    }.get(loss_type)

    if loss_fn is None:
        raise ValueError(f"Invalid loss_type '{loss_type}'. Choose from 'entropy', 'l1', or 'l0'.")

    return sum(loss_fn(activation.sum(dim=0)) for activation in activations)