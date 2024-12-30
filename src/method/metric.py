import torch
from torch import nn
import torch.nn.functional as F


def distillation_loss(outputs_new, outputs_old, T=2):
    size = outputs_old.size(dim=1)
    prob_new = F.softmax(outputs_new[:,:size]/T,dim=1)
    prob_old = F.softmax(outputs_old/T,dim=1)
    return prob_old.mul(-1*torch.log(prob_new)).sum(1).mean()*T*T


def activation_entropy_loss(activations, gamma=1e-4):
    loss = 0.0
    for activation in activations:
        activation_sum = activation.sum(dim=0)
        entropy = F.softmax(activation_sum) * F.log_softmax(activation_sum)
        loss += -1.0 * entropy.sum()
    return gamma * loss