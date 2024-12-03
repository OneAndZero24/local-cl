import torch.nn.functional as F

from avalanche.core import SupervisedPlugin


def activation_entropy_loss(activations, gamma=1e-4):
    loss = 0.0
    for activation in activations:
        entropy = F.softmax(activation, dim=1) * F.log_softmax(activation, dim=1)
        loss += -1.0 * entropy.sum()
    return gamma * loss
    

class ActivationEntropyPlugin(SupervisedPlugin):
    def __init__(self, gamma=1e-4):
        super().__init__()
        self.gamma = gamma

    def before_backward(self, strategy, *args, **kwargs):
        if hasattr(strategy.model, "activations"):
            strategy.loss += activation_entropy_loss(strategy.model.activations, self.gamma)
