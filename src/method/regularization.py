import torch
import torch.nn.functional as F


def distillation_loss(outputs_new, outputs_old, T=2):
    """
    Computes the distillation loss between the new model outputs and the old model outputs.
    Distillation loss is used to transfer knowledge from a teacher model (old model) to a student model (new model).
    It measures the difference between the softened output probabilities of the two models.

    Args:
        outputs_new (torch.Tensor): The output logits from the new (student) model.
        outputs_old (torch.Tensor): The output logits from the old (teacher) model.

        T (float, optional): The temperature parameter to soften the probabilities. Default is 2.
    Returns:
        torch.Tensor: The computed distillation loss.
    """

    size = outputs_old.size(dim=1)
    prob_new = F.softmax(outputs_new[:,:size]/T,dim=1)
    prob_old = F.softmax(outputs_old/T,dim=1)
    return prob_old.mul(-1*torch.log(prob_new)).sum(1).mean()*T*T


def param_change_loss(model, multiplier, params_buffer, head_opt):
    """
    Computes the parameter change loss for a given model.
    This function calculates the loss based on the difference between the current 
    parameters of the model and a buffer of previous parameters, weighted by a 
    multiplier. The loss is computed only for parameters that require gradients.

    Args:
        model (torch.nn.Module): The neural network model containing the parameters.
        multiplier (dict): A dictionary where keys are parameter names and values 
                           are the corresponding multipliers for the loss calculation.
        params_buffer (dict): A dictionary where keys are parameter names and values 
                              are the previous parameter values to compare against.z
        head_opt (bool): A flag to indicate whether EWC should be applied to the incremental head.
                              
    Returns:
        torch.Tensor: The computed parameter change loss.
    """

    loss = 0
    for name, p in model.named_parameters():
        if not head_opt and "head" in name:
            continue
        if p.requires_grad:
            loss += (multiplier[name] * (p - params_buffer[name]) ** 2).sum()
    return loss


def sharpen_loss(indices, activations, gamma):
    """
    Compute the sharpen loss for given activations.
    This function modifies the activations based on the provided indices and a sharpening factor gamma.
    It increases the activations at the specified indices and decreases the activations at other indices,
    then calculates the difference between the new and original activations.

    Args:
        indices (Tensor): A 1D tensor containing the indices of the activations to be sharpened.
        activations (Tensor): A tensor containing the original activations.
        gamma (float): The sharpening factor to adjust the activations.

    Returns:
        Tensor: The sum of squared differences between the new and original activations.
    """

    new_activations = activations.clone()
    mask = torch.zeros_like(new_activations.view(-1), dtype=torch.bool)
    mask[indices] = True
    
    new_activations.view(-1)[mask] = gamma * (1 - activations.view(-1)[mask])
    new_activations.view(-1)[~mask] -= gamma * activations.view(-1)[~mask]

    diff = torch.sum(torch.square(new_activations - activations))
    return diff
    

def entropy_loss(activation_sum):
    """
    Computes the entropy loss for a given activation sum.
    The entropy loss is calculated as the negative sum of the element-wise 
    product of the softmax of the activation sum and the log softmax of the 
    activation sum.

    Args:
        activation_sum (torch.Tensor): The input tensor containing the sum of activations.

    Returns:
        torch.Tensor: The computed entropy loss.
    """

    entropy = F.softmax(activation_sum, dim=0) * F.log_softmax(activation_sum, dim=0)
    return -1*entropy.sum()


def l1_loss(activation_sum):
    return activation_sum.mean()


def l0_loss(activation_sum):
    return (activation_sum > 0).float().mean()


def regularization(activations, loss_type='entropy'):
    """
    Apply a regularization function to a list of activations.

    Args:
        activations (list of torch.Tensor): A list of activation tensors.
        loss_type (str): The type of loss function to use for regularization. 
                     Options are 'entropy', 'l1', or 'l0'. Default is 'entropy'.

    Returns:
        float: The computed regularization loss.

    Raises:
        ValueError: If an invalid loss_type is provided.
    """

    loss_fn = {
        'entropy': entropy_loss,
        'l1': l1_loss,
        'l0': l0_loss
    }.get(loss_type)

    if loss_fn is None:
        raise ValueError(f"Invalid loss_type '{loss_type}'. Choose from 'entropy', 'l1', or 'l0'.")

    return sum(loss_fn(activation.sum(dim=0)) for activation in activations)