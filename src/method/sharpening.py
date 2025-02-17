from src.method.method_plugin_abc import MethodPluginABC

from method.regularization import sharpen_loss

class Sharpening(MethodPluginABC):
    """
    Sharpening is a method plugin that adjusts activations by additional backward increasing most active nodes.
        https://cdn.aaai.org/Symposia/Spring/1993/SS-93-06/SS93-06-007.pdf

    Attributes:
        alpha (float): The scaling factor for the original loss.
        gamma (float): The scaling factor for the sharpening loss.
        K (int): The number of top activations to consider for sharpening.

    Methods:
        setup_task(task_id: int):
            Placeholder method for setting up a task. Currently does nothing.

        forward(x, y, loss, preds):
            Adjusts the loss by combining the original loss with a sharpening loss.
    """

    def __init__(self, 
        alpha: float,
        gamma: float,
        K : int
    ):
        """
        Initializes the sharpening method with the given parameters.

        Args:
            alpha (float): The alpha parameter for the sharpening method.
            gamma (float): The gamma parameter for the sharpening method.
            K (int): The K parameter for the sharpening method.
        """
                
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.K = K


    def setup_task(self, task_id: int):
        """
        Sets up a task with the given task ID.

        Args:
            task_id (int): The unique identifier for the task to be set up.
        """

        pass  


    def forward(self, x, y, loss, preds):
        """
        Perform the forward pass of the sharpening method.

        Args:
            x (torch.Tensor): Input tensor.
            y (torch.Tensor): Target tensor.
            loss (torch.Tensor): Initial loss value.
            preds (torch.Tensor): Predictions tensor.
            
        Returns:
            tuple: Updated loss and predictions tensors.
        """

        loss *= self.alpha
        loss += (1-self.alpha)*sharpen_loss(self.module.activations, x.shape[0], self.gamma, self.K)
        return loss, preds