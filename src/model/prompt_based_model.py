import torch
import torch.nn as nn
from timm import create_model

from model.cl_module_abc import CLModuleABC


class PromptBasedModels(CLModuleABC):
    """
    Continual learning wrapper around timm.create_model with support for prompt-based models.

    Args:
        head (nn.Module): Custom head module for classification.
        model_name (str): Timm model name (supports prompt args).
        pretrained (bool): Whether to load pretrained weights. Default: True.
        frozen (bool): Whether to freeze the backbone. Default: False.
        size (tuple[int]): Input image size. Default: (224, 224).
        drop_rate (float): Dropout rate. Default: 0.0.
        drop_path_rate (float): Drop path rate. Default: 0.0.
        drop_block_rate (float|None): Drop block rate. Default: None.
        prompt_length (int): Prompt length.
        embedding_key (str): Which embedding to prompt.
        prompt_init (str): How to initialize prompt keys.
        prompt_pool (bool): Whether to use a prompt pool.
        prompt_key (bool): Whether to use prompt keys.
        pool_size (int): Size of prompt pool.
        top_k (int): Top-k selection of prompts.
        batchwise_prompt (bool): Use batchwise prompts.
        prompt_key_init (str): Initialization of prompt keys.
        head_type (str): Head type to use in backbone.
        use_prompt_mask (bool): Whether to mask prompts.
    """

    def __init__(
        self,
        head: nn.Module,
        model_name: str,
        pretrained: bool = True,
        frozen: bool = False,
        size: tuple[int] = (224, 224),
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        drop_block_rate: float | None = None,
        prompt_length: int = 0,
        embedding_key: str = "cls",
        prompt_init: str = "normal",
        prompt_pool: bool = False,
        prompt_key: bool = False,
        pool_size: int = 0,
        top_k: int = 1,
        batchwise_prompt: bool = False,
        prompt_key_init: str = "uniform",
        head_type: str = "linear",
        use_prompt_mask: bool = False,
    ):
        super().__init__(head.head)

        self.flatten_output = False
        self.frozen = frozen
        self.size = size

        # Model to extract keys
        self.original_model = create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            drop_block_rate=None,
        )

        self.original_model.eval()

        # Model to process prompts
        self.model = create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            drop_block_rate=drop_block_rate,
            prompt_length=prompt_length,
            embedding_key=embedding_key,
            prompt_init=prompt_init,
            prompt_pool=prompt_pool,
            prompt_key=prompt_key,
            pool_size=pool_size,
            top_k=top_k,
            batchwise_prompt=batchwise_prompt,
            prompt_key_init=prompt_key_init,
            head_type=head_type,
            use_prompt_mask=use_prompt_mask,
        )

        # Classification head
        self.c_head = head
        self.layers = head.layers

        if frozen:
            # All parameters are frozen for original vit model
            for p in self.original_model.parameters():
                p.requires_grad = False
        
        # Freeze blocks, patch_embed, cls_token parameters
        for n, p in self.model.named_parameters():
            if n.startswith(tuple(frozen)):
                p.requires_grad = False

    def forward(self, x: torch.Tensor, task_id: int, set_training_mode: bool = True) -> torch.Tensor:
        """
        Performs a forward pass through the prompt-based model.
        Args:
            x (torch.Tensor): Input tensor to the model.
            task_id (int): Identifier for the current task, used for task-specific processing.
            set_training_mode (bool, optional): If True, sets the model to training mode during the forward pass. Defaults to True.
        Returns:
            torch.Tensor: Output tensor after passing through the classification head.
        """
        
        self.reset_activations()

        with torch.no_grad():
            if self.original_model is not None:
                output = self.original_model(x)
                cls_features = output['pre_logits']
            else:
                cls_features = None

        output = self.model(x, task_id=task_id, cls_features=cls_features, train=set_training_mode)
        logits = output['logits']

        return self.c_head(logits)
