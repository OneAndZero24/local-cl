"""
Source code: https://github.com/GT-RIPL/CODA-Prompt/blob/main/models/zoo.py
"""

from typing import List

import torch
from torch import Tensor

from method.dual_prompt import DualPrompt

class L2P(DualPrompt):
    """
    L2P (Learning to Prompt) method built on top of DualPrompt.

    This class specializes DualPrompt for the L2P continual learning method.
    It customizes initialization of expert prompt layers and pool size, 
    while disabling task bootstrapping and setting top-k prompt selection.

    Args:
        emb_d (int): Embedding dimension for prompts.
        prompt_param (list[int]): Prompt configuration list where:
            - prompt_param[0]: E-prompt pool size
            - prompt_param[1]: E-prompt length
            - prompt_param[2]: Flag (>0 enables E-prompts across all layers)
        key_dim (int, optional): Dimensionality of key vectors for similarity 
            matching. Defaults to 768.

    Attributes:
        top_k (int): Number of top keys to select (set to 5).
        task_id_bootstrap (bool): Whether to bootstrap using task ID (False).
        g_layers (list[int]): Indices of layers where G-prompts are applied (empty).
        e_layers (list[int]): Indices of layers where E-prompts are applied.
        g_p_length (int): Length of G-prompts (-1, unused).
        e_p_length (int): Length of E-prompts.
        e_pool_size (int): Number of E-prompts in the pool.
    """
    def __init__(emb_d: int, prompt_param: List[int], key_dim: int = 768):
        super().__init__(emb_d, prompt_param, key_dim)

    def _init_smart(self,prompt_param: List[int]) -> None:
        """
        Initialize L2P-specific prompt configuration.

        Args:
            emb_d (int): Embedding dimension.
            prompt_param (list[int]): Prompt configuration list where:
                - prompt_param[0]: E-prompt pool size
                - prompt_param[1]: E-prompt length
                - prompt_param[2]: Flag controlling E-prompt layers
        """
        self.top_k = 5
        self.task_id_bootstrap = False

        # Prompt locations
        self.g_layers = []
        if prompt_param[2] > 0:
            self.e_layers = [0,1,2,3,4]
        else:
            self.e_layers = [0]

        # Prompt pool size
        self.g_p_length = -1
        self.e_p_length = int(prompt_param[1])
        self.e_pool_size = int(prompt_param[0])

    def forward(self, x: Tensor, y: Tensor, loss: Tensor, preds: Tensor) -> Tensor:

        with torch.no_grad():
            q, _ = self.module.feat(x)
            q = q[:,0,:]
        out, prompt_loss = self.module.feat(x, prompt=self.prompt_forward, q=q, train=self.module.training, task_id=self.task_id)
        out = out[:,0,:]
       
        out = out.view(out.size(0), -1)
        out = self.module.c_head(out)

        loss += prompt_loss

        return out, loss