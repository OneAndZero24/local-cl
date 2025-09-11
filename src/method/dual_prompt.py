"""
Based on the code: https://github.com/GT-RIPL/CODA-Prompt/blob/main/models/zoo.py
"""

import torch.nn as nn
import torch

from method.method_plugin_abc import MethodPluginABC
from method.prompt_utils import tensor_prompt

from typing import Optional, Tuple, List, Union
from torch import Tensor

class DualPrompt(MethodPluginABC):
    """
    Dual-Prompt module for continual learning.

    This class implements the Dual-Prompt mechanism, which combines
    general prompts (G-prompts) and expert prompts (E-prompts).
    G-prompts are shared across tasks, while E-prompts are selected
    based on task-specific similarity matching.

    Args:
        emb_d (int): Embedding dimension for prompts.
        prompt_param (list[int]): Prompt configuration list where:
            - prompt_param[0]: E-prompt pool size
            - prompt_param[1]: E-prompt length
            - prompt_param[2]: G-prompt length
        key_dim (int, optional): Dimensionality of key vectors for similarity
            matching. Defaults to 768.

    Attributes:
        task_id (int or None): Current task ID, set during training.
        emb_d (int): Embedding dimension.
        key_d (int): Key embedding dimension.
        top_k (int): Number of top keys to select when not using task bootstrap.
        task_id_bootstrap (bool): Whether to use task ID directly during training.
        g_layers (list[int]): Layers at which G-prompts are inserted.
        e_layers (list[int]): Layers at which E-prompts are inserted.
        g_p_length (int): Length of general prompts (divided into key/value).
        e_p_length (int): Length of expert prompts (divided into key/value).
        e_pool_size (int): Number of expert prompts per layer.
    """
    def __init__(self, emb_d: int, prompt_param: List[int], key_dim: int = 768) -> None:
        super().__init__()
        self.task_id = None
        self.emb_d = emb_d
        self.key_d = key_dim
        self._init_smart(emb_d, prompt_param)

        # G prompt init
        for g in self.g_layers:
            p = tensor_prompt(self.g_p_length, emb_d)
            setattr(self, f'g_p_{g}',p)

        # E prompt init
        for e in self.e_layers:
            p = tensor_prompt(self.e_pool_size, self.e_p_length, emb_d)
            k = tensor_prompt(self.e_pool_size, self.key_d)
            setattr(self, f'e_p_{e}',p)
            setattr(self, f'e_k_{e}',k)

    def _init_smart(self, prompt_param: List[int]) -> None:
        """
        Initializes prompt configuration based on provided parameters.

        Args:
            prompt_param (list[int]): Prompt configuration list:
                [e_pool_size, e_p_length, g_p_length].
        """
        
        self.top_k = 1
        self.task_id_bootstrap = True

        # Prompt locations
        self.g_layers = [0,1]
        self.e_layers = [2,3,4]

        # Prompt pool size
        self.g_p_length = int(prompt_param[2])
        self.e_p_length = int(prompt_param[1])
        self.e_pool_size = int(prompt_param[0])

    def setup_task(self, task_id: int) -> None:
        self.task_id = task_id

    def prompt_forward(self,
        x_querry: Tensor,
        l: int,
        x_block: Tensor,
        train: bool = False,
        task_id: Optional[int] = None
    ) -> Tuple[Optional[List[Tensor]], Union[Tensor, int], Tensor]:
        """
        Selects and applies prompts for a given layer.

        Args:
            x_querry (Tensor): Query embeddings of shape (B, C),
                where B is batch size and C is embedding dimension.
            l (int): Current layer index.
            x_block (Tensor): Input tensor block (passed through unchanged).
            train (bool, optional): If True, use training mode (with optional
                task ID bootstrap). Defaults to False.
            task_id (int, optional): Task ID for bootstrap. Used only if
                train=True and task_id_bootstrap=True.

        Returns:
            tuple:
                - p_return (list[Tensor] or None): List of selected prompts
                  [P_k, P_v] or None if no prompts apply at this layer.
                - loss (Tensor or int): Prompt matching loss (0 if not used).
                - x_block (Tensor): The unchanged input block.
        """

        # E prompts
        e_valid = False
        if l in self.e_layers:
            e_valid = True
            B, C = x_querry.shape
            K = getattr(self,f'e_k_{l}') # 0 based indexing here
            p = getattr(self,f'e_p_{l}') # 0 based indexing here
            
            # Cosine similarity to match keys/querries
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(x_querry, dim=1).detach()
            cos_sim = torch.einsum('bj,kj->bk', q, n_K)
            
            if train:
                # Dual prompt during training uses task id
                if self.task_id_bootstrap:
                    loss = (1.0 - cos_sim[:,task_id]).sum()
                    P_ = p[task_id].expand(len(x_querry),-1,-1)
                else:
                    top_k = torch.topk(cos_sim, self.top_k, dim=1)
                    k_idx = top_k.indices
                    loss = (1.0 - cos_sim[:,k_idx]).sum()
                    P_ = p[k_idx]
            else:
                top_k = torch.topk(cos_sim, self.top_k, dim=1)
                k_idx = top_k.indices
                P_ = p[k_idx]
                
            # Select prompts
            if train and self.task_id_bootstrap:
                i = int(self.e_p_length/2)
                Ek = P_[:,:i,:].reshape((B,-1,self.emb_d))
                Ev = P_[:,i:,:].reshape((B,-1,self.emb_d))
            else:
                i = int(self.e_p_length/2)
                Ek = P_[:,:,:i,:].reshape((B,-1,self.emb_d))
                Ev = P_[:,:,i:,:].reshape((B,-1,self.emb_d))
        
        # G prompts
        g_valid = False
        if l in self.g_layers:
            g_valid = True
            j = int(self.g_p_length/2)
            p = getattr(self,f'g_p_{l}') # 0 based indexing here
            P_ = p.expand(len(x_querry),-1,-1)
            Gk = P_[:,:j,:]
            Gv = P_[:,j:,:]

        # Combine prompts for prefix tuning
        if e_valid and g_valid:
            Pk = torch.cat((Ek, Gk), dim=1)
            Pv = torch.cat((Ev, Gv), dim=1)
            p_return = [Pk, Pv]
        elif e_valid:
            p_return = [Ek, Ev]
        elif g_valid:
            p_return = [Gk, Gv]
            loss = 0
        else:
            p_return = None
            loss = 0

        if train:
            return p_return, loss, x_block
        else:
            return p_return, 0, x_block
        
    def forward(self, x: Tensor, y: Tensor, loss: Tensor, preds: Tensor) -> Tensor:
        pass