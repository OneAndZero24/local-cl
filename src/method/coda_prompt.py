"""
Based on the code: https://github.com/GT-RIPL/CODA-Prompt/blob/main/models/zoo.py
"""

import torch.nn as nn
from torch import Tensor
import torch

import copy
from typing import Optional, Tuple, List, Union

from method.prompt_utils import ortho_penalty, tensor_prompt
from method.method_plugin_abc import MethodPluginABC

class CodaPrompt(MethodPluginABC):
    """
    CODA-Prompt module for continual learning.

    This class implements the CODA-Prompt mechanism, which uses expert prompts (E-prompts)
    and key/attention mechanisms to adapt to new tasks in a continual learning setup.

    Args:
        emb_d (int): Embedding dimension for prompts.
        n_tasks (int): Number of tasks in the continual learning setup.
        prompt_param (list[int]): Prompt configuration list where:
            - prompt_param[0]: E-prompt pool size
            - prompt_param[1]: E-prompt length
            - prompt_param[2]: Orthogonal penalty coefficient (mu)
        key_dim (int, optional): Dimensionality of key vectors for similarity matching. Defaults to 768.

    Attributes:
        task_id (Optional[int]): Current task ID.
        emb_d (int): Embedding dimension.
        key_d (int): Key embedding dimension.
        n_tasks (int): Number of tasks.
        e_pool_size (int): Number of E-prompts per layer.
        e_p_length (int): Length of each E-prompt (divided into key/value).
        e_layers (List[int]): Layers where E-prompts are applied.
        ortho_mu (float): Strength of orthogonalization penalty.
    """
    def __init__(self, emb_d: int, n_tasks: int, prompt_param: List[int], key_dim: int = 768):
        super().__init__()
        self.task_id = None
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self._init_smart(emb_d, prompt_param)

        # Initialize E-prompts
        for e in self.e_layers:
            e_l = self.e_p_length
            p = tensor_prompt(self.e_pool_size, e_l, emb_d)
            k = tensor_prompt(self.e_pool_size, self.key_d)
            a = tensor_prompt(self.e_pool_size, self.key_d)
            p = self.gram_schmidt(p)
            k = self.gram_schmidt(k)
            a = self.gram_schmidt(a)
            setattr(self, f'e_p_{e}',p)
            setattr(self, f'e_k_{e}',k)
            setattr(self, f'e_a_{e}',a)

    def _init_smart(self, prompt_param: List[int]) -> None:
        """
        Initialize basic E-prompt parameters and orthogonal penalty strength.

        Args:
            prompt_param (list[int]): [e_pool_size, e_p_length, ortho_mu]
        """

        # Prompt basic param
        self.e_pool_size = int(prompt_param[0])
        self.e_p_length = int(prompt_param[1])
        self.e_layers = [0,1,2,3,4]

        # Strenth of ortho penalty
        self.ortho_mu = prompt_param[2]
        
    def setup_task(self, task_id: int) -> None:
        """
        Reinitialize the E-prompt components for a new task using Gram-Schmidt orthogonalization.

        Args:
            task_id (int): Task ID to setup.
        """
        self.task_id = task_id
        for e in self.e_layers:
            K = getattr(self,f'e_k_{e}')
            A = getattr(self,f'e_a_{e}')
            P = getattr(self,f'e_p_{e}')
            k = self.gram_schmidt(K)
            a = self.gram_schmidt(A)
            p = self.gram_schmidt(P)
            setattr(self, f'e_p_{e}',p)
            setattr(self, f'e_k_{e}',k)
            setattr(self, f'e_a_{e}',a)

    def gram_schmidt(self, vv: Tensor) -> nn.Parameter:
        """
        Apply Gram-Schmidt orthogonalization to the input tensor.

        Args:
            vv (Tensor): Input tensor of shape (N, D) or (N, L, D).

        Returns:
            nn.Parameter: Orthogonalized tensor of the same shape.
        """

        def projection(u: Tensor, v: Tensor) -> Optional[Tensor]:
            denominator = (u * u).sum()

            if denominator < 1e-8:
                return None
            else:
                return (v * u).sum() / denominator * u

        is_3d = len(vv.shape) == 3
        if is_3d:
            shape_2d = copy.deepcopy(vv.shape)
            vv = vv.view(vv.shape[0],-1)

        vv = vv.T

        uu = torch.zeros_like(vv, device=vv.device)
        
        # Get starting point
        task_count = self.task_id + 1
        pt = int(self.e_pool_size / (self.n_tasks))
        s = int(task_count * pt)
        f = int((task_count + 1) * pt)
        if s > 0:
            uu[:, 0:s] = vv[:, 0:s].clone()
        for k in range(s, f):
            redo = True
            while redo:
                redo = False
                vk = torch.randn_like(vv[:,k]).to(vv.device)
                uk = 0
                for j in range(0, k):
                    if not redo:
                        uj = uu[:, j].clone()
                        proj = projection(uj, vk)
                        if proj is None:
                            redo = True
                            print('restarting!!!')
                        else:
                            uk = uk + proj
                if not redo: uu[:, k] = vk - uk
        for k in range(s, f):
            uk = uu[:, k].clone()
            uu[:, k] = uk / (uk.norm())

        uu = uu.T 

        if is_3d:
            uu = uu.view(shape_2d)
        
        return torch.nn.Parameter(uu) 

    def prompt_forward(
        self,
        x_querry: Tensor,
        l: int,
        x_block: Tensor,
        train: bool = False
    ) -> Tuple[Optional[List[Tensor]], Union[Tensor, int], Tensor]:
        """
        Compute and select E-prompts for a given layer.

        Args:
            x_querry (Tensor): Input query embeddings (B x C).
            l (int): Layer index.
            x_block (Tensor): Input block tensor, passed through unchanged.
            train (bool, optional): Whether training mode is active. Defaults to False.

        Returns:
            Tuple containing:
                - Selected prompts (Ek, Ev) or None
                - Orthogonal penalty loss or 0
                - x_block (unchanged)
        """

        # E prompts
        e_valid = False
        if l in self.e_layers:
            e_valid = True
            B, C = x_querry.shape
            task_count = self.task_id + 1

            K = getattr(self,f'e_k_{l}')
            A = getattr(self,f'e_a_{l}')
            p = getattr(self,f'e_p_{l}')
            pt = int(self.e_pool_size / (self.n_tasks))
            s = int(task_count * pt)
            f = int((task_count + 1) * pt)
            
            # Freeze/control past tasks
            if train:
                if task_count > 0:
                    K = torch.cat((K[:s].detach().clone(),K[s:f]), dim=0)
                    A = torch.cat((A[:s].detach().clone(),A[s:f]), dim=0)
                    p = torch.cat((p[:s].detach().clone(),p[s:f]), dim=0)
                else:
                    K = K[s:f]
                    A = A[s:f]
                    p = p[s:f]
            else:
                K = K[0:f]
                A = A[0:f]
                p = p[0:f]

            # With attention and cosine sim
            # (b x 1 x d) * soft([1 x k x d]) = (b x k x d) -> attention = k x d
            a_querry = torch.einsum('bd,kd->bkd', x_querry, A)
            # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(a_querry, dim=2)
            aq_k = torch.einsum('bkd,kd->bk', q, n_K)
            # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
            P_ = torch.einsum('bk,kld->bld', aq_k, p)

            # Select prompts
            i = int(self.e_p_length/2)
            Ek = P_[:,:i,:]
            Ev = P_[:,i:,:]

            # Ortho penalty
            if train and self.ortho_mu > 0:
                loss = ortho_penalty(K) * self.ortho_mu
                loss += ortho_penalty(A) * self.ortho_mu
                loss += ortho_penalty(p.view(p.shape[0], -1)) * self.ortho_mu
            else:
                loss = 0
        else:
            loss = 0

        # Combine prompts for prefix tuning
        if e_valid:
            p_return = [Ek, Ev]
        else:
            p_return = None

        # Return
        return p_return, loss, x_block
    
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