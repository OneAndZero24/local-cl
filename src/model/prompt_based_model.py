import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple

from model.cl_module_abc import CLModuleABC
from src.model.vit_utils import VisionTransformer
        

class PromptViT(CLModuleABC):
    """
    Vision Transformer (ViT) model with optional prompt-based conditioning.

    This class wraps a `VisionTransformer` backbone and applies a custom head 
    to the CLS token embedding. It supports prompt tuning, task conditioning, 
    and pretrained weight loading.

    Args:
        head (nn.Module): 
            Head module applied on top of the CLS embedding.
        img_size (int, optional): 
            Input image size. Defaults to 224.
        patch_size (int, optional): 
            Patch size for ViT. Defaults to 16.
        in_chans (int, optional): 
            Number of input channels. Defaults to 3.
        embed_dim (int, optional): 
            Embedding dimension of transformer. Defaults to 768.
        depth (int, optional): 
            Number of transformer blocks. Defaults to 12.
        num_heads (int, optional): 
            Number of attention heads. Defaults to 12.
        mlp_ratio (float, optional): 
            MLP expansion ratio in each block. Defaults to 4.0.
        qkv_bias (bool, optional): 
            If True, add bias to QKV projections. Defaults to True.
        qk_scale (float, optional): 
            Override default QK scaling factor. Defaults to None.
        drop_rate (float, optional): 
            Dropout rate. Defaults to 0.0.
        attn_drop_rate (float, optional): 
            Attention dropout rate. Defaults to 0.0.
        drop_path_rate (float, optional): 
            Stochastic depth rate. Defaults to 0.0.
        norm_layer (nn.Module, optional): 
            Normalization layer. Defaults to None.

    Attributes:
        vit (VisionTransformer): 
            The ViT backbone.
        output_dim (int): 
            Dimensionality of the CLS token embedding.
        c_head (nn.Module): 
            The final head module applied on top of CLS embeddings.
        layers (nn.Module or None): 
            Extra layers provided by the head (if any).
        size (tuple[int, int]): 
            Input image size as (height, width).

    Methods:
        forward(x, register_blk=-1, prompt=None, q=None, train=False, task_id=None):
            Run a forward pass through the model and return the head output 
            and optional prompt loss.
    """

    def __init__(self,
                 head: nn.Module,
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_chans: int = 3,
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 qk_scale: Optional[float] = None,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0,
                 norm_layer: Optional[nn.Module] = None) -> None:

        c_head = getattr(head, "head", head)
        super().__init__(c_head)

        self.feat = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
        )

        from timm.models.vision_transformer import vit_base_patch16_224

        load_dict = vit_base_patch16_224(pretrained=True).state_dict()
        del load_dict['head.weight']
        del load_dict['head.bias']
        self.feat.load_state_dict(load_dict)


        self.output_dim = embed_dim
        self.c_head = c_head
        self.layers = getattr(head, "layers", None)
        self.size = (img_size, img_size)

    def forward(
        self,
        x: Tensor,
        register_blk: int = -1,
        prompt: Optional[Tensor] = None,
        q: Optional[Tensor] = None,
        train: bool = False,
        task_id: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Run a forward pass through the ViT and apply the custom head.

        Args:
            x (Tensor): Input image tensor of shape (B, C, H, W).
            register_blk (int, optional): 
                If >= 0, registers activations from a specific block. Defaults to -1.
            prompt (Tensor, optional): 
                Prompt embeddings for prompt-based learning. Defaults to None.
            q (Tensor, optional): 
                Query tensor for prompt conditioning. Defaults to None.
            train (bool, optional): 
                Training mode flag (may affect prompt logic). Defaults to False.
            task_id (int, optional): 
                Task identifier (for multi-task setups). Defaults to None.

        Returns:
            Tuple[Tensor, Tensor]: 
                Head output tensor and a scalar tensor representing prompt loss.
        """
        self.reset_activations()

        seq, _ = self.vit(
            x,
            register_blk=register_blk,
            prompt=prompt,
            q=q,
            train=train,
            task_id=task_id,
        )

        cls = seq[:, 0, :]  # CLS token embedding
        out = self.c_head(cls)
        return out
