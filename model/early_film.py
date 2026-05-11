import types
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.data import create_transform, resolve_data_config


def _attn_forward_wrapper(self, x: torch.Tensor) -> torch.Tensor:
    """timm ViT attention forward with optional last-layer attention capture."""
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(
        B, N, 3, self.num_heads, self.head_dim
    ).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)

    if self.fused_attn:
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
    else:
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        self.attn_map = attn
        attn = self.attn_drop(attn)
        x = attn @ v

    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


class ResidualGlobalFiLM(nn.Module):
    """Prompt-conditioned FiLM for ViT patch tokens.

    The module follows the zero-init residual FiLM convention already used in
    model.network.FiLMBlockZero: x' = (1 + gamma) * x + beta. Gamma and beta
    are global per prompt and broadcast over all patch tokens.
    """

    def __init__(
        self,
        dim: int,
        prompt_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        modulate_prefix_tokens: bool = False,
    ):
        super().__init__()
        prompt_dim = prompt_dim or dim
        hidden_dim = hidden_dim or dim
        self.modulate_prefix_tokens = modulate_prefix_tokens

        self.gamma_mlp = nn.Sequential(
            nn.LayerNorm(prompt_dim),
            nn.Linear(prompt_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.beta_mlp = nn.Sequential(
            nn.LayerNorm(prompt_dim),
            nn.Linear(prompt_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self._init_identity()

    def _init_identity(self):
        nn.init.zeros_(self.gamma_mlp[-1].weight)
        nn.init.zeros_(self.gamma_mlp[-1].bias)
        nn.init.zeros_(self.beta_mlp[-1].weight)
        nn.init.zeros_(self.beta_mlp[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        prompt_emb: torch.Tensor,
        num_prefix_tokens: int,
    ) -> torch.Tensor:
        gamma = self.gamma_mlp(prompt_emb).unsqueeze(1)
        beta = self.beta_mlp(prompt_emb).unsqueeze(1)

        if self.modulate_prefix_tokens or num_prefix_tokens == 0:
            return (1.0 + gamma) * x + beta

        prefix = x[:, :num_prefix_tokens, :]
        patch_tokens = x[:, num_prefix_tokens:, :]
        patch_tokens = (1.0 + gamma) * patch_tokens + beta
        return torch.cat((prefix, patch_tokens), dim=1)


def _block_forward(self, inputs):
    x, prompt_emb = inputs

    if self.early_film is not None and prompt_emb is not None:
        x = self.early_film(
            x,
            prompt_emb,
            num_prefix_tokens=self.num_prefix_tokens,
        )

    x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    return (x, prompt_emb)


def _forward_features(self, inputs) -> torch.Tensor:
    if isinstance(inputs, tuple):
        x, prompt_emb = inputs
    else:
        x, prompt_emb = inputs, None

    x = self.patch_embed(x)
    x = self._pos_embed(x)
    x = self.patch_drop(x)
    x = self.norm_pre(x)

    x, prompt_emb = self.blocks((x, prompt_emb))
    x = self.norm(x)
    return x


class EarlyFiLMViTBackbone(nn.Module):
    """DINO/ViT backbone with SteerViT-style early FiLM injection."""

    def __init__(
        self,
        model_name: str = "vit_large_patch14_dinov2.lvd142m",
        resolution: int = 518,
        film_layers: Optional[Iterable[int]] = None,
        pretrained: bool = True,
        film_hidden_dim: Optional[int] = None,
        modulate_prefix_tokens: bool = False,
        freeze_trunk: bool = True,
    ):
        super().__init__()
        self.resolution = resolution
        self.trunk = timm.create_model(
            model_name,
            pretrained=pretrained,
            img_size=resolution,
        )
        self.trunk.attn_pool = None
        self.trunk.forward_features = types.MethodType(_forward_features, self.trunk)

        if freeze_trunk:
            for param in self.trunk.parameters():
                param.requires_grad = False

        film_layers = set(film_layers or [])
        num_prefix_tokens = self.trunk.num_prefix_tokens
        for layer_idx, block in enumerate(self.trunk.blocks):
            block.forward = types.MethodType(_block_forward, block)
            block.num_prefix_tokens = num_prefix_tokens
            if layer_idx in film_layers:
                block.early_film = ResidualGlobalFiLM(
                    dim=self.trunk.embed_dim,
                    prompt_dim=self.trunk.embed_dim,
                    hidden_dim=film_hidden_dim,
                    modulate_prefix_tokens=modulate_prefix_tokens,
                )
            else:
                block.early_film = None

            block.attn.forward = types.MethodType(_attn_forward_wrapper, block.attn)
            block.attn.fused_attn = layer_idx != (len(self.trunk.blocks) - 1)

    @property
    def patch_size(self) -> int:
        return self.trunk.patch_embed.patch_size[0]

    @property
    def embed_dim(self) -> int:
        return self.trunk.embed_dim

    @property
    def num_prefix_tokens(self) -> int:
        return self.trunk.num_prefix_tokens

    def forward(self, images: torch.Tensor, prompt_emb: Optional[torch.Tensor] = None):
        return self.trunk.forward_features((images, prompt_emb))


class EarlyFiLMDINOv2(nn.Module):
    """RoBERTa-large conditioned Early FiLM model for frozen DINOv2 ViT-L/14."""

    def __init__(
        self,
        vision_model_name: str = "vit_large_patch14_dinov2.lvd142m",
        text_encoder: str = "roberta-large",
        resolution: int = 518,
        film_layers: Optional[Iterable[int]] = None,
        film_hidden_dim: Optional[int] = None,
        feature_aggregation: str = "mean",
        modulate_prefix_tokens: bool = False,
        pretrained_vision: bool = True,
    ):
        super().__init__()
        if film_layers is None:
            film_layers = (6, 12, 18, 23)

        self.feature_aggregation = feature_aggregation
        if self.feature_aggregation not in {"cls", "mean"}:
            raise ValueError("feature_aggregation must be 'cls' or 'mean'")

        self.vision_model = EarlyFiLMViTBackbone(
            model_name=vision_model_name,
            resolution=resolution,
            film_layers=film_layers,
            pretrained=pretrained_vision,
            film_hidden_dim=film_hidden_dim,
            modulate_prefix_tokens=modulate_prefix_tokens,
            freeze_trunk=True,
        )
        self.visual_dim = self.vision_model.embed_dim

        from transformers import RobertaModel, RobertaTokenizer

        self.tokenizer = RobertaTokenizer.from_pretrained(text_encoder)
        self.text_model = RobertaModel.from_pretrained(text_encoder).eval()
        self.text_dim = self.text_model.config.hidden_size
        for param in self.text_model.parameters():
            param.requires_grad = False

        if self.text_dim == self.visual_dim:
            self.text_connector = nn.Identity()
        else:
            self.text_connector = nn.Sequential(
                nn.LayerNorm(self.text_dim),
                nn.Linear(self.text_dim, self.visual_dim),
                nn.GELU(),
                nn.Linear(self.visual_dim, self.visual_dim),
            )

        self.lin_seg_head = nn.Linear(self.visual_dim, 1, bias=True)
        nn.init.zeros_(self.lin_seg_head.weight)
        nn.init.zeros_(self.lin_seg_head.bias)

    @property
    def patch_size(self) -> int:
        return self.vision_model.patch_size

    @property
    def image_size(self) -> tuple[int, int]:
        return (self.vision_model.resolution, self.vision_model.resolution)

    @property
    def num_prefix_tokens(self) -> int:
        return self.vision_model.num_prefix_tokens

    def get_transforms(self):
        vision_config = resolve_data_config({}, model=self.vision_model.trunk)
        vision_config["input_size"] = (3, self.image_size[0], self.image_size[1])
        return create_transform(**vision_config)

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        roberta_dict = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        text_device = next(self.text_model.parameters()).device
        roberta_dict = {k: v.to(text_device) for k, v in roberta_dict.items()}

        with torch.no_grad():
            text_tokens = self.text_model(**roberta_dict).last_hidden_state

        attn_mask = roberta_dict["attention_mask"].unsqueeze(-1).to(text_tokens.dtype)
        prompt_emb = (text_tokens * attn_mask).sum(dim=1)
        prompt_emb = prompt_emb / attn_mask.sum(dim=1).clamp_min(1.0)
        prompt_emb = F.normalize(prompt_emb, dim=-1)
        return self.text_connector(prompt_emb)

    def forward(self, images: torch.Tensor, texts: Optional[list[str]] = None):
        if texts is not None:
            if images.size(0) != len(texts):
                raise ValueError("Batch size of images and texts must match")
            prompt_emb = self.encode_text(texts).to(images.device)
        else:
            prompt_emb = None

        return self.vision_model(images, prompt_emb)

    def get_dense_features(self, images: torch.Tensor, texts: Optional[list[str]] = None):
        feats = self.forward(images, texts)
        return feats[:, self.num_prefix_tokens:, :]

    def get_global_features(self, images: torch.Tensor, texts: Optional[list[str]] = None):
        feats = self.forward(images, texts)
        if self.feature_aggregation == "cls":
            return feats[:, 0, :]
        return feats[:, self.num_prefix_tokens:, :].mean(dim=1)

    def get_heatmap_logits(
        self,
        images: torch.Tensor,
        texts: Optional[list[str]] = None,
        interpolate: bool = False,
    ) -> torch.Tensor:
        patch_feats = self.get_dense_features(images, texts)
        logits = self.lin_seg_head(patch_feats).squeeze(-1)

        B = images.size(0)
        H = images.shape[-2] // self.patch_size
        W = images.shape[-1] // self.patch_size
        logits = logits.view(B, 1, H, W)

        if interpolate:
            logits = F.interpolate(
                logits,
                size=images.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return logits

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)
