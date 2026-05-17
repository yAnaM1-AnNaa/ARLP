import sys
import types
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.data import create_transform, resolve_data_config
from timm.layers import resample_abs_pos_embed


def _install_fake_omegaconf_modules():
    """Allow loading SteerViT checkpoints saved with OmegaConf metadata."""

    class FakeBase:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def __setstate__(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)
            else:
                self.state = state

    class ListConfig(list):
        def __new__(cls, *args, **kwargs):
            return list.__new__(cls)

        def __setstate__(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)
            else:
                self.state = state

    class DictConfig(dict):
        def __new__(cls, *args, **kwargs):
            return dict.__new__(cls)

        def __setstate__(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)
            else:
                self.state = state

    previous = {}
    for modname in [
        "omegaconf",
        "omegaconf.listconfig",
        "omegaconf.dictconfig",
        "omegaconf.base",
        "omegaconf.nodes",
    ]:
        previous[modname] = sys.modules.get(modname)
        sys.modules[modname] = types.ModuleType(modname)

    sys.modules["omegaconf.listconfig"].ListConfig = ListConfig
    sys.modules["omegaconf.dictconfig"].DictConfig = DictConfig
    sys.modules["omegaconf.base"].ContainerMetadata = type(
        "ContainerMetadata",
        (FakeBase,),
        {},
    )
    sys.modules["omegaconf.base"].Metadata = type("Metadata", (FakeBase,), {})
    sys.modules["omegaconf.nodes"].AnyNode = type("AnyNode", (FakeBase,), {})
    return previous


def _restore_modules(previous):
    for modname, module in previous.items():
        if module is None:
            sys.modules.pop(modname, None)
        else:
            sys.modules[modname] = module


def load_steervit_checkpoint(checkpoint_path, map_location="cpu"):
    """Load a SteerViT checkpoint even when OmegaConf is not installed."""
    try:
        return torch.load(checkpoint_path, map_location=map_location)
    except ModuleNotFoundError as exc:
        if exc.name != "omegaconf":
            raise

    previous = _install_fake_omegaconf_modules()
    try:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    finally:
        _restore_modules(previous)


def _load_hf_dinov2_weights_into_timm(trunk: nn.Module, checkpoint_path: str):
    """Load Hugging Face facebook/dinov2-* weights into the matching timm ViT."""
    if checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        hf_state = load_file(checkpoint_path)
    else:
        hf_state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(hf_state, dict) and "state_dict" in hf_state:
            hf_state = hf_state["state_dict"]

    timm_state = {}
    if "embeddings.cls_token" in hf_state:
        timm_state["cls_token"] = hf_state["embeddings.cls_token"]
    if "embeddings.patch_embeddings.projection.weight" in hf_state:
        timm_state["patch_embed.proj.weight"] = hf_state[
            "embeddings.patch_embeddings.projection.weight"
        ]
        timm_state["patch_embed.proj.bias"] = hf_state[
            "embeddings.patch_embeddings.projection.bias"
        ]

    if "embeddings.position_embeddings" in hf_state:
        pos_embed = hf_state["embeddings.position_embeddings"]
        target_shape = trunk.pos_embed.shape
        if pos_embed.shape != target_shape:
            grid_size = trunk.patch_embed.grid_size
            pos_embed = resample_abs_pos_embed(
                pos_embed,
                new_size=grid_size,
                num_prefix_tokens=trunk.num_prefix_tokens,
            )
        timm_state["pos_embed"] = pos_embed

    num_blocks = len(trunk.blocks)
    for layer_idx in range(num_blocks):
        src = f"encoder.layer.{layer_idx}"
        dst = f"blocks.{layer_idx}"

        for norm_idx in (1, 2):
            timm_state[f"{dst}.norm{norm_idx}.weight"] = hf_state[
                f"{src}.norm{norm_idx}.weight"
            ]
            timm_state[f"{dst}.norm{norm_idx}.bias"] = hf_state[
                f"{src}.norm{norm_idx}.bias"
            ]

        q_weight = hf_state[f"{src}.attention.attention.query.weight"]
        k_weight = hf_state[f"{src}.attention.attention.key.weight"]
        v_weight = hf_state[f"{src}.attention.attention.value.weight"]
        q_bias = hf_state[f"{src}.attention.attention.query.bias"]
        k_bias = hf_state[f"{src}.attention.attention.key.bias"]
        v_bias = hf_state[f"{src}.attention.attention.value.bias"]
        timm_state[f"{dst}.attn.qkv.weight"] = torch.cat(
            [q_weight, k_weight, v_weight],
            dim=0,
        )
        timm_state[f"{dst}.attn.qkv.bias"] = torch.cat(
            [q_bias, k_bias, v_bias],
            dim=0,
        )
        timm_state[f"{dst}.attn.proj.weight"] = hf_state[
            f"{src}.attention.output.dense.weight"
        ]
        timm_state[f"{dst}.attn.proj.bias"] = hf_state[
            f"{src}.attention.output.dense.bias"
        ]

        timm_state[f"{dst}.mlp.fc1.weight"] = hf_state[f"{src}.mlp.fc1.weight"]
        timm_state[f"{dst}.mlp.fc1.bias"] = hf_state[f"{src}.mlp.fc1.bias"]
        timm_state[f"{dst}.mlp.fc2.weight"] = hf_state[f"{src}.mlp.fc2.weight"]
        timm_state[f"{dst}.mlp.fc2.bias"] = hf_state[f"{src}.mlp.fc2.bias"]

        timm_state[f"{dst}.ls1.gamma"] = hf_state[f"{src}.layer_scale1.lambda1"]
        timm_state[f"{dst}.ls2.gamma"] = hf_state[f"{src}.layer_scale2.lambda1"]

    if "layernorm.weight" in hf_state:
        timm_state["norm.weight"] = hf_state["layernorm.weight"]
        timm_state["norm.bias"] = hf_state["layernorm.bias"]

    return trunk.load_state_dict(timm_state, strict=False)


class SteerCrossAttention(nn.Module):
    def __init__(self, dim: int = 768, heads: int = 12, dim_head: int = 96):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        bsz, num_tokens, _ = x.shape
        num_context = context.shape[1]

        q = self.to_q(self.norm(x))
        q = q.view(bsz, num_tokens, self.heads, self.dim_head).transpose(1, 2)

        kv = self.to_kv(context)
        kv = kv.view(bsz, num_context, 2, self.heads, self.dim_head)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(bsz, num_tokens, self.heads * self.dim_head)
        return self.to_out(out)


class GatedCrossAttention(nn.Module):
    def __init__(self, dim: int = 768, heads: int = 12, dim_head: int = 96):
        super().__init__()
        self.attn_gate = nn.Parameter(torch.zeros(1))
        self.cross_attn = SteerCrossAttention(dim=dim, heads=heads, dim_head=dim_head)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return x + self.attn_gate.tanh() * self.cross_attn(x, context)


class SteerTextConnector(nn.Module):
    def __init__(self, text_dim: int = 1024, visual_dim: int = 768):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.GELU(),
            nn.Linear(text_dim, visual_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class SteerViTBackbone(nn.Module):
    def __init__(
        self,
        model_name: str = "vit_base_patch14_dinov2.lvd142m",
        resolution: int = 336,
        cross_attn_layers: Optional[Iterable[int]] = None,
        pretrained: bool = True,
        vision_checkpoint_path: Optional[str] = None,
        freeze_trunk: bool = True,
        cross_attn_heads: int = 12,
        cross_attn_dim_head: int = 96,
    ):
        super().__init__()
        self.resolution = resolution
        self.trunk = timm.create_model(
            model_name,
            pretrained=pretrained and vision_checkpoint_path is None,
            img_size=resolution,
        )
        if vision_checkpoint_path is not None:
            result = _load_hf_dinov2_weights_into_timm(
                self.trunk,
                vision_checkpoint_path,
            )
            if result.unexpected_keys:
                print(
                    "Loaded local DINOv2 weights with unexpected keys: "
                    f"{result.unexpected_keys}"
                )

        if freeze_trunk:
            for param in self.trunk.parameters():
                param.requires_grad = False

        if cross_attn_layers is None:
            cross_attn_layers = (1, 3, 5, 7, 9, 11)
        self.cross_attn_layers = set(cross_attn_layers)

        for layer_idx, block in enumerate(self.trunk.blocks):
            if layer_idx in self.cross_attn_layers:
                block.gated_cross_attn = GatedCrossAttention(
                    dim=self.trunk.embed_dim,
                    heads=cross_attn_heads,
                    dim_head=cross_attn_dim_head,
                )
            else:
                block.gated_cross_attn = None

    @property
    def patch_size(self) -> int:
        return self.trunk.patch_embed.patch_size[0]

    @property
    def embed_dim(self) -> int:
        return self.trunk.embed_dim

    @property
    def num_prefix_tokens(self) -> int:
        return self.trunk.num_prefix_tokens

    def forward(self, images: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = self.trunk.patch_embed(images)
        x = self.trunk._pos_embed(x)
        x = self.trunk.patch_drop(x)
        x = self.trunk.norm_pre(x)

        for block in self.trunk.blocks:
            x = block(x)
            if block.gated_cross_attn is not None:
                x = block.gated_cross_attn(x, context)

        return self.trunk.norm(x)


class SteerViTDINOv2(nn.Module):
    def __init__(
        self,
        vision_model_name: str = "vit_base_patch14_dinov2.lvd142m",
        text_encoder: str = "roberta-large",
        resolution: int = 336,
        cross_attn_layers: Optional[Iterable[int]] = None,
        feature_aggregation: str = "cls",
        pretrained_vision: bool = True,
        vision_checkpoint_path: Optional[str] = None,
        max_text_tokens: int = 161,
        text_dim: int = 1024,
        cross_attn_heads: int = 12,
        cross_attn_dim_head: int = 96,
    ):
        super().__init__()
        self.feature_aggregation = feature_aggregation
        self.max_text_tokens = int(max_text_tokens)

        self.vision_model = SteerViTBackbone(
            model_name=vision_model_name,
            resolution=resolution,
            cross_attn_layers=cross_attn_layers,
            pretrained=pretrained_vision,
            vision_checkpoint_path=vision_checkpoint_path,
            freeze_trunk=True,
            cross_attn_heads=cross_attn_heads,
            cross_attn_dim_head=cross_attn_dim_head,
        )
        self.visual_dim = self.vision_model.embed_dim

        from transformers import RobertaModel, RobertaTokenizer

        self.tokenizer = RobertaTokenizer.from_pretrained(text_encoder)
        self.text_model = RobertaModel.from_pretrained(text_encoder).eval()
        for param in self.text_model.parameters():
            param.requires_grad = False

        self.connector = SteerTextConnector(text_dim=text_dim, visual_dim=self.visual_dim)
        self.textPE = nn.Embedding(self.max_text_tokens, self.visual_dim)
        self.lin_seg_head = nn.Linear(self.visual_dim, 1, bias=True)

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
        tokenized = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_tokens,
            return_tensors="pt",
        )
        text_device = next(self.text_model.parameters()).device
        tokenized = {k: v.to(text_device) for k, v in tokenized.items()}

        with torch.no_grad():
            text_tokens = self.text_model(**tokenized).last_hidden_state

        context = self.connector(text_tokens)
        pos_ids = torch.arange(context.size(1), device=context.device)
        return context + self.textPE(pos_ids).unsqueeze(0)

    def forward(self, images: torch.Tensor, texts: Optional[list[str]] = None):
        if texts is None:
            raise ValueError("SteerViTDINOv2.forward requires texts")
        if images.size(0) != len(texts):
            raise ValueError("Batch size of images and texts must match")

        context = self.encode_text(texts).to(images.device)
        return self.vision_model(images, context)

    def get_dense_features(self, images: torch.Tensor, texts: list[str]):
        feats = self.forward(images, texts)
        return feats[:, self.num_prefix_tokens:, :]

    def get_global_features(self, images: torch.Tensor, texts: list[str]):
        feats = self.forward(images, texts)
        if self.feature_aggregation == "cls":
            return feats[:, 0, :]
        return feats[:, self.num_prefix_tokens:, :].mean(dim=1)

    def get_heatmap_logits(
        self,
        images: torch.Tensor,
        texts: list[str],
        interpolate: bool = False,
    ) -> torch.Tensor:
        patch_feats = self.get_dense_features(images, texts)
        logits = self.lin_seg_head(patch_feats).squeeze(-1)

        batch = images.size(0)
        height = images.shape[-2] // self.patch_size
        width = images.shape[-1] // self.patch_size
        logits = logits.view(batch, 1, height, width)

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
