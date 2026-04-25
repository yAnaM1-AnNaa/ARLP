import torch
from torch.nn import nn
import timm
import types
from .steer_crossattention import GatedCrossAttention, attn_forward_wrapper


def block_forward(self, x):
    '''
    x = (image_tokens, text_feats, attn_mask)
    x --> LN1 --> attn -->  LN2 --> FFN --> x
    - injecting cross_attention layer before LN1
    note: both LayerScale and stochastic_depth are Identity()
    '''
    # turn x from tuple into image_tokens
    x, text_feats, attn_mask = x

    # gated cross attention
    if self.gated_cross_attn is not None and text_feats is not None:
        '''x -> x + fusion(x, text_feats)'''
        x = self.gated_cross_attn(x, text_feats, attn_mask = attn_mask)

    # self attention
    attn_input = self.norm1(x)
    attn_output = self.attn(attn_input) # original self-attention of ViT
    attn_output = self.ls1(attn_output) # LayerScale, control the magnitude of attention output
    attn_output = self.drop_path1(attn_output) # Stochastic Depth, randomly drop the residual branch during training
    x = x + attn_output # residual connection

    # FFN/MLP
    mlp_input = self.norm2(x)
    mlp_output = self.mlp(mlp_input) # FFN, process tokens independently
    mlp_output = self.ls2(mlp_output) # LayerScale
    mlp_output = self.drop_path2(mlp_output) # Stochastic Depth
    x = x + mlp_output # residual connection

    return (x, text_feats, attn_mask)


def forward_features(self, x, attn_mask = None) -> torch.Tensor:
    '''x = (images, text_feats)'''
    if isinstance(x, tuple):
        x, text_feats = x # unwrap the tuple
    else:
        text_feats = attn_mask = None

    x = self.patch_embed(x) # (B, N_patches, embed_dim)
    x = self._pos_embed(x)  # (B, N_patches+CLS, embed_dim)
    x = self.patch_drop(x)  # randomly drop some patches during training, for data augmentation
    x = self.norm_pre(x)    # LayerNorm before feeding into transformer blocks

    x, text_feats, attn_mask = self.blocks((x, text_feats, attn_mask))
    x = self.norm(x) # LayerNorm at the end of ViT backbone
    return x


class ViTBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.resolution = config['resolution']
        self.trunk = timm.create_model(config['model_name'],
                                       pretrained=True,
                                       img_size=self.resolution)
        self.trunk.attn_pool = None
        self.trunk.forward_features = types.MethodType(forward_features, self.trunk) # replace the original forward_features with modified version
        
        for layer_idx, block in enumerate(self.trunk.blocks):
            block.forward = types.MethodType(block_forward, block)
            if (layer_idx in config['cross_attn_layers']):
                block.gated_cross_attn = GatedCrossAttention(layer_idx,
                                                             dim=self.trunk.embed_dim,
                                                             ff_mult=config['cross_attn_ffn_mult'],
                                                             use_ffn=config["use_ffn"])
            else:
                block.gated_cross_attn = None
            block.attn.forward = types.MethodType(attn_forward_wrapper, block.attn)

            # Disable fused attention for the last block for visualizing attention maps.
            block.attn.fused_attn = False if ((layer_idx == len(self.trunk.blocks)-1)) else True

    def forward(self, image, text, attn_mask=None):
        return self.trunk.forward_features((image,text), attn_mask=attn_mask)
