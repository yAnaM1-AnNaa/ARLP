"""
Defines basic network building blocks and network architecture
Some code adapted from PerAct: https://github.com/peract/peract
"""

import torch
import torch.nn as nn

from typing import List

########################################
### layers
########################################
LRELU_SLOPE = 0.02

def act_layer(act):
    if act == 'relu':
        return nn.ReLU()
    elif act == 'lrelu':
        return nn.LeakyReLU(LRELU_SLOPE)
    elif act == 'elu':
        return nn.ELU()
    elif act == 'tanh':
        return nn.Tanh()
    elif act == 'prelu':
        return nn.PReLU()
    else:
        raise ValueError('%s not recognized.' % act)

def norm_layer2d(norm, channels):
    if norm == 'batch':
        return nn.BatchNorm2d(channels)
    elif norm == 'instance':
        return nn.InstanceNorm2d(channels, affine=True)
    elif norm == 'layer':
        return nn.GroupNorm(1, channels, affine=True)
    elif norm == 'group':
        return nn.GroupNorm(4, channels, affine=True)
    else:
        raise ValueError('%s not recognized.' % norm)


########################################
### network blocks
########################################

class FiLMBlockRand(nn.Module):
    """
    FiLM block with random init gamma and beta. 
    x = gamma * x + beta
    Adapted from PerAct (and original FiLM paper)
    """
    def __init__(self, lang_emb_dim, num_channels):
        super(FiLMBlockRand, self).__init__()

        self.fc_gamma = nn.Linear(lang_emb_dim, num_channels)
        self.fc_beta = nn.Linear(lang_emb_dim, num_channels)

    def forward(self, x, lang_emb):
        gamma = self.fc_gamma(lang_emb)
        beta = self.fc_beta(lang_emb)

        beta = beta.view(x.size(0), x.size(1), 1, 1)
        gamma = gamma.view(x.size(0), x.size(1), 1, 1)

        x = gamma * x + beta

        return x
    

class FiLMBlockZero(nn.Module):
    """
    FiLM block with zero init gamma and beta.
    x = (1 + gamma) * x + beta
    Adapted from RT-1 https://github.com/google-research/robotics_transformer/blob/master/film_efficientnet/film_conditioning_layer.py
    """
    def __init__(self, lang_emb_dim, num_channels):
        super(FiLMBlockZero, self).__init__()

        self.fc_gamma = nn.Linear(lang_emb_dim, num_channels)
        self.fc_beta = nn.Linear(lang_emb_dim, num_channels)

        nn.init.zeros_(self.fc_gamma.weight)
        nn.init.zeros_(self.fc_gamma.bias)
        nn.init.zeros_(self.fc_beta.weight)
        nn.init.zeros_(self.fc_beta.bias)

    def forward(self, x, lang_emb):
        gamma = self.fc_gamma(lang_emb)
        beta = self.fc_beta(lang_emb)

        beta = beta.view(x.size(0), x.size(1), 1, 1)
        gamma = gamma.view(x.size(0), x.size(1), 1, 1)

        x = (1 + gamma) * x + beta

        return x
    

class Conv2DBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_sizes, strides,
                 norm=None, activation=None, padding_mode='replicate'):
        super(Conv2DBlock, self).__init__()
        padding = kernel_sizes // 2 if isinstance(kernel_sizes, int) else (
            kernel_sizes[0] // 2, kernel_sizes[1] // 2)
        self.conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_sizes, strides, padding=padding,
            padding_mode=padding_mode)

        if activation is None:
            nn.init.xavier_uniform_(self.conv2d.weight,
                                    gain=nn.init.calculate_gain('linear'))
            nn.init.zeros_(self.conv2d.bias)
        elif activation == 'tanh':
            nn.init.xavier_uniform_(self.conv2d.weight,
                                    gain=nn.init.calculate_gain('tanh'))
            nn.init.zeros_(self.conv2d.bias)
        elif activation == 'lrelu':
            nn.init.kaiming_uniform_(self.conv2d.weight, a=LRELU_SLOPE,
                                     nonlinearity='leaky_relu')
            nn.init.zeros_(self.conv2d.bias)
        elif activation == 'relu':
            nn.init.kaiming_uniform_(self.conv2d.weight, nonlinearity='relu')
            nn.init.zeros_(self.conv2d.bias)
        else:
            raise ValueError()

        self.activation = activation
        self.norm = norm
        if norm is not None:
            self.norm = norm_layer2d(norm, out_channels)
        if activation is not None:
            self.activation = act_layer(activation)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.norm(x) if self.norm is not None else x
        x = self.activation(x) if self.activation is not None else x
        return x
    
class Conv2DFiLMBlock(Conv2DBlock):

    def __init__(self, in_channels, out_channels, kernel_sizes, strides,
                 lang_emb_dim,
                 norm=None, activation=None, padding_mode='replicate', 
                 film_mode='rand', film_place='after'
                 ):
        super(Conv2DFiLMBlock, self).__init__(
            in_channels, out_channels, kernel_sizes, strides, norm, activation,
            padding_mode)
        
    
        self.film_place = film_place    
        if film_place == 'after':
            film_channels = out_channels
        elif film_place == 'before':
            film_channels = in_channels
        else:
            raise ValueError(f"film_place {film_place} not recognized")
        
        if film_mode == 'rand':
            self.film = FiLMBlockRand(lang_emb_dim, film_channels)
        elif film_mode == 'zero':
            self.film = FiLMBlockZero(lang_emb_dim, film_channels)
        else:
            raise ValueError(f"film_mode {film_mode} not recognized")

    def forward(self, x, lang_emb):
        if self.film_place == 'before':
            x = self.film(x, lang_emb)
            x = self.conv2d(x)
            x = self.norm(x) if self.norm is not None else x
            x = self.activation(x) if self.activation is not None else x

        elif self.film_place == 'after':
            x = self.conv2d(x) # (B, C, H, W)
            x = self.norm(x) if self.norm is not None else x
            x = self.film(x, lang_emb) # lang_emb: (B, lang_emb_dim), output: (B, C, H, W)
            x = self.activation(x) if self.activation is not None else x

        else:
            raise ValueError(f"film_place {self.film_place} not recognized")
        
        return x
    

##############################################
#### Network
##############################################

class Conv2DFiLMNet(nn.Module):

    def __init__(self,
                 in_channels: int,
                 filters: List[int], # num of output channels for each Conv2D layer
                 kernel_sizes: List[int], 
                 strides: List[int], 
                 norm: str = None, 
                 activation: str = 'relu',

                 lang_emb_dim: int = 256, 
                 film_mode: str = 'zero',
                 film_place: str = 'after'
                 ):
        super(Conv2DFiLMNet, self).__init__()

        self._in_channels = in_channels
        self._filters = filters
        self._kernel_sizes = kernel_sizes
        self._strides = strides
        self._norm = norm
        self._activation = activation

        self._lang_emb_dim = lang_emb_dim
        self._film_mode = film_mode
        self._film_place = film_place

    def build(self):
        self.conv_blocks = nn.ModuleList()
        for i in range(len(self._filters)):
            in_channels = self._in_channels if i == 0 else self._filters[i-1]
            out_channels = self._filters[i]
            kernel_sizes = self._kernel_sizes[i]
            strides = self._strides[i]
            norm = self._norm 
            activation = self._activation if i < len(self._filters) - 1 else None # no activation for the last layer
            conv_block = Conv2DFiLMBlock(
                in_channels, out_channels, kernel_sizes, strides,
                self._lang_emb_dim,
                norm=norm, activation=activation,
                film_mode=self._film_mode,
                film_place=self._film_place
            )
            self.conv_blocks.append(conv_block)

    def forward(self, x, lang_emb):
        """
        Args:
            x: (B, C, H, W)
            lang_emb: (B, lang_emb_dim)
        """
        for conv_block in self.conv_blocks:
            x = conv_block(x, lang_emb)
        return x


if __name__ == "__main__":

    use_cuda = False
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    # from IPython import embed; embed(); exit(0)

    # Test Conv2DFiLMNet
    in_channels = 1024
    filters = [256, 64, 1]
    kernel_sizes = [3, 3, 1]
    strides = [1, 1, 1]
    norm = None
    activation = 'lrelu'
    lang_emb_dim = 1536
    film_mode = 'zero'

    net = Conv2DFiLMNet(
        in_channels, filters, kernel_sizes, strides, norm, activation,
        lang_emb_dim, film_mode
    )

    net.build()

    from IPython import embed; embed(); exit(0)