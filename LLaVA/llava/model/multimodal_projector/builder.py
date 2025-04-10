import torch
import torch.nn as nn
import re
from .projectors.token_select import TokenFilter
from .projectors.wico import WiCo
from .projectors.token_crossatt import Perciever
from .projectors.token_mixer import TokenMixer
from .projectors.token_concat import TokenConcat
from .projectors.cabstractor import CAbstractor


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)

    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)
    
    elif projector_type.lower() == 'wico':
        return WiCo(in_channels=config.mm_hidden_size, out_channels=config.hidden_size, in_size=(24, 24), out_size=(12, 12))    

    elif projector_type.lower() == 'tokenfilter':
        return TokenFilter(in_channels=config.mm_hidden_size, out_channels=config.hidden_size, k=144)    

    elif projector_type.lower() == 'perciever':
        return Perciever(in_channels=config.mm_hidden_size, out_channels=config.hidden_size, num_query=144)    

    elif projector_type.lower() == 'cabstractor':
        return CAbstractor(depth=3, mlp_depth=2, input_tokens=576, output_tokens=144, 
                              hidden_size=1024, in_channels=1024, out_channels=4096)

    elif projector_type.lower() == 'tokenoncat':
        return TokenConcat(in_channels=config.mm_hidden_size, out_channels=config.hidden_size, k=4) 

    elif projector_type.lower() == 'tokenmixer':
        return TokenMixer(in_channels=config.mm_hidden_size, out_channels=config.hidden_size, in_tokens=576, out_tokens=144) 

    elif projector_type.lower() == 'prumerge':
        mlp_depth = 2
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    elif projector_type == 'tome':
        mlp_depth = 2
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)
    
    elif projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')
