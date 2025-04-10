import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import LayerNorm, LayerNorm2d
from timm.models.regnet import RegStage

from functools import partial


def build_pos_embeds(
    input_tokens: int, vision_hidden_size: int
):
    pos_emb = torch.nn.Parameter(torch.zeros(1, input_tokens, vision_hidden_size))
    nn.init.trunc_normal_(pos_emb, mean=0.0, std=0.02)

    return pos_emb

def build_eos_tokens(out_channels: int):
    # think tokens
    eos_tokens = torch.nn.Parameter(torch.randn(1, 1, out_channels))
    nn.init.trunc_normal_(eos_tokens, mean=0.0, std=0.02)

    return eos_tokens

def build_prenorm(in_channels):
    prenorm = LayerNorm(in_channels)
    return prenorm

def build_mlp(depth, hidden_size, out_channels):
    layers = [nn.Linear(hidden_size, out_channels)]
    for _ in range(1, depth):
        layers.append(nn.SiLU())
        layers.append(nn.Linear(out_channels, out_channels))
    return nn.Sequential(*layers)

class Projector(nn.Module):
    """Base projector class"""

    def __init__(
        self,
        depth=3,
        mlp_depth=2,
        input_tokens=576,
        output_tokens=144,
        hidden_size=1024,
        in_channels=1024,
        out_channels=4096,
    ):
        super().__init__()
        self.depth = depth
        self.mlp_depth = mlp_depth
        self.in_channels = in_channels
        self.output_tokens = output_tokens
        self.hidden_size = hidden_size
        self.input_tokens = input_tokens
        self.out_channels = out_channels

        # think tokens
        # self.eos_tokens = build_eos_tokens(out_channels)
        # pos emb
        self.pos_emb = build_pos_embeds(input_tokens, in_channels)

        self.prenorm = build_prenorm(in_channels)

        self.build_net()

    def build_net(self):
        raise NotImplementedError()

    def _forward(self, x):
        raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, in_channels) tensor from the visual backbone (CLIP visual encoder), including cls token.
        """
        if self.prenorm is not None:
            x = self.prenorm(x)
            
        flag = False
        if len(x.shape)==2:
            x = x.unsqueeze(0)
            flag = True
            
        if self.pos_emb is not None:
            x = x + self.pos_emb

        x = self._forward(x)  # (B, L, out_channels)

        B = x.size(0)
        # if self.eos_tokens is not None:
        #     x = torch.cat([x, self.eos_tokens.expand(B, -1, -1)], dim=1)
        if flag: return x.squeeze(0)
        return x

class ConvProjector(Projector):
    def _forward(self, x):
        # x: [B, L, dim]
        # x = x[:, 1:]  # drop cls token and 2d forward
        hw = int(x.size(1) ** 0.5)
        x = rearrange(x, "b (h w) d -> b d h w", h=hw, w=hw)
        x = self.net(x)
        x = rearrange(x, "b d h w -> b (h w) d")
        x = self.readout(x)

        return x
    
class CAbstractor(ConvProjector):
    """This is the implementation of C-Abstractor from Honeybee: Locality-enhanced projector for multimodal llm (http://openaccess.thecvf.com/content/CVPR2024/papers/Cha_Honeybee_Locality-enhanced_Projector_for_Multimodal_LLM_CVPR_2024_paper.pdf)"""
    def build_net(self):
        in_channels = self.in_channels
        hidden_size = self.hidden_size
        out_channels = self.out_channels
        depth = self.depth
        mlp_depth = self.mlp_depth

        n_queries = self.output_tokens
        assert (n_queries ** 0.5).is_integer(), "n_queries must be square number"
        hw = int(n_queries ** 0.5)

        # RegBlock = ResBlock + SE
        RegBlock = partial(
            RegStage,
            stride=1,
            dilation=1,
            act_layer=nn.SiLU,
            norm_layer=LayerNorm2d,
        )

        s1 = RegBlock(
            depth,
            in_channels,
            hidden_size,
        )
        sampler = nn.AdaptiveAvgPool2d((hw, hw))
        s2 = RegBlock(
            depth,
            hidden_size,
            hidden_size,
        )

        self.net = nn.Sequential(s1, sampler, s2)
        self.readout = build_mlp(mlp_depth, hidden_size, out_channels)


if __name__ == '__main__':
    patches = torch.randn([16, 256, 1024])
    model = CAbstractor(depth=3, mlp_depth=2, input_tokens=256, output_tokens=64, 
                              hidden_size=1024, in_channels=1024, out_channels=4096)
    patches = model(patches)
    print(patches.shape)
