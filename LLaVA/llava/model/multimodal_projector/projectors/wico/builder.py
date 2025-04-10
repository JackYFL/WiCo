import torch
import torch.nn as nn
import math


class WiCo(nn.Module):
    def __init__(self,  in_channels=1024, out_channels=4096, in_size=(16, 16), out_size=(8, 8), num_heads=32):
        super(WiCo, self).__init__()
        """
        Self attention with adaptive window patch concatenation
        in_size: the spatial size of input patches
        out_size: the output size of output patches
        num_heads: number of heads for cross attention
        in_channels: the number of input channels
        out_channels: the number of output channels
        """

        # adaptive window kernel and stride
        self.out_size = out_size
        self.window_size = int((in_size[0]/out_size[0])*(in_size[1]/out_size[1]))
        self.stride = tuple(int(s / o) for s, o in zip(in_size[-2:], out_size))
        self.kernel_size = in_size[-2] - (out_size[0] - 1) * self.stride[0], in_size[-1] - (out_size[1] - 1) * self.stride[1]
        
        # Self-attention from Visual Encoder
        from transformers import CLIPVisionModel
        vit_tower = CLIPVisionModel.from_pretrained('openai/clip-vit-large-patch14-336')
        self.self_att = vit_tower.vision_model.encoder.layers[-1:]
        self.patch_proj = nn.Linear(in_channels, self.window_size*in_channels)
        self.window_proj = nn.Linear(self.window_size*in_channels, out_channels)
        
    def forward(self, x):
        """
        x: (B, N, D)
        return: (B, out_size[0]*out_size[1], out_channels)
        """
        flag = False
        if len(x.shape)==2:
            x = x.unsqueeze(0)
            flag = True
        B, N, D = x.shape
        
        H = int(math.sqrt(N))
        for blk in self.self_att:
            x = blk(x, None, None)
            x = x[0]
        x = x.reshape([B, H, H, D]) # x (B, H, H, D)
        
        window_patches = self.split_window(x).view(B, -1, self.window_size*D) # window_patches: (B, out_size[0]*out_size[1], (H/out_size[0])*(W/out_size[1])*D)        
        window_patches_proj = self.window_proj(window_patches) # patches_proj: (B, N//(window_size), out_channels)
        
        if flag: return window_patches_proj.squeeze(0)
        
        return window_patches_proj
    
    def split_window(self, x):
        """
        Split all patches into several windows
        x: (B, H, W, D)
        return sliding windows of x: (B, out_size[0], out_size[1], (H/out_size[0])*(W/out_size[1])*D)
        """
        B, H, W, D = x.shape
        unfolded = x.unfold(1, size=self.kernel_size[0], step=self.stride[0]).unfold(2, size=self.kernel_size[1], step=self.stride[1]) # x: (B, out_size[0], out_size[1], D, kernel_size[0], kernel_size[1])
        unfolded = unfolded.reshape([B, self.out_size[0], self.out_size[1], -1]) # x: (B, out_size, out_size, (H/out_size[0])*(W/out_size[1])*D))
        return unfolded
