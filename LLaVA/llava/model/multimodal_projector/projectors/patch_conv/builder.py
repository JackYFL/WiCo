import torch
import torch.nn as nn
import math


class PatchConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(PatchConv, self).__init__()
        self.out_channel = 4096
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        if len(x.shape)==3:
            # x: (B, L, D)
            B, L, D = x.shape
        elif len(x.shape)==2:
            # x: (L, D)
            L, D = x.shape
            B = 1
            x = x.unsqueeze(0)
            
        H = int(math.sqrt(L))
        feature_map = x.view(B, H, H, D) # (B, H, H, D)
        feature_map = feature_map.permute(0, 3, 1, 2) # (B, D, H, H)
        feature_map = self.conv1(feature_map) # (B, D, H, H)
        feature_map = self.conv2(feature_map) # (B, D, H/2, H/2)
        feature_map = self.conv3(feature_map) # (B, output_feature, H/2, H/2)
        output = feature_map.reshape(B, self.out_channel, -1) # (B, output_feature, H*H/4)
        output = output.permute(0, 2, 1) # (B, H*H/4, output_feature)
        
        if B!=1:
            return output
        else:
            return output.squeeze(0)

if __name__ == '__main__':
    model = PatchConv(in_channel=1024, out_channel=4096)
    patches = torch.randn([16, 256, 1024])
    # patches = torch.randn([256, 1024])
    out = model(patches)
    print(out.shape)