import torch
import torch.nn as nn


class PatchConcat(nn.Module):
    def __init__(self, k=4, in_channels=1024, out_channels=4096):
        super(PatchConcat, self).__init__()
        self.k = k
        self.proj = nn.Linear(in_channels*k, out_channels)

    def forward(self, x):
        if len(x.shape)==3:
            b, p, h = x.shape
            x = x.view(b, int(p/self.k), int(h*self.k))
        elif len(x.shape)==2:
            p, h = x.shape
            x = x.view(int(p/self.k), int(h*self.k))
            
        return self.proj(x)

if __name__ == '__main__':
    # Example use
    data = torch.randn(128, 256, 1024).cuda()  # Assuming using CUDA

    model = PatchConcat().cuda()
    transformed_data = model(data)

    print(transformed_data.shape)  # Should be (B, 64, D)
