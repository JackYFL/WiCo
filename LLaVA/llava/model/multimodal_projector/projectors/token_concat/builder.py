import torch
import torch.nn as nn


class TokenConcat(nn.Module):
    def __init__(self, k=4, in_channels=1024, out_channels=4096):
        super(TokenConcat, self).__init__()
        """
        Implementation of token concatenation in MiniGPT-v2: large language model as a unified interface for vision-language multi-task learning (https://arxiv.org/pdf/2310.09478)
        """
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

    model = TokenConcat().cuda()
    transformed_data = model(data)

    print(transformed_data.shape)  # Should be (B, 64, D)
