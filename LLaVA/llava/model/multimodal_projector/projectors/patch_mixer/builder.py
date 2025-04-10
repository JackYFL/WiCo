import torch
import torch.nn as nn


class PatchMixer(nn.Module):
    def __init__(self, in_tokens=256, out_tokens=200, in_channels=1024, out_channels=4096):
        super(PatchMixer, self).__init__()
        self.proj = nn.Linear(in_channels, out_channels)
        self.token_mixer = nn.Linear(in_tokens, out_tokens)
        
    def mix(self, data): # data (Batch size, Length, Dimension)
        if len(data.shape)==2:
            data = data.unsqueeze(0)
        B, L, D = data.shape
        mixed_tokens = self.token_mixer(data.permute(0, 2, 1))
        mixed_tokens = mixed_tokens.permute(0, 2, 1)
        return mixed_tokens

    def forward(self, x):
        mixed_tokens = self.mix(x)
        return self.proj(mixed_tokens)

if __name__ == '__main__':
    # Example use
    # Suppose data is a tensor with 128 batches, each batch has 256 samples and 1024 features
    data = torch.randn(128, 256, 1024).cuda()  # Assuming using CUDA

    # Create a PCA transformer, setting the number of principal components to retain to 200
    model = PatchMixer(out_tokens=200).cuda()

    # Apply PCA transformation
    transformed_data = model(data)

    print(transformed_data.shape)  # Should be (B, 200, D)
