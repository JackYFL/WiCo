import torch
import torch.nn as nn


class PatchSelect(nn.Module):
    def __init__(self, num_tokens=200, in_features=1024, out_features=4096):
        super(PatchSelect, self).__init__()
        self.num_tokens = num_tokens  # Number of principal components
        self.proj = nn.Linear(in_features, out_features)
        self.score_func = nn.Sequential(nn.Linear(in_features, in_features//2),
                                     nn.GELU(),
                                     nn.Linear(in_features//2, 1))
        
    def select(self, data): # data (Batch size, Length, Dimension)
        if len(data.shape)==2:
            data = data.unsqueeze(0)
        B, _, D = data.shape
        # mean_tokens = data.mean(dim=1) # B, D
        # token_importance = torch.einsum('bld,bd->bl', (data, mean_tokens)) # B, L
        token_importance = self.score_func(data).squeeze(2) # B, L
        _, topk_indices = token_importance.topk(k=self.num_tokens, dim=1)
        expanded_topk_indices = topk_indices.unsqueeze(-1).expand(B, self.num_tokens, D)
        select_tokens = torch.gather(data, 1, expanded_topk_indices)
        return select_tokens

    def forward(self, x):
        select_tokens = self.select(x)
        return self.proj(select_tokens)

class orgPatchSelect(nn.Module):
    def __init__(self, num_tokens=200, in_features=1024, out_features=4096):
        super(orgPatchSelect, self).__init__()
        self.num_tokens = num_tokens  # Number of principal components
        self.proj = nn.Linear(in_features, out_features)
        
    def select(self, data): # data (Batch size, Length, Dimension)
        if len(data.shape)==2:
            data = data.unsqueeze(0)
        B, _, D = data.shape
        mean_tokens = data.mean(dim=1) # B, D
        token_importance = torch.einsum('bld,bd->bl', (data, mean_tokens)) # B, L
        _, topk_indices = token_importance.topk(k=self.num_tokens, dim=1)
        expanded_topk_indices = topk_indices.unsqueeze(-1).expand(B, self.num_tokens, D)
        select_tokens = torch.gather(data, 1, expanded_topk_indices)
        return select_tokens

    def forward(self, x):
        select_tokens = self.select(x)
        return self.proj(select_tokens)

if __name__ == '__main__':
    # Example use
    # Suppose data is a tensor with 128 batches, each batch has 256 samples and 1024 features
    data = torch.randn(128, 256, 1024).cuda()  # Assuming using CUDA

    # Create a PCA transformer, setting the number of principal components to retain to 200
    model = PatchSelect(num_tokens=200).cuda()

    # Apply PCA transformation
    transformed_data = model(data)

    print(transformed_data.shape)  # Should be (B, 200, D)
