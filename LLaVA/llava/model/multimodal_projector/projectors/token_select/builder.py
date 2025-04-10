import torch
import torch.nn as nn

class TokenFilter(nn.Module):
    def __init__(self, k=64, input_tokens=256, in_channels=1024, out_channels=4096, num_heads=32):
        super(TokenFilter, self).__init__()
        """
        TokenFilter (adapt from TextMonkey: An OCR-Free Large Multimodal Model for Understanding Document: https://arxiv.org/pdf/2403.04473)
        k: number of selected patches
        num_heads: number of heads
        in_channels: the number of input channels
        out_channels: the number of output channels
        """
        self.k = k
        self.patch_proj = nn.Linear(in_channels, out_channels)
        self.cross_att = MultiHeadAttention(in_channels, num_heads)
        self.pos_emb = torch.nn.Parameter(torch.zeros(1, input_tokens, in_channels))
        nn.init.trunc_normal_(self.pos_emb, mean=0.0, std=0.02)
    
    def select_patches(self, x):
        """
        x: (B, N, D)
        return: (B, k, D)
        """
        B, N, D = x.shape
        sim_mat = torch.einsum("bik,bjk->bij", (x, x)) # (B, N, N)
        mask = torch.eye(N, dtype=torch.bool).unsqueeze(0).repeat(B, 1, 1)
        sim_mat[mask] = -float('inf')
        sim_mat = sim_mat.softmax(dim=-1)
        
        max_val, _ = sim_mat.max(dim=-1) # (B, N)
        importance = 1-max_val # (B, N)
        _, topk_indices = importance.topk(self.k, dim=1) # (B, k)
        topk_indices = topk_indices.unsqueeze(-1).expand(-1, -1, D)
        
        selected_patches = torch.gather(x, 1, topk_indices) # (B, k, D)
        
        return selected_patches
        
    def forward(self, x):
        """
        x: (B, N, D)
        return: (B, k, out_channels)
        """
        flag = False
        if len(x.shape)==2:
            x = x.unsqueeze(0)
            flag = True
        
        x = x + self.pos_emb
        selected_patches = self.select_patches(x) # B, k, D
        
        select_patches_pooled = self.cross_att(selected_patches, x, x) + selected_patches # select_patches_pooled: (B, k, D)
        select_patches_pooled_proj = self.patch_proj(select_patches_pooled) # select_patches_pooled: (B, k, out_channels)
        
        if flag: return select_patches_pooled_proj.squeeze(0)
        
        return select_patches_pooled_proj


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        """
        multi-head attention
        d_model: the dimention of patch
        num_heads: the number of heads when performing multi-head attention
        """
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)
    
    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, query, key, value, mask=None):
        """
        query: (B, N, D)
        key: (B, Nk, D)
        value: (B, Nk, D)
        """
        batch_size = query.shape[0] # B
        query = self.split_heads(self.wq(query), batch_size) # (B, Nh, N, Hd)
        key = self.split_heads(self.wk(key), batch_size) # (B, Nh, N, Hd)
        value = self.split_heads(self.wv(value), batch_size) # (B, Nh, N, Hd)

        scores = torch.matmul(query, key.transpose(-2, -1)) / self.depth**0.5 # (B, Nh, N, N)
        if mask is not None:
            scores += (mask * -1e9)
        attention_weights = torch.nn.functional.softmax(scores, dim=-1) # (B, Nh, N, N)
        output = torch.matmul(attention_weights, value) # (B, Nh, N, Hd)
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model) # (B, N, D)
        return self.dense(output)

    
if __name__ == '__main__':
    patches = torch.randn([16, 256, 1024])
    model = TokenFilter(k=100, in_channels=1024, out_channels=4096, num_heads=32)
    patches = model(patches)
    print(patches.shape)

