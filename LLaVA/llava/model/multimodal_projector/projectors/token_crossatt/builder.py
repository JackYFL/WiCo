import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
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
        batch_size = query.shape[0]
        query = self.split_heads(self.wq(query), batch_size)
        key = self.split_heads(self.wk(key), batch_size)
        value = self.split_heads(self.wv(value), batch_size)

        scores = torch.matmul(query, key.transpose(-2, -1)) / self.depth**0.5
        if mask is not None:
            scores += (mask * -1e9)
        attention_weights = torch.nn.functional.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, value)
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)
        return self.dense(output)


class Perciever(nn.Module):
    def __init__(self, 
                 num_query=200,
                 out_channels=4096,
                 in_channels=1024,
                 ):
        """
        This module implements the Perciever from Flamingo: a Visual Language Model for Few-Shot Learning (https://proceedings.neurips.cc/paper_files/paper/2022/file/960a172bc7fbf0177ccccbb411a7d800-Paper-Conference.pdf).
        It can pool image patch tokens using learnable queries.

        num_query: the number of input queries
        out_channels: the dimension of output features
        in_channels: the dimension of input feature,
            the input features for creating pyramid features
        """
        super(Perciever, self).__init__()
        self.query = nn.Embedding(num_query, in_channels)
        self.cross_attention = MultiHeadAttention(d_model=in_channels, num_heads=32)
        self.projector = nn.Sequential(
            nn.Linear(in_channels, out_channels//4),
            nn.GELU(),
            nn.Linear(out_channels//4, out_channels)
        )
        
    def forward(self, x):
        """
        x: the patch tokens from ViT (N,L,C)
        """
        if len(x.shape)==3:
            B = x.shape[0]
            query_repeat = self.query.weight.unsqueeze(0).repeat(B, 1, 1)
            query = self.cross_attention(query_repeat, x, x)
        elif len(x.shape)==2:
            B = 1
            query_repeat = self.query.weight.unsqueeze(0).repeat(B, 1, 1)
            query = self.cross_attention(query_repeat, x.unsqueeze(dim=0), x.unsqueeze(dim=0))
            query = query.squeeze(dim=0)
        return self.projector(query)


if __name__ == '__main__':
    projector = Perciever(num_query=64, in_channels=1024, out_channels=4096)
    patches = torch.randn([256, 1024])
    out = projector(patches)
    print(out.shape)