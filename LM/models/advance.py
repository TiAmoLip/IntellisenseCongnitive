import torch
from torch import nn
from torch.nn import functional as F


class RoPE(nn.Module):
    def __init__(self, d_model,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.cos_cached = None
        self.sin_cached = None
        self.base=10000
    
    def cache(self, x:torch.Tensor, seq_len:int):
        if self.cos_cached is not None and self.sin_cached is not None:
            return
        theta = 1./(self.base**torch.arange(0, self.d_model, 2).float()/self.d_model).to(x.device)
        seq_idx = torch.arange(0, seq_len).float().to(x.device)
        idx_theta = torch.einsum('n, d->nd', seq_idx,theta)
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)
        self.cos_cached = idx_theta2.cos()[:, None, None, :]
        self.sin_cached = idx_theta2.sin()[:, None, None, :]
    
    def neg_half(self, x:torch.Tensor):
        return torch.cat([-x[...,self.d_model//2:], x[...,:self.d_model//2]], dim=-1)
    
    def forward(self, x:torch.Tensor):
        
        """
        input x: [bsz, seq_len, n_heads, d] or [bsz, seq_len, d]
        """
        
        if len(x.shape) == 3:
            seq_len, bsz,d = x.shape
        if len(x.shape) == 4:
            seq_len, bsz, n_heads, d = x.shape
        
        self.cache(x, seq_len)
        neg_x = self.neg_half(x)
        # print(x.shape, self.cos_cached.shape, neg_x.shape, self.sin_cached.shape)
        return x*self.cos_cached + neg_x*self.sin_cached
        

# class FFN(nn.Module):
#     def __init__(self,embedding_size,hidden_size):
#         super(FFN,self).__init__()
#         self.l1 = nn.Linear(embedding_size,hidden_size)
#         self.l2 = nn.Linear(hidden_size,embedding_size)
#         self.drop = nn.Dropout(0.2)
#     def forward(self,x):
#         x = F.leaky_relu(self.l1(x),0.2)
#         return self.l2(self.drop(x))
    
class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout_rate) -> None:
        super().__init__()
        self.dropout_rate = dropout_rate
        self.heads = heads
        self.d_k = d_model // heads
        self.query = nn.Linear(d_model, self.d_k*heads)
        self.key = nn.Linear(d_model, self.d_k*heads)
        self.value = nn.Linear(d_model, self.d_k*heads)
    
    def get_scores(self, q:torch.Tensor, k:torch.Tensor):
        scores = torch.einsum('ibhd, jbhd->ijbh', q, k) / (self.d_k**0.5)
        return scores
    
    def forward(self, x:torch.Tensor):
        """
        x: [seq_len, bsz, d_model]
        """
        seq_len, bsz, d_model = x.shape
        # x = x.view(seq_len, bsz, self.heads, self.d_k)
        
        q = self.query(x).view(seq_len, bsz, self.heads, self.d_k)
        k = self.key(x).view(seq_len, bsz, self.heads, self.d_k)
        v = self.value(x).view(seq_len, bsz, self.heads, self.d_k)
        
        scores = self.get_scores(q, k)
        T = torch.tril(torch.ones(seq_len, seq_len)).to(x.device).unsqueeze(-1).unsqueeze(-1)
        # print(scores.shape)
        scores = scores.masked_fill(T==0, float('-inf'))
        
        attn = softmax_one(scores, dim=1)
        attn = F.dropout(attn, p=self.dropout_rate)
        out = torch.einsum('ijbh, jbhd->ibhd', attn, v)
        out = out.reshape(seq_len, bsz, d_model)
        return out

class RoPEAttention(MultiHeadAttention):
    def __init__(self, heads, d_model, dropout_rate) -> None:
        super().__init__(heads, d_model, dropout_rate)
        self.query_rope = RoPE(self.d_k)
        self.key_rope = RoPE(self.d_k)
    
    def get_scores(self, q: torch.Tensor, k: torch.Tensor):
        return torch.einsum('ibhd, jbhd->ijbh', self.query_rope(q), self.key_rope(k)) / (self.d_k**0.5)

# class Block(nn.Module):
#     def __init__(self, d_model,hidden,n_heads) -> None:
#         super().__init__()
#         self.sa = MultiHeadAttention(n_heads, d_model, 0.2)
#         self.ffwd = FFN(d_model, hidden)
#         self.ln1 = nn.LayerNorm(d_model)
#         self.ln2 = nn.LayerNorm(d_model)

#     def forward(self, x):
#         # print(x.shape)
#         # print(self.sa(x).shape)
#         z = self.ln1(x + self.sa(x))

#         z = self.ln2(z + self.ffwd(z))
#         return z

# class RoPEModel(nn.Module):
#     def __init__(self, vocab_size, d_model, n_heads, hidden, n_layers, dropout) -> None:
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, d_model)
#         self.pos_emb = PositionalEncoding(d_model, dropout)
#         self.lm_head = nn.Linear(d_model, vocab_size)
        
#         self.blocks = nn.Sequential(
#             *[Block(d_model, hidden, n_heads) for _ in range(n_layers)],
#             nn.LayerNorm(d_model)
#         )
#     def forward(self, x):
#         z = self.embedding(x)
#         z = self.pos_emb(z)
#         z = self.blocks(z)
#         z = self.lm_head(z)
#         return torch.log(softmax_one(z, dim=-1))
    
#     def init_weights(self):
#         pass
    
        
def softmax_one(x:torch.Tensor, dim=None):
    #subtract the max for stability
    x = x - x.max(dim=dim, keepdim=True).values
    #compute exponentials
    exp_x = torch.exp(x)
    #compute softmax values and add on in the denominator
    return exp_x / (1 + exp_x.sum(dim=dim, keepdim=True))



class TinyL(nn.Module):
    def __init__(self, vocab, hidden_dim, ffn_hidden, heads,n_layers=18,rope=False) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab, hidden_dim)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(TinyLBlock(hidden_dim, heads, ffn_hidden,rope))
        self.ln = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab)
        # self.init_weights()
    def forward(self, input:torch.Tensor):
        x = self.embedding(input)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.ln(x)
        x = self.lm_head(x)
        # print(x)
        # assert 0
        return torch.log_softmax(x, dim=-1)
    def init_hidden(self, bsz):
        pass
    # def init_weights(self):
    #     # initrange = 0.1
    #     for layer in self.layers:
    #         for p in layer.parameters():
    #             if p.dim() > 1:
    #                 nn.init.xavier_uniform_(p)
        # nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        # nn.init.zeros_(self.decoder.bias)
        # nn.init.uniform_(self.decoder.weight, -initrange, initrange)
    
class FFN_L(nn.Module):
    def __init__(self, hidden_dim, ffn_hidden_dim) -> None:
        super().__init__()
        self.l1 = nn.Linear(hidden_dim, ffn_hidden_dim)
        self.act = nn.SiLU()
        self.l2 = nn.Linear(hidden_dim, ffn_hidden_dim)
        self.collect = nn.Linear(ffn_hidden_dim, hidden_dim)
    def forward(self, input:torch.Tensor):
        x = self.l1(input)
        x = self.act(x)
        y = self.l2(input)
        x = x*y
        
        return self.collect(x)
    
class TinyLBlock(nn.Module):
    def __init__(self, hidden_size, heads, ffn_hidden_dim, rope=False) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = RoPEAttention(heads, hidden_size, 0.2) if rope else MultiHeadAttention(heads, hidden_size, 0.2)

        self.ln2 = nn.LayerNorm(hidden_size)
        self.ffn = FFN_L(hidden_size, ffn_hidden_dim)
    def forward(self, input:torch.Tensor):
        x = self.ln1(input)
        x = self.attn(x)
        
        y = x + input
        # print(x)
        # assert 0
        x = self.ln2(y)
        x = self.ffn(x)
        x = x + y
        return x
        

        