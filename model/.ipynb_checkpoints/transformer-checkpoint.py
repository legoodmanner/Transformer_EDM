from fast_transformers import attention
from fast_transformers.attention.attention_layer import AttentionLayer
import torch
import torch.nn as nn
import math
from torch import Tensor
import numpy as np

from fast_transformers.transformers import TransformerEncoderLayer, TransformerEncoder
from fast_transformers.attention import FullAttention, LinearAttention
from fast_transformers.masking import FullMask, LengthMask, TriangularCausalMask
from torch.nn.functional import embedding, threshold

def nucleus(probs, p=0.5):
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    probs /= (sum(probs) + 1e-5)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    cusum_sorted_probs = np.cumsum(sorted_probs)
    after_threshold = cusum_sorted_probs > p
    if sum(after_threshold) > 0:
        last_index = np.where(after_threshold)[0][0] + 1
        candi_index = sorted_index[:last_index]
    else:
        candi_index = sorted_index[:]
    candi_probs = [probs[i] for i in candi_index]
    candi_probs /= sum(candi_probs)
    word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    return word


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.permute(1,0,2)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:,:x.size(1),:]
        return self.dropout(x)

class VQTransformer(nn.Module):
    def __init__(self, d_model, n_layer, n_embed, softmax_temp=None):
        super().__init__()
        layers = []
        for n in range(n_layer):
            layers.append(
                TransformerEncoderLayer(
                    attention=AttentionLayer(FullAttention(softmax_temp=softmax_temp), d_model, 16),
                    d_model=d_model,
                    d_ff=4
                )
            )
        self.n_embed = n_embed
        self.layers = nn.ModuleList(layers)
        self.pos_enc = PositionalEncoding(d_model, max_len=100)
        self.token_embed = nn.Embedding(n_embed+1, d_model)
        self.output_mapping = nn.Linear(d_model, n_embed)

    
    def forward(self, x, attn_mask=None, length_mask=None):
        # Input Shape: [bs, seq_len] (token id)
        
        # x = x.permute(0,2,1) # [bs, d_model, seq_len] --> [bs, seq_len, d_model]
        N = x.shape[0]
        L = x.shape[1]
        if not isinstance(x, torch.LongTensor):
            x = x.long()
       
        attn_mask = TriangularCausalMask(N=L, device=x.device)
        length_mask = length_mask or \
            LengthMask(x.new_full((N,), L, dtype=torch.int64))
        start = torch.LongTensor([[self.n_embed]]*N).to(x.device) #[bs, 1]
        x = torch.cat((start, x[:,:-1]), dim=-1) # [bs, seq_len]
        x = self.token_embed(x) # [bs, seq_len, d_model]
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask, length_mask=length_mask)
        # p = torch.softmax(self.output_mapping(x), dim=-1) # probability distribution of all token embeddeding | shape:[bs, seq_len, n_embed]
        p = self.output_mapping(x)
        return p
    
    def evaluate(self, n_sample, L, quantizer, device, threshold_p=1):
        x = torch.LongTensor([[self.n_embed]] * n_sample).to(device) #[n_sample, 1]
        pl = []
        for l in range(0, L):
            px = self.token_embed(x)
            px = self.pos_enc(px)
            for layer in self.layers:
                px = layer(px, attn_mask=None) # [n_sample, l, embed_dim]
            p = torch.softmax(self.output_mapping(px), dim=-1)[:,-1,:] # [n_sample, n_embed]
            
            pred_id = torch.empty(size=(n_sample,1)).long().to(device)
            for i in range(n_sample):
                pred_id[i,0] = nucleus(p[i], p=threshold_p) # [n_sample, 1]
            pl.append(pred_id[:,0])
            x = torch.cat((x, pred_id), dim=-1) # [n_sample, l]
            
        pred_quant = quantizer.embed_code(x[:,1:]) # [n_sample, seq_len]
    
        return pred_quant, pl #[n_sample, L, embed_dim]