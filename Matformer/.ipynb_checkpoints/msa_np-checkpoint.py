import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pickle
from utils import parse_args
import pdb

args = parse_args()
def build_model(vocab, tgt, dist_bar, N=6, embed_dim=args.embed_dim, ffn_dim=args.fnn_dim, head=4, dropout=args.dropout, out_both=False):
    c = copy.deepcopy
    attn = MultiHeadedAttention(head, embed_dim, dist_bar)
    ff = FeedForward(embed_dim, ffn_dim, dropout)
    position = PositionalEncoding3D(embed_dim, dropout)

    model = Encoder3D(Encoder(EncoderLayer(embed_dim, c(attn), c(ff), dropout), N), Embeddings(embed_dim, vocab),
                      c(position), Generator3D(embed_dim, tgt, dropout), out_both)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


class Encoder3D(nn.Module):
    def __init__(self, encoder, src_embed, src_pe, generator, out_both):
        super(Encoder3D, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.src_pe = src_pe
        self.generator = generator
        self.dist = None
        self.out_both = out_both

    def forward_once(self, src, src_mask, pos):
        dist = torch.cdist(pos, pos)
        # dist = torch.cdist(pos, pos) < self.dist_bar
        # dist[:, 0, :] = 1
        # dist[:, :, 0] = 1

        self.dist = dist

    
        return self.encoder(self.src_pe(self.src_embed(src), pos), dist.unsqueeze(1), src_mask)[:, 0, :]

    def forward(self, src, src_mask, pos):
        if self.out_both:
            h = self.forward_once(src, src_mask, pos)
            return self.generator(h), h

        return self.generator(self.forward_once(src, src_mask, pos))


#######################################
## Encoder
#######################################

class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, dist, mask):
        for layer in self.layers:
            x = layer(x, dist, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    """Encoder_XL is made up of self-attn and feed forward (defined below)"""

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, dist, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, dist, mask))
        return self.sublayer[1](x, self.feed_forward)


class Generator3D(nn.Module):
    """Define standard linear + activation generation step."""

    def __init__(self, embed_dim, tgt, dropout):
        super(Generator3D, self).__init__()
        self.tgt = tgt
        self.proj = nn.Linear(embed_dim, tgt)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.dropout(self.proj(x))


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


#######################################
## attention
#######################################

def constrained_attention(query, key, value, dist, dist_bar, mask=None, dropout=None):
    """Compute Scaled Dot Product Attention"""
    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

        # scores = scores.masked_fill(dist == 0, -1e9)
    
    for tmp in dist_bar:
        tmp_scores_dist = []
        for index, i in enumerate(tmp):
            tmp_scores = scores[:, index, :, :].unsqueeze(1)
            dist_mask = dist < i
            
            dist_mask[:, :, 0, :] = 1
            dist_mask[:, :, :, 0] = 1
            
            tmp_scores_dist.append(tmp_scores.masked_fill(dist_mask == 0, -1e9))

    scores_dist = torch.cat(tmp_scores_dist, dim=1)
    p_attn = F.softmax(scores_dist, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, embed_dim, dist_bar, dropout=0.1):
        """Take in model size and number of heads."""
        super(MultiHeadedAttention, self).__init__()
        assert embed_dim % h == 0
        self.h = h
        self.linears = clones(nn.Linear(embed_dim, embed_dim), 4)
        self.dist_bar = dist_bar
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

        self.d_k = embed_dim // h

    def forward(self, query, key, value, dist, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from embed_dim => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = constrained_attention(query, key, value, dist, self.dist_bar,  mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


#######################################
## Encoder
#######################################


class LayerNorm(nn.Module):
    """ layernorm"""

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True) + self.eps
        return self.a_2 * (x - mean) / std + self.b_2


class SublayerConnection(nn.Module):
    """A residual connection followed by a layer norm. For code simplicity the norm is first as opposed to last."""

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))


#######################################
## Position-wise
#######################################

class FeedForward(nn.Module):
    """Implements FFN equation."""

    def __init__(self, embed_dim, ffn_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(embed_dim, ffn_dim)
        self.w_2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


#######################################
## Embedding
#######################################

class PositionalEncoding3D(nn.Module):
    def __init__(self, embed_dim, dropout):
        super(PositionalEncoding3D, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embed_dim = embed_dim
        self.pe_linear=nn.Linear(3*embed_dim,embed_dim)

    def forward(self, x, pos):
      
        pos = pos * 10

       
        div = torch.exp(torch.arange(0., self.embed_dim, 2) * -(math.log(10000.0) / self.embed_dim)).double().cuda()
        for i in range(3):
            pe = torch.zeros(x.shape).cuda()
            pe[..., 0::2] = torch.sin(pos[..., i].unsqueeze(2) * div)
            pe[..., 1::2] = torch.cos(pos[..., i].unsqueeze(2) * div)
            if i==0:
                pe_all=pe
            else:
                pe_all=torch.cat((pe_all,pe),dim=-1)
        pe_final=self.pe_linear(pe_all)
        x += pe_final
        return self.dropout(x)


class Embeddings(nn.Module):    
    def __init__(self, embed_dim, vocab):
        super(Embeddings, self).__init__()
        loaded_dict = pickle.load(open('RotatE_512_256_emb.pkl', 'rb'))
        atomic_emb = loaded_dict['atomic_emb']
        self.embed = nn.Embedding.from_pretrained(atomic_emb[:101], freeze=False)   # (101, 512)
        # self.embed = nn.Embedding(vocab, embed_dim)     # (101, 512)
        self.embed_dim = embed_dim

    def forward(self, x):
        
        return self.embed(x) * math.sqrt(self.embed_dim)    # [8, 1024, 512]