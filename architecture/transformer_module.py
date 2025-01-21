import torch
import torch.nn as nn
import math
from torch.nn.functional import softmax, log_softmax


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, n_layers, n_heads, n_embed, dropout):
        super().__init__()
        # embedding + positional encoding 
        self.src_embedding = nn.Sequential( nn.Embedding(src_vocab_size, n_embed), PositionalEncoding(n_embed) )
        self.trg_embedding = nn.Sequential( nn.Embedding(trg_vocab_size, n_embed), PositionalEncoding(n_embed) )

        self.encoder = Encoder(n_layers, n_heads, n_embed, dropout)
        self.decoder = Decoder(n_layers, n_heads, n_embed, dropout)
        
        self.projection = nn.Linear(n_embed, trg_vocab_size)
        self.init_weights()
        self.temperature = 4*torch.ones(trg_vocab_size)

    def forward(self, src, trg, src_mask, trg_mask):
        return self.decode(self.encode(src, src_mask), trg, src_mask, trg_mask)

    def encode(self, src, src_mask):
        return self.encoder( self.src_embedding(src), src_mask )

    def decode(self, memory, trg, src_mask, trg_mask):
        return self.decoder(memory, self.trg_embedding(trg), src_mask, trg_mask)
    
    def get_logits(self, x):
        return self.projection(x)
    
    def get_probabilities(self, x):
        return softmax(self.projection(x)/self.temperature.to(x.device), dim=-1)
    
    def get_masks(self, src, tgt, pad):
        return self.make_src_mask(src, pad), self.make_tgt_mask(tgt, pad)

    def make_src_mask(self, src, pad):
         return (src != pad).unsqueeze(-2)

    def make_tgt_mask(self, tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        # hide padding and future words
        tgt_mask = tgt_mask & self.subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask
    
    def subsequent_mask(self, size):
        # mask out subsequent positions
        attn_shape = (1, size, size)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
            torch.uint8
        )
        return subsequent_mask == 0
    
    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    

class Encoder(nn.Module):
    def __init__(self, n_layers, n_heads, n_embed, dropout=0.1):
        super().__init__()
        self.encoder_layers = nn.ModuleList([EncoderLayer(n_heads, n_embed, dropout) for i in range(n_layers)])
        self.norm = NormLayer(n_embed)

    def forward(self, x, mask):
        for enc_layer in self.encoder_layers:
            x = enc_layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, n_layers, n_heads, n_embed, dropout=0.1):
        super().__init__()
        self.decoder_layers = nn.ModuleList([DecoderLayer(n_heads, n_embed, dropout) for i in range(n_layers)])
        self.norm = NormLayer(n_embed)

    def forward(self, memory, x, src_mask, trg_mask):
        for dec_layer in self.decoder_layers:
            x = dec_layer(memory, x, src_mask, trg_mask)
        return self.norm(x)        


class EncoderLayer(nn.Module):
    def __init__(self, n_heads, n_embed, dropout=0.1):
        super().__init__()
        # first step: attention
        self.multi_head_attn = MultiHeadAttention(n_heads, n_embed)
        self.norm_1 = NormLayer(n_embed)
        # second step: reflection on attention
        self.ff = FeedForward(n_embed)
        self.norm_2 = NormLayer(n_embed)
        # dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        apply_attention = lambda x: self.multi_head_attn(x,x,x, mask)
        # first step with skip connections and dropout
        x = x + self.dropout( apply_attention( self.norm_1( x ) ) ) # (B,T,E)
        # second step with skip connections and dropout
        x = x + self.dropout( self.ff( self.norm_2( x ) ) ) # (B,T,E)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, n_heads, n_embed, dropout=0.1):
        super().__init__()
        # first step: self-attention on translated sentence
        self.self_attention = MultiHeadAttention(n_heads, n_embed) 
        self.norm_1 = NormLayer(n_embed)
        # second step: cross-attention between embdeddings of encoder and the embeddings of the decoder
        self.cross_attention = MultiHeadAttention(n_heads, n_embed)
        self.norm_2 = NormLayer(n_embed)
        # third step: reflection on previous attention blocks
        self.ff = FeedForward(n_embed)
        self.norm_3 = NormLayer(n_embed)  
        # dropout
        self.dropout = nn.Dropout(dropout)


    def forward(self, src, x, src_mask, trg_mask):
        apply_self_attention = lambda x: self.self_attention(x, x, x, trg_mask)
        # first step with skip connections and dropout
        x = x + self.dropout( apply_self_attention( self.norm_1( x ) ) ) # (B,T,E)

        # second step with skip connections and dropout
        apply_cross_attention = lambda x,y: self.self_attention(x, y, y, src_mask)
        x = x + self.dropout( apply_cross_attention( self.norm_2( x ), src ) ) # (B,T,E)

        # third step with skip connections and dropout
        x = x + self.dropout( self.ff( self.norm_3( x ) ) ) # (B,T,E)
        return x



class NormLayer(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(NormLayer, self).__init__()
        self.beta = nn.Parameter(torch.ones(features))
        self.gamma = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.beta * (x - mean) / (std + self.eps) + self.gamma
    

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, n_embed):
        super().__init__()
        assert n_embed % n_heads == 0, f"The embedding size (={n_embed}) and the number of heads (={n_heads}) must be divisible"
        head_size = n_embed // n_heads
        self.heads = nn.ModuleList( [ AttentionHead(n_embed, head_size) for i in range(n_heads) ] )
    
    def forward(self, query, key, value, mask):
        return torch.cat([head(query, key, value, mask) for head in self.heads], dim=-1) # -> (B,T,EMBED_SIZE)


class AttentionHead(nn.Module):
    def __init__(self, n_embed, head_size):
        super().__init__()
        self.K = nn.Linear(n_embed, head_size, bias=False) # key matrix
        self.Q = nn.Linear(n_embed, head_size, bias=False) # query matrix
        self.V = nn.Linear(n_embed, head_size, bias=False) # value matrix

    def forward(self, query:torch.Tensor, key:torch.Tensor, value:torch.Tensor, mask=None):
        # batch, time, channels
        # B,T,C = query.shape 

        key = self.K(key)       #(B,T,H)  where H is the head size
        query = self.Q(query)   #(B,T,H)
        value = self.V(value)   #(B,T,H)
        affinities = self.attn_weights(query, key, mask)

        return affinities @ value # (B,T,T) @ (B,T,H) = (B,T,H)
    
    def attn_weights(self, query, key, mask=None):
        # compute attention scores
        affinities = query @ key.transpose(-2,-1) # (B,T,H) @ (B,H,T) = (B,T,T)
        # scale attention
        head_size = query.size(-1)
        affinities = affinities * head_size ** (-0.5)
        # mask
        if mask is not None:
            affinities = affinities.masked_fill(mask==0, float('-inf'))
            #print(f"affinities {affinities[0,0,:]}")
        # compute probabilities
        return affinities.softmax(-1) 
 

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(n_embed, n_embed),
            nn.ReLU())
    def forward(self, x):
        return self.linear(x)
    

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, dropout=0.1, max_seq_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_seq_len, embedding_dim)
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2) * -(math.log(10000.0) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

   