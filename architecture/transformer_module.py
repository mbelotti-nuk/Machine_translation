import torch
import torch.nn as nn
import math
from torch.nn.functional import softmax, log_softmax


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, n_layers, n_heads, n_embed, context_length, dropout):
        super().__init__()
        # embedding + positional encoding 
        self.src_embedding = nn.Sequential( nn.Embedding(src_vocab_size, n_embed), PositionalEncoding(n_embed) )
        self.trg_embedding = nn.Sequential( nn.Embedding(trg_vocab_size, n_embed), PositionalEncoding(n_embed) )

        self.encoder = Encoder(n_layers, n_heads, n_embed, dropout)
        self.decoder = Decoder(n_layers, n_heads, n_embed, dropout)
        
        self.projection = nn.Linear(n_embed, trg_vocab_size)
        self.init_weights()

        self.temperature = 0.2*torch.ones(trg_vocab_size)

        self.context_length = context_length
        self.register_buffer(
            'full_trg_mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

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
        """This mask is applied to source sentences which are filled with [PAD]
        token if they end before the largest dimension of a sentece in a batch
        of sentences

        Args:
            src (torch_Tensor): _description_
            pad (_type_): character used for padding

        Returns:
            _type_: _description_
        """
        return (src != pad).unsqueeze(-2).unsqueeze(-2)
    

    def make_tgt_mask(self, tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2).unsqueeze(-2)
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
        #return self.full_trg_mask[:size, :size]
    
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
        self.multi_head_attn = MultiHeadAttentionWrapper(n_heads, n_embed) # MultiHeadAttention(n_heads, n_embed)
        #self.multi_head_attn = MultiHeadAttention(n_heads, n_embed)
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
        self.self_attention = MultiHeadAttentionWrapper(n_heads, n_embed) # MultiHeadAttention(n_heads, n_embed) 
        #self.self_attention = MultiHeadAttention(n_heads, n_embed) 
        self.norm_1 = NormLayer(n_embed)
        # second step: cross-attention between embdeddings of encoder and the embeddings of the decoder
        self.cross_attention = MultiHeadAttentionWrapper(n_heads, n_embed) # MultiHeadAttention(n_heads, n_embed)
        self.norm_2 = NormLayer(n_embed)
        # third step: reflection on previous attention blocks
        self.ff = FeedForward(n_embed)
        self.norm_3 = NormLayer(n_embed)  
        # dropout
        self.dropout = nn.Dropout(dropout)


    def forward(self, src, x, src_mask, trg_mask):
        apply_self_attention = lambda x: self.self_attention(x, x, x, trg_mask)
        # first step with skip connections and dropout - self-attention on translated sentence
        x = x + self.dropout( apply_self_attention( self.norm_1( x ) ) ) # (B,T,E)

        # second step with skip connections and dropout - cross-attention between embdeddings of encoder and the embeddings of the decoder
        apply_cross_attention = lambda x,y: self.cross_attention(x, y, y, src_mask)
        #query --> x
        #key   --> src
        #value --> src
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


class MultiHeadAttentionWrapper(nn.Module):
    """
        Multi head attention in ones step
    """
    def __init__(self, n_heads, n_embed):
        super().__init__()
        assert n_embed % n_heads == 0, f"The embedding size (={n_embed}) and the number of heads (={n_heads}) must be divisible"
        
        self.n_heads = n_heads
        self.head_size = n_embed // n_heads

        self.K = nn.Linear(n_embed, n_embed, bias=False)
        self.Q = nn.Linear(n_embed, n_embed, bias=False)
        self.V = nn.Linear(n_embed, n_embed, bias=False)


    def forward(self, query:torch.Tensor, key:torch.Tensor, value:torch.Tensor, mask):
        # batch, time, channels. 
        # --> time is equal to context window
        # --> channels is equal to the embed size
        B, Tq, Cq = query.shape 
        B, Tk, Ck = key.shape

        key = self.K(key)       #(B, Tk, Ck)
        query = self.Q(query)   #(B, Tq, Cq)
        value = self.V(value)   #(B, Tk, Ck)

        key = key.view(B, Tk, self.n_heads, self.head_size)      #(B, Tk, N_HEADS, HEAD_SIZE)
        query = query.view(B, Tq, self.n_heads, self.head_size)  #(B, Tq, N_HEADS, HEAD_SIZE)
        value = value.view(B, Tk, self.n_heads, self.head_size)  #(B, Tk, N_HEADS, HEAD_SIZE)

        key = key.transpose(1,2)        #(B, N_HEADS, Tk, HEAD_SIZE)
        query = query.transpose(1,2)    #(B, N_HEADS, Tq, HEAD_SIZE)
        value = value.transpose(1,2)    #(B, N_HEADS, Tk, HEAD_SIZE)

        affinities = self.attn_weights(query, key, mask) # (B, N_HEADS,Tk,Tq)

        attn = affinities @ value # (B, N_HEADS,Tq,Tk) @ (B,N_HEADS,Tk,HEAD_SIZE)= (B,N_HEADS,Tq,HEAD_SIZE)
        attn = attn.transpose(1, 2) # (B,Tq,N_HEADS,HEAD_SIZE)
        attn = attn.contiguous().view(B, Tq, Cq)
        return attn

    def attn_weights(self, query, key, mask=None)->torch.Tensor:
        # compute attention scores
        affinities = query @ key.transpose(-2,-1) # (B, N_HEADS, Tq, HEAD_SIZE) @ (B, N_HEADS, HEAD_SIZE, Tk) = (B, N_HEADS,Tq,Tk)
        # scale attention
        head_size = query.size(-1)
        affinities = affinities * head_size ** (-0.5)

        if mask is not None:
            # il mask puo avvenire sia per source_sentences sia per target_sentences
            # nel primo caso per mascherare i pad di riempimento
            # nel secondo per mascherare i token futuri della frase che in questo caso verra tradotta
            affinities = affinities.masked_fill(mask==0, -torch.inf)
    
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

   