import random
import torch
import random
from torch import nn
import torch.nn.functional as F

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        assert (encoder.n_layers == decoder.n_layers), "num. of layers of encoder and decoder must be equal"

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def init_weights(self):        
        for name, param in self.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)


    def forward(self, src, trg, teacher_forcing_ratio):
        # src = [batch_size, src_length]
        # trg = [batch size, trg_length]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        batch_size, trg_length  = trg.shape
        trg_vocab_size = self.decoder.output_dim

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        encoder_output, encoder_hidden, encoder_cell = self.encoder(src)

        # tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_length, trg_vocab_size).to(self.device)
        # Initialize decoder hidden state
        decoder_hidden = torch.zeros_like(encoder_hidden)  
        decoder_cell = torch.zeros_like(encoder_hidden)

        # first input to the decoder is the [SOS] tokens
        input = trg[:, 0]
        # input = [batch size]
        for t in range(1, trg_length):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, decoder_hidden, decoder_cell = self.decoder(input.unsqueeze(1), decoder_hidden, decoder_cell, encoder_output)
            # place predictions in a tensor holding predictions for each token
            outputs[:, t, :] = output.squeeze(1)
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            # get the highest predicted token from our predictions
            best_guess = output.argmax(2).squeeze(1)
            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[:, t] if teacher_force else best_guess
            # input = [batch size]
        return outputs
    
    
    
# based on https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html
class seq2seq_attention(nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim, attention_dim, bi_version=False):
        super(seq2seq_attention, self).__init__()
        self.bi_version = bi_version
        if bi_version:
            self.W_h = nn.Linear(encoder_hidden_dim, attention_dim)
            self.W_s = nn.Linear(decoder_hidden_dim, attention_dim)
        else:
            self.W = nn.Linear(encoder_hidden_dim + decoder_hidden_dim, attention_dim)
        self.v_a = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden  ->  [n_layers, batch_size, hidden_dim] -> current decoder hidden state
        # encoder_outputs ->  [batch_size, sequence_length , hidden_dim * n_directions] = [batch_size, sequence_length, encoder_hidden_dim]
        sequence_length = encoder_outputs.shape[1]
        # take last layer's hidden state and repeat for all the sequence length
        decoder_hidden = decoder_hidden[-1].unsqueeze(1).repeat(1, sequence_length, 1) # [batch_size, sequence_length, hidden_dim]
        # compute MLP attention score (Bahdanau attention)
        # the scores indicates how each source token (represented by encoder_outputs) is relevant for the target step (represented by decoder_hidden)
        scores = self.get_scores( decoder_hidden, encoder_outputs ) # [batch_size, hidden_dim]
        attention = F.softmax(scores, dim=-1)
        return attention  # [batch_size, sequence_length]
    
    def get_scores(self, decoder_hidden, encoder_outputs):
        if self.bi_version:
            return self.v_a( torch.tanh(self.W_s(decoder_hidden) + self.W_h(encoder_outputs)) ).squeeze(-1)
        else:
            return self.v_a( torch.tanh(self.W( torch.cat([decoder_hidden, encoder_outputs], dim=-1)) ) ).squeeze(-1)

class seq2seq_encoder(nn.Module):
    def __init__(self, vocab_size:int, embedding_dim:int, hidden_dim:int, n_layers:int, dropout:float, bidirectional=False):
        """Encoder layer
        Args:
            vocab_size (int): Dimension of the sequence of tokens
            embedding_dim (int): Dimension of the embedding layer
            hidden_dim (int): Dimension of the hidden state in the LSTM
            n_layers (int): Number of layers in the LSTM
            dropout (float): Dropout rate to be applied
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           n_layers, 
                           dropout=dropout,  
                           batch_first=True,
                           bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src -> [batch_size, sequence_length]
        embedded = self.dropout(self.embedding(src))
        # embedded_vec -> [batch_size, sequence_length, embedding dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs -> [batch_size, sequence_length, hidden_dim * n directions] 
        # hidden -> [n_layers * n directions, batch_size, hidden_dim] 
        # cell -> [n_layers * n directions, batch size, hidden dim] 
        if self.bidirectional:
            hidden = torch.cat([ hidden[i].unsqueeze(0) for i in range(-1*self.n_layers, 0)], dim=0) 
            # hidden -> [n_layers, batch_size, hidden_dim]  
        return outputs, hidden, cell
    
class seq2seq_decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, dropout, attention=True, enc_hidden_dim=None):
        super().__init__()
        self.attention_mech = attention
        self.output_dim = vocab_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        enc_hidden_dim = enc_hidden_dim if enc_hidden_dim != None else hidden_dim    
        inp_size = embedding_dim + enc_hidden_dim if attention else embedding_dim
        self.rnn = nn.LSTM(inp_size, 
                           hidden_dim, 
                           n_layers, 
                           dropout=dropout, 
                           batch_first=True)
        if self.attention_mech:
                self.attention = seq2seq_attention(encoder_hidden_dim = enc_hidden_dim, 
                                                   decoder_hidden_dim =hidden_dim, 
                                                   attention_dim = hidden_dim) 
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, decoder_hidden, decoder_cell, encoder_outputs=None):
        # input ->  [batch size, 1] which is one single token
        # decoder_hidden -> [n layers * n_directions, batch_size, hidden_dim] = [n layers, batch_size, hidden_dim]
        # decoder_cell ->   [n layers * n_directions, batch size, hidden_dim] = [n layers, batch size, hidden_dim]
        # encoder_outputs -> [batch_size, sequence_length, hidden_dim * n_directions] = [batch_size, sequence_length, encoder_hidden_dim]

        embedded = self.dropout(self.embedding(input))

        if self.attention_mech:
            # Compute attention weights and context vector
            # these are a probability distribution over the sequence length
            attn_weights = self.attention(decoder_hidden, encoder_outputs)  # [batch_size, sequence_length]
            # for each sequence in the batch
            # att_weights [sequence_length,1] * encoder_outputs [sequence_length, encoder_hidden_dim]
            # which is a weighted sum of encoder states with attention weights.
            context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # [batch_size, 1, encoder_hidden_dim]

            # Combine embedded input and context vector
            rnn_input = torch.cat((embedded, context), dim=2)  # [batch_size, 1, embed_dim + encoder_hidden_dim]
        else:
            rnn_input = embedded

        # Pass through LSTM
        output, (decoder_hidden, decoder_cell) = self.rnn(rnn_input, (decoder_hidden, decoder_cell))  
        # output -> [batch_size, 1, hidden dim]
        # hidden -> [batch_size, n layers, hidden dim]
        # cell ->   [batch_size, n layers, hidden dim]
        # Final prediction
        #prediction = self.fc_out(torch.cat((output.squeeze(1), context.squeeze(1)), dim=1))  # [batch_size, output_dim]
        prediction = self.fc_out(output.squeeze(0))
        return prediction, decoder_hidden, decoder_cell

