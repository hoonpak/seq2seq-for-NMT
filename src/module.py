import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import config
from attention import Attention

class PositionRegression(nn.Module):
    def __init__(self, hidden_size):
        super(PositionRegression, self).__init__()
        self.W_p = nn.Linear(hidden_size, hidden_size, bias=False)
        self.tanh = nn.Tanh()
        self.v_p = nn.Linear(hidden_size, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, output):
        w_p_out = self.W_p(output)
        tanh_out = self.tanh(w_p_out)
        v_p_out = self.v_p(tanh_out)
        sig_out = self.sigmoid(v_p_out)
        position = torch.mul(config.MAX_LENGTH+2, sig_out)
        return position
    
    def initialization(self):
        nn.init.uniform_(self.W_p.weight, a=-config.uniform_init_range, b=config.uniform_init_range)
        nn.init.uniform_(self.v_p.weight, a=-config.uniform_init_range, b=config.uniform_init_range)

class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size, padding_idx=config.PAD)
        self.dropout = nn.Dropout(p=dropout)
        self.lstm_layer = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        
    def forward(self, input, lengths):
        """
        input (N, L)
        h_0   (num of layers, N, H)
        c_0   (num of layers, N, H)
        """
        emb = self.embedding_layer(input) #batch size, max lenth, dimension
        emb = self.dropout(emb)
        packed_emb = pack_padded_sequence(input=emb, lengths=lengths, batch_first=True, enforce_sorted=False)
        packed_output, (h_n, c_n) = self.lstm_layer(packed_emb) #it's the result of to compute every step each layers.
        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=config.MAX_LENGTH+2)
        h_n = torch.clamp(h_n, min=-config.clipForward, max=config.clipForward)
        c_n = torch.clamp(c_n, min=-config.clipForward, max=config.clipForward)
        return output, h_n, c_n
    
    def initialization(self):
        nn.init.uniform_(self.embedding_layer.weight, a=-config.uniform_init_range, b=config.uniform_init_range)
        for i in range(config.num_layers):
            nn.init.uniform_(getattr(self.lstm_layer, f'weight_ih_l{i}'), a=-config.uniform_init_range, b=config.uniform_init_range)
            nn.init.uniform_(getattr(self.lstm_layer, f'weight_hh_l{i}'), a=-config.uniform_init_range, b=config.uniform_init_range)
            nn.init.uniform_(getattr(self.lstm_layer, f'bias_ih_l{i}'), a=-config.uniform_init_range, b=config.uniform_init_range)
            nn.init.uniform_(getattr(self.lstm_layer, f'bias_hh_l{i}'), a=-config.uniform_init_range, b=config.uniform_init_range)

class Decoder(nn.Module):
    def __init__(self, attn_type, align_type, input_feeding, vocab_size, hidden_size, num_layers, dropout, is_reverse):
        super(Decoder, self).__init__()
        
        self.embedding_layer = nn.Embedding(vocab_size, hidden_size, padding_idx=config.PAD)
        self.dropout = nn.Dropout(p=dropout)
        
        if input_feeding:
            self.lstm_layer = nn.LSTM(2*hidden_size, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        else:
            self.lstm_layer = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        
        if attn_type != 'no':
            self.attn_layer = Attention(hidden_size, attn_type, align_type)
            if attn_type == "local_p":
                self.position_layer = PositionRegression(hidden_size)
        
        self.output_layer = nn.Linear(hidden_size, vocab_size, bias=False)
        
        self.attn_type = attn_type
        self.input_feeding = input_feeding
        self.reverse = is_reverse
        
    def forward(self, src_len, encoder_outputs, h_0, c_0, target = None):
        """
        h_0            (num of layers, N, H)
        c_0            (num of layers, N, H)
        """
        decoder_input = target[:,0].unsqueeze(1) #N, L -> N, 1
        decoder_hidden = h_0
        decoder_cell = c_0
        decoder_outputs = []
        attn_vec = torch.zeros(config.batch_size, 1, config.dimension).to(config.device) #N, 1, H
        for time_step in range(1, config.MAX_LENGTH+2): # total processing time -> 51 // total tgt time step -> 52
            if self.attn_type != 'no':
                decoder_output, attn_vec, decoder_hidden, decoder_cell = self.forward_step(src_len=src_len, encoder_outputs=encoder_outputs, attn_vec=attn_vec, time_step=time_step,
                                                                                        input=decoder_input, hidden=decoder_hidden, cell=decoder_cell)
            else:
                decoder_output, decoder_hidden, decoder_cell = self.forward_step(src_len=src_len, encoder_outputs=encoder_outputs, attn_vec=attn_vec, time_step=time_step,
                                                                                input=decoder_input, hidden=decoder_hidden, cell=decoder_cell)
            decoder_outputs.append(decoder_output) #decoder_output -> (N, 1, V)
            decoder_input = target[:,time_step].unsqueeze(1) #N, 1
                
        decoder_outputs = torch.cat(decoder_outputs, dim=1) #decoder_outputs -> (L, N, 1, V) => (N, L, V) L = 51
        return decoder_outputs, decoder_hidden, decoder_cell
    
    def forward_step(self, src_len, encoder_outputs, attn_vec, time_step, input, hidden, cell):
        """
        input  (N, 1)
        hidden (num of layers, H)
        cell   (num of layers, H)
        """
        emb = self.embedding_layer(input) #N, 1, H
        emb = self.dropout(emb)
        if self.input_feeding:
            print(emb.shape, attn_vec.shape)
            emb = torch.cat((emb, attn_vec), dim=2)
        output_t, (h_t, c_t) = self.lstm_layer(emb, (hidden, cell))
        h_t = torch.clamp(h_t, min=-config.clipForward, max=config.clipForward)
        c_t = torch.clamp(c_t, min=-config.clipForward, max=config.clipForward)
        if self.attn_type != 'no':
            if self.attn_type == 'global':
                attn_vec = self.attn_layer(encoder_outputs, output_t, src_len)
            else:
                if self.attn_type == 'local_m':
                    if self.reverse:
                        p_t = torch.max(src_len - time_step, torch.fill(torch.empty_like(src_len),-1))
                    else:
                        p_t = time_step.repeat(config.batch_size)
                else:
                    p_t = self.position_layer(output_t).squeeze()
                attn_vec = self.attn_layer(encoder_outputs, output_t, src_len, p_t) #p_t (N,)
            output = self.output_layer(attn_vec) #N, 1, V
            return output, attn_vec, h_t, c_t
        else:
            output = self.output_layer(output_t) #N, 1, V
            return output, h_t, c_t
    
    def initialization(self):
        nn.init.uniform_(self.embedding_layer.weight, a=-config.uniform_init_range, b=config.uniform_init_range)
        for i in range(config.num_layers):
            nn.init.uniform_(getattr(self.lstm_layer, f'weight_ih_l{i}'), a=-config.uniform_init_range, b=config.uniform_init_range)
            nn.init.uniform_(getattr(self.lstm_layer, f'weight_hh_l{i}'), a=-config.uniform_init_range, b=config.uniform_init_range)
            nn.init.uniform_(getattr(self.lstm_layer, f'bias_ih_l{i}'), a=-config.uniform_init_range, b=config.uniform_init_range)
            nn.init.uniform_(getattr(self.lstm_layer, f'bias_hh_l{i}'), a=-config.uniform_init_range, b=config.uniform_init_range)
        nn.init.uniform_(self.output_layer.weight, a=-config.uniform_init_range, b=config.uniform_init_range)
        
        if self.attn_type != 'no':
            self.attn_layer.initialization()
            if self.attn_type == "local_p":
                self.position_layer.initialization()