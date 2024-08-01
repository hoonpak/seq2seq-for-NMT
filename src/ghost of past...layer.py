import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import config
from score import Score

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
        return output, h_n, c_n
    
    def initialization(self):
        nn.init.uniform_(self.embedding_layer.weight, a=-config.uniform_init_range, b=config.uniform_init_range)
        for i in range(config.num_layers):
            nn.init.uniform_(getattr(self.lstm_layer, f'weight_ih_l{i}'), a=-config.uniform_init_range, b=config.uniform_init_range)
            nn.init.uniform_(getattr(self.lstm_layer, f'weight_hh_l{i}'), a=-config.uniform_init_range, b=config.uniform_init_range)
            nn.init.uniform_(getattr(self.lstm_layer, f'bias_ih_l{i}'), a=-config.uniform_init_range, b=config.uniform_init_range)
            nn.init.uniform_(getattr(self.lstm_layer, f'bias_hh_l{i}'), a=-config.uniform_init_range, b=config.uniform_init_range)

class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, dropout):
        super(Decoder, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, hidden_size, padding_idx=config.PAD)
        self.dropout = nn.Dropout(p=dropout)
        self.lstm_layer = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, vocab_size, bias=False)
        
    def forward(self, encoder_outputs, h_0, c_0, target = None):
        """
        h_0            (num of layers, N, H)
        c_0            (num of layers, N, H)
        """
        decoder_input = target[:,0].unsqueeze(1) #N, L -> N, 1
        decoder_hidden = h_0
        decoder_cell = c_0
        decoder_outputs = []
        
        for time_step in range(1, config.MAX_LENGTH+2): # total processing time -> 51 // total tgt time step -> 52
            decoder_output, decoder_hidden, decoder_cell = self.forward_step(input=decoder_input, hidden=decoder_hidden, cell=decoder_cell)
            decoder_outputs.append(decoder_output) #decoder_output -> (N, 1, V)
            decoder_input = target[:,time_step].unsqueeze(1) #N, 1
        
        decoder_outputs = torch.cat(decoder_outputs, dim=1) #decoder_outputs -> (L, N, 1, V) => (N, L, V) L = 51
        return decoder_outputs, decoder_hidden, decoder_cell
    
    def forward_step(self, input, hidden, cell):
        """
        input  (N, 1)
        hidden (num of layers, H)
        cell   (num of layers, H)
        """
        emb = self.embedding_layer(input) #N, 1, H
        emb = self.dropout(emb)
        output, (h, c) = self.lstm_layer(emb, (hidden, cell))
        output = self.output_layer(output) #N, 1, V
        return output, h, c
    
    def initialization(self):
        nn.init.uniform_(self.embedding_layer.weight, a=-config.uniform_init_range, b=config.uniform_init_range)
        for i in range(config.num_layers):
            nn.init.uniform_(getattr(self.lstm_layer, f'weight_ih_l{i}'), a=-config.uniform_init_range, b=config.uniform_init_range)
            nn.init.uniform_(getattr(self.lstm_layer, f'weight_hh_l{i}'), a=-config.uniform_init_range, b=config.uniform_init_range)
            nn.init.uniform_(getattr(self.lstm_layer, f'bias_ih_l{i}'), a=-config.uniform_init_range, b=config.uniform_init_range)
            nn.init.uniform_(getattr(self.lstm_layer, f'bias_hh_l{i}'), a=-config.uniform_init_range, b=config.uniform_init_range)
        nn.init.uniform_(self.output_layer.weight, a=-config.uniform_init_range, b=config.uniform_init_range)


#####################################################################################################################################
######################################################### GLOBAL ATTENTION  #########################################################
#####################################################################################################################################

class GlobalAttention(nn.Module):
    def __init__(self, hidden_size, align_type):
        super(GlobalAttention, self).__init__()
        self.align_layer = getattr(Score, align_type)(hidden_size)
        self.W_c = nn.Linear(2*hidden_size, hidden_size, bias=False)
        self.tanh = nn.Tanh()
    
    def forward(self, encoder_outputs, output):
        align_score = self.align_layer(encoder_outputs, output) #N, 1, L
        context_vector = torch.bmm(align_score, encoder_outputs)  #N, 1, L * N, L, H -> N, 1, H
        output = self.tanh(self.W_c(torch.cat((context_vector, output), dim=2))) #N, 1, 2H -> N, 1, H
        return output
        
    def initialization(self):
        self.align_layer.initialization()
        nn.init.uniform_(self.W_c.weight, a=-config.uniform_init_range, b=config.uniform_init_range)

class GlobalAttentionDecoder(nn.Module):
    def __init__(self, align_type, input_feeding, vocab_size, hidden_size, num_layers, dropout):
        super(GlobalAttentionDecoder, self).__init__()
        self.input_feeding = input_feeding
        self.hidden_size = hidden_size
        
        self.embedding_layer = nn.Embedding(vocab_size, hidden_size, padding_idx=config.PAD)
        self.dropout = nn.Dropout(p=dropout)
        if input_feeding:
            self.lstm_layer = nn.LSTM(2*hidden_size, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        else:
            self.lstm_layer = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.attention_layer = GlobalAttention(hidden_size, align_type)
        self.output_layer = nn.Linear(hidden_size, vocab_size, bias=False)
        
    def forward(self, encoder_outputs, h_0, c_0, target = None):
        """
        h_0            (num of layers, N, H)
        c_0            (num of layers, N, H)
        """
        # mask_info = encoder_outputs.mask_info
        # encoder_outputs = encoder_outputs.masked_fill(encoder_outputs.mask_info.unsqueeze(2), -float('inf'))
        # encoder_outputs.mask_info = mask_info
        N, L, H = encoder_outputs.shape
        decoder_input = target[:,0].unsqueeze(1) #N, L -> N, 1
        decoder_hidden = h_0
        decoder_cell = c_0
        decoder_outputs = []
        attn_vec = torch.zeros(N, 1, H).to(config.device) #N, 1, H
        
        for time_step in range(1, config.MAX_LENGTH+2): # total processing time -> 51 // total tgt time step -> 52
            attn_vec, decoder_output, decoder_hidden, decoder_cell = self.forward_step(attn_vec, encoder_outputs, decoder_input, decoder_hidden, decoder_cell)
            decoder_outputs.append(decoder_output) #decoder_output -> (N, 1, V)
            decoder_input = target[:,time_step].unsqueeze(1) #N, 1
        
        decoder_outputs = torch.cat(decoder_outputs, dim=1) #decoder_outputs -> (L, N, 1, V) => (N, L, V) L = 51
        return decoder_outputs, decoder_hidden, decoder_cell
    
    def forward_step(self, attn_vec, encoder_outputs, input, hidden, cell):
        """
        encoder_outputs (N, L, H)
        input           (N, 1)
        hidden          (num of layers, H)
        cell            (num of layers, H)
        """
        emb = self.embedding_layer(input) #N, 1, H
        # print(emb.shape, attn_vec.shape)
        if self.input_feeding:
            emb = torch.cat((emb, attn_vec), dim=2) #N, 1, 2H
        emb = self.dropout(emb)
        output, (h, c) = self.lstm_layer(emb, (hidden, cell))
        attn_vec = self.attention_layer(encoder_outputs, output)
        output = self.output_layer(attn_vec) #N, 1, H -> N, 1, V
        return attn_vec, output, h, c
    
    def initialization(self):
        nn.init.uniform_(self.embedding_layer.weight, a=-config.uniform_init_range, b=config.uniform_init_range)
        for i in range(config.num_layers):
            nn.init.uniform_(getattr(self.lstm_layer, f'weight_ih_l{i}'), a=-config.uniform_init_range, b=config.uniform_init_range)
            nn.init.uniform_(getattr(self.lstm_layer, f'weight_hh_l{i}'), a=-config.uniform_init_range, b=config.uniform_init_range)
            nn.init.uniform_(getattr(self.lstm_layer, f'bias_ih_l{i}'), a=-config.uniform_init_range, b=config.uniform_init_range)
            nn.init.uniform_(getattr(self.lstm_layer, f'bias_hh_l{i}'), a=-config.uniform_init_range, b=config.uniform_init_range)
        self.attention_layer.initialization()
        nn.init.uniform_(self.output_layer.weight, a=-config.uniform_init_range, b=config.uniform_init_range)

####################################################################################################################################
######################################################### LOCAL ATTENTION  #########################################################
####################################################################################################################################

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
        position = torch.mul(config, sig_out)
        return position
    
    def initialization(self):
        nn.init.uniform_(self.W_p.weight, a=-config.uniform_init_range, b=config.uniform_init_range)
        nn.init.uniform_(self.v_p.weight, a=-config.uniform_init_range, b=config.uniform_init_range)

class LocalAttention(nn.Module):
    def __init__(self, hidden_size, align_type, predictive):
        super(LocalAttention, self).__init__()
        self.align_layer = getattr(Score, align_type)(hidden_size)
        self.W_c = nn.Linear(2*hidden_size, hidden_size, bias=False)
        if predictive:
            self.predictive = predictive
            self.position_layer = PositionRegression(hidden_size)
        self.tanh = nn.Tanh()
        self.dev_pow = config.dev_pow
    
    def forward(self, encoder_outputs, output, time_step, src_len):
        align_score = self.align_layer(encoder_outputs, output) #N, 1, L
        N, _, L = encoder_outputs.shape
        if self.predictive: # eq 10
            p_t = self.position_layer(output, L)
        else:
            p_t = torch.max(src_len - time_step, torch.fill(torch.empty_like(src_len),-1))
        gaussian_distribution = self.gaussian(p_t, N, L) #eq 11 // N, 1, L
        align_score = torch.mul(align_score, gaussian_distribution)
        context_vector = torch.bmm(align_score, encoder_outputs)  #N, 1, L * N, L, H -> N, 1, H
        output = self.tanh(self.W_c(torch.cat((context_vector, output), dim=2))) #N, 1, 2H -> N, 1, H
        return output
    
    def gaussian(self, time_step, N, L):
        length_vec = torch.arange(0,L).repeat(N,1,1).to(config.device) # eq 11: s // N, 1, L
        pow_sub = torch.pow(torch.sub(length_vec, time_step), 2) #time_step: real number
        div = torch.mul(-1, torch.div(pow_sub, self.dev_pow))
        output = torch.exp(div)
        return output
    
    def initialization(self):
        self.align_layer.initialization()
        nn.init.uniform_(self.W_c.weight, a=-config.uniform_init_range, b=config.uniform_init_range)
        if self.predictive:
            self.position_layer.initialization()
        
class LocalAttentionDecoder(nn.Module):
    def __init__(self, align_type, input_feeding, predictive, vocab_size, hidden_size, num_layers, dropout):
        super(LocalAttentionDecoder, self).__init__()
        self.input_feeding = input_feeding
        self.hidden_size = hidden_size
        
        self.embedding_layer = nn.Embedding(vocab_size, hidden_size, padding_idx=config.PAD)
        self.dropout = nn.Dropout(p=dropout)
        if input_feeding:
            self.lstm_layer = nn.LSTM(2*hidden_size, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        else:
            self.lstm_layer = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.attention_layer = LocalAttention(hidden_size, align_type, predictive)
        self.output_layer = nn.Linear(hidden_size, vocab_size, bias=False)
        
    def forward(self, src_len, encoder_outputs, h_0, c_0, target = None):
        """
        h_0            (num of layers, N, H)
        c_0            (num of layers, N, H)
        """
        N, L, H = encoder_outputs.shape
        decoder_input = target[:,0].unsqueeze(1) #N, L -> N, 1
        decoder_hidden = h_0
        decoder_cell = c_0
        decoder_outputs = []
        attn_vec = torch.zeros(N, 1, H).to(config.device) #N, 1, H
        
        for time_step in range(1, config.MAX_LENGTH+2): #total processing time -> 51 // total tgt time step -> 52
            attn_vec, decoder_output, decoder_hidden, decoder_cell = self.forward_step(src_len, time_step, attn_vec, encoder_outputs, decoder_input, decoder_hidden, decoder_cell)
            decoder_outputs.append(decoder_output) #decoder_output -> (N, 1, V)
            decoder_input = target[:,time_step].unsqueeze(1) #N, 1
        
        decoder_outputs = torch.cat(decoder_outputs, dim=1) #decoder_outputs -> (L, N, 1, V) => (N, L, V) L = 51
        return decoder_outputs, decoder_hidden, decoder_cell
    
    def forward_step(self, src_len, time_step, attn_vec, encoder_outputs, input, hidden, cell):
        """
        encoder_outputs (N, L, H)
        input           (N, 1)
        hidden          (num of layers, H)
        cell            (num of layers, H)
        """
        emb = self.embedding_layer(input) #N, 1, H
        if self.input_feeding:
            emb = torch.cat((emb, attn_vec), dim=2) #N, 1, 2H
        emb = self.dropout(emb)
        output, (h, c) = self.lstm_layer(emb, (hidden, cell))
        attn_vec = self.attention_layer(encoder_outputs, output, time_step, src_len)
        output = self.output_layer(attn_vec) #N, 1, H -> N, 1, V
        return attn_vec, output, h, c
    
    def initialization(self):
        nn.init.uniform_(self.embedding_layer.weight, a=-config.uniform_init_range, b=config.uniform_init_range)
        for i in range(config.num_layers):
            nn.init.uniform_(getattr(self.lstm_layer, f'weight_ih_l{i}'), a=-config.uniform_init_range, b=config.uniform_init_range)
            nn.init.uniform_(getattr(self.lstm_layer, f'weight_hh_l{i}'), a=-config.uniform_init_range, b=config.uniform_init_range)
            nn.init.uniform_(getattr(self.lstm_layer, f'bias_ih_l{i}'), a=-config.uniform_init_range, b=config.uniform_init_range)
            nn.init.uniform_(getattr(self.lstm_layer, f'bias_hh_l{i}'), a=-config.uniform_init_range, b=config.uniform_init_range)
        self.attention_layer.initialization()
        nn.init.uniform_(self.output_layer.weight, a=-config.uniform_init_range, b=config.uniform_init_range)