import torch
from torch import nn

import config
from score import Score

class Attention(nn.Module):
    def __init__(self, hidden_size, attn_type, align_type):
        super(Attention, self).__init__()
        self.score_layer = getattr(Score, align_type)(hidden_size)
        self.W_c = nn.Linear(2*hidden_size, hidden_size, bias=False)
        self.tanh = nn.Tanh()
        
        self.attn_type = attn_type
        self.align_type = align_type
        self.max_length = config.MAX_LENGTH + 2
        self.index_matrix = torch.arange(0,self.max_length).to(config.device)
        self.dev_pow = config.dev_pow
        
    def forward(self, encoder_outputs, decoder_h_t, src_len, p_t = None):
        # Masking process
        N, L, H = encoder_outputs.shape
        src_start = 0
        src_end = src_len.to(config.device)
        if p_t != None: # local attention
            p_t = p_t.to(config.device)
            attn_start = p_t - config.window_size
            attn_end = p_t + config.window_size
            exc_start_index = (attn_start < src_start) # if under 0
            exc_end_index = (attn_end > src_end) # if upper src_len
            attn_start[exc_start_index] = src_start # if under 0, change it to 0
            if self.attn_type == 'local_p':
                attn_end[exc_end_index] = src_end[exc_end_index].to(torch.float32) # if upper src_len, change it to sentence length
            else:
                attn_end[exc_end_index] = src_end[exc_end_index]
            src_start = attn_start
            src_end = attn_end
        length_vec = self.index_matrix.repeat(N,1) # N, L
        if self.attn_type.startswith("local"):
            mask_info_start = (length_vec < src_start.unsqueeze(-1))
        else:
            mask_info_start = (length_vec < src_start)
        mask_info_end = (length_vec > src_end.unsqueeze(-1))
        mask_info = (mask_info_start | mask_info_end) # N, L
        
        # Compute align score and make context vector
        score4align = self.score_layer(decoder_h_t, encoder_outputs)
        align_score = score4align.masked_fill(mask_info.unsqueeze(1), -float('inf')).softmax(dim=2) #N, 1, L
        if self.attn_type.startswith("local"):
            gaussian_distribution = self.gaussian(p_t, N) #eq 11 // N, L
            align_score = torch.mul(align_score, gaussian_distribution.unsqueeze(1))
        context_vector = torch.bmm(align_score, encoder_outputs) # (N, 1, L)*(N, L, H) => N, 1, H
        attention_output = self.tanh(self.W_c(torch.cat((context_vector, decoder_h_t), dim=2)))
        
        return attention_output
        
    def gaussian(self, time_steps, N):
        length_vec = self.index_matrix.repeat(N, 1) # eq 11: s // N, L
        if time_steps.dim() == 1:
            pow_sub = torch.pow(torch.sub(length_vec, time_steps.unsqueeze(-1)), 2) #time_step: real number # before modified -> if shape of time step is (N, )
        elif time_steps.dim() == 2:
            pow_sub = torch.pow((length_vec - time_steps), 2) #time_step: real numberb -> if shape of time step is (N, 1)
        else:
            raise Exception('check your time steps dimensions.')
        div = torch.mul(-1, torch.div(pow_sub, self.dev_pow))
        output = torch.exp(div)
        return output # N,L
    
    def initialization(self):
        self.score_layer.initialization()
        nn.init.uniform_(self.W_c.weight, a=-config.uniform_init_range, b=config.uniform_init_range)
