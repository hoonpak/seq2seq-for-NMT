import config
import torch
from torch import nn

class Score:
    class dot(nn.Module):
        def __init__(self, hidden_size):
            super(Score.dot, self).__init__()
            
        def forward(self, decoder_h_t, encoder_outputs):
            """
            encoder_outputs  (N, L, H)
            decoder_h_t      (N, 1, H)
            """
            score4align = torch.bmm(decoder_h_t, encoder_outputs.permute(0,2,1)) #(N, 1, H) * (N, H, L) -> (N, 1, L)
            return score4align
        
        def initialization(self):
            pass
        
    class general(nn.Module):
        def __init__(self, hidden_size):
            super(Score.general, self).__init__()
            self.W_a = nn.Linear(hidden_size, hidden_size, bias=False)
            
        def forward(self, decoder_h_t, encoder_outputs):
            """
            encoder_outputs  (N, L, H)
            decoder_h_t      (N, 1, H)
            """
            decoder_h_t = self.W_a(decoder_h_t) #N, L, H
            score4align = torch.bmm(decoder_h_t, encoder_outputs.permute(0,2,1)) #(N, 1, H) * (N, H, L) -> (N, 1, L)
            return score4align
        
        def initialization(self):
            nn.init.uniform_(self.W_a.weight, a=-config.uniform_init_range, b=config.uniform_init_range)
        
    class concat(nn.Module):
        def __init__(self, hidden_size):
            super(Score.concat, self).__init__()
            self.W_a = nn.Linear(hidden_size, hidden_size, bias=False)
            self.v_a = nn.Linear(hidden_size, 1, bias=False)
            self.tanh = nn.Tanh()
            
        def forward(self, decoder_h_t, encoder_outputs):
            """
            encoder_outputs  (N, L, H)
            decoder_h_t      (N, 1, H)
            """
            src_ht_hid = self.tanh(encoder_outputs + self.W_a(decoder_h_t))#(N, L, H)
            score4align = self.v_a(src_ht_hid).permute(0,2,1) #N, 1, L
            return score4align
        
        def initialization(self):
            nn.init.uniform_(self.W_a.weight, a=-config.uniform_init_range, b=config.uniform_init_range)
            nn.init.uniform_(self.v_a.weight, a=-config.uniform_init_range, b=config.uniform_init_range)
        
    #1. align score <- softmax(output*W_a)
    #2. context vector <- align score*encoder outputs (weighted average)
    #3. output_ <- tanh(W_c*[context_vector;output]) # 2H -> H
        
    class location(nn.Module):
        def __init__(self, hidden_size):
            super(Score.location, self).__init__()
            self.W_a = nn.Linear(hidden_size, config.MAX_LENGTH+2, bias=False)
        
        def forward(self, decoder_h_t, encoder_outputs):
            """
            encoder_outputs  (N, L, H)
            decoder_h_t      (N, 1, H)
            """
            score4align = self.W_a(decoder_h_t) #N, 1, H -> N, 1, L
            return score4align
        
        def initialization(self):
            nn.init.uniform_(self.W_a.weight, a=-config.uniform_init_range, b=config.uniform_init_range)