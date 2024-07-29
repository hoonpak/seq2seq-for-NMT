import torch
from torch import nn

import config
import layer

# class Seq2Seq(nn.Module):
#     def __init__(self, src_vocab_size, tgt_vocab_size, hidden_size, num_layers, dropout):
#         super(Seq2Seq, self).__init__()
#         self.encoder = layer.Encoder(src_vocab_size, hidden_size, num_layers, dropout)
#         self.decoder = layer.Decoder(tgt_vocab_size, hidden_size, num_layers, dropout)
#         self.tgt_vocab_size = tgt_vocab_size
    
#     def forward(self, src, src_len, tgt = None):
#         encoder_output, decoder_h_0, decoder_c_0 = self.encoder(src, src_len)
#         decoder_output, _, _ = self.decoder(decoder_h_0, decoder_c_0, tgt) #decoder_output = N, L, V // L = 51
#         # decoder_output = decoder_output.permute(0,2,1) #decoder output = N, V, L
#         decoder_output = decoder_output.reshape(-1, self.tgt_vocab_size) #decoder output = N*L, V
#         return decoder_output
    
#     def initialization(self):
#         self.encoder.initialization()
#         self.decoder.initialization()

class Seq2Seq(nn.Module):
    def __init__(self, attn, align, input_feeding, src_vocab_size, tgt_vocab_size, hidden_size, num_layers, dropout):
        super(Seq2Seq, self).__init__()
        self.encoder = layer.Encoder(src_vocab_size, hidden_size, num_layers, dropout)
        if attn == 'global': # align_type, input_feeding, vocab_size, hidden_size, num_layers, dropout
            self.decoder = layer.GlobalAttentionDecoder(align, input_feeding, tgt_vocab_size, hidden_size, num_layers, dropout)
        elif attn == 'local_m': # align_type, input_feeding, predictive, vocab_size, hidden_size, num_layers, dropout
            predictive = False
            self.decoder = layer.LocalAttentionDecoder(align, input_feeding, predictive, tgt_vocab_size, hidden_size, num_layers, dropout)
        elif attn == 'local_p': # align_type, input_feeding, predictive, vocab_size, hidden_size, num_layers, dropout
            predictive = True
            self.decoder = layer.LocalAttentionDecoder(align, input_feeding, predictive, tgt_vocab_size, hidden_size, num_layers, dropout)
        else:
            self.decoder = layer.Decoder(tgt_vocab_size, hidden_size, num_layers, dropout)
        self.tgt_vocab_size = tgt_vocab_size
        
    def forward(self, src, src_len, tgt = None):
        encoder_output, decoder_h_0, decoder_c_0 = self.encoder(src, src_len)
        decoder_output, _, _ = self.decoder(decoder_h_0, decoder_c_0, tgt) #decoder_output = N, L, V // L = 51
        decoder_output = decoder_output.permute(0,2,1) #decoder output = N, V, L
        # decoder_output = decoder_output.reshape(-1, self.tgt_vocab_size) #decoder output = N*L, V
        return decoder_output
    
    def initialization(self):
        self.encoder.initialization()
        self.decoder.initialization()