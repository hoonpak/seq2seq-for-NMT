# import torch
from torch import nn

# import config
import module

class Seq2Seq(nn.Module):
    def __init__(self, attn_type, align_type, input_feeding, src_vocab_size, tgt_vocab_size, hidden_size, num_layers, dropout, is_reverse):
        """
        Args:
            attn           (str)    : Type of attention is global, local_m, local_p or no.
            align          (str)    : Type of align score fuction is general, dot, concat or location.
            input_feeding  (boolean): Choose applying input feeding method or not.
            src_vocab_size (int)    : Vocab size of the source training data.
            tgt_vocab_size (int)    : Vocab size of the target training data.
            hidden_size    (int)    : Dimension of the hidden layer which is one of embedding, lstm, output...
            num_layers     (int)    : Number of lstm layers.
            dropout        (float)  : Dropout rate of embedding dropout, lstm dropout.
        """
        super(Seq2Seq, self).__init__()
        self.encoder = module.Encoder(src_vocab_size, hidden_size, num_layers, dropout)
        self.decoder = module.Decoder(attn_type, align_type, input_feeding, tgt_vocab_size, hidden_size, num_layers, dropout, is_reverse)
        self.tgt_vocab_size = tgt_vocab_size
        
        #Apply uniform initialization to all layers of the model
        self.initialization()
        
    def forward(self, src, src_len, tgt = None):
        """
        Args:
            src     (torch.Tensor): _description_
            src_len (torch.Tensor): _description_
            tgt     (torch.Tensor, None): The data for teacher forcing. Defaults to None for generate text.

        Returns:
            _type_: _description_
        """
        encoder_outputs, decoder_h_0, decoder_c_0 = self.encoder(src, src_len)
        decoder_output, _, _ = self.decoder(src_len, encoder_outputs, decoder_h_0, decoder_c_0, tgt) #decoder_output = N, L, V // L = 51
        # decoder_output = decoder_output.permute(0,2,1) #decoder output = N, V, L
        decoder_output = decoder_output.reshape(-1, self.tgt_vocab_size) #decoder output = N*L, V
        return decoder_output
    
    def initialization(self):
        self.encoder.initialization()
        self.decoder.initialization()