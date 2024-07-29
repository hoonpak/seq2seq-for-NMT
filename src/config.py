PAD = 0 #<pad>
UNK = 1 #<unk>
SOS = 2 #<s>
EOS = 3 #</s>

MAX_LENGTH = 50 # sos + 50 + eos
WORD_FREQUENCY = 50000

# device = "cuda:0"

num_layers = 4
batch_size = 128
uniform_init_range = 0.1
dimension = 1000
max_epoch = 10
lr_update_point = 5
start_lr = 1
normalized_gradient = 5
dropout_rate = 0.2
window_size = 10
dev_pow = 50
#1 epoch에 257984 iter