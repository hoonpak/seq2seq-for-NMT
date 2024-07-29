import argparse
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import PrepareData, CustomDataset
from model import Seq2Seq
from test import TestRNN
import config

parser = argparse.ArgumentParser()
parser.add_argument("--reverse", action=argparse.BooleanOptionalAction, help='reverse or not') # --reverse True, --no-reverse False
parser.add_argument("--dropout", action=argparse.BooleanOptionalAction, help='dropout or not') # --dropout True, --no-dropout False
parser.add_argument("--input_feeding", action=argparse.BooleanOptionalAction, help='input feeding or not')
parser.add_argument("--attn", choices=['global', 'local_m', 'local_p', 'no'])
parser.add_argument("--align", choices=['dot', 'general', 'concat', 'location', 'no'])
parser.add_argument("--device")
option = parser.parse_args()

name = "np_v2_base"

if option.reverse:
    name += "reverse"
    print("Reverse ready")

dropout = 0
if option.dropout:
    name += "dropout"
    dropout = config.dropout_rate
    config.max_epoch = 12
    config.lr_update_point = 8
    print(f"{dropout} - Dropout ready")

if option.input_feeding:
    name += option.input_feeding
    
if option.attn == 'no':
    pass
else:
    name += option.attn
    
if option.align == 'no':
    pass
else:
    name += option.align
    
print(f"System:{name} is ready!!")

device = option.device

training_src_path = "../dataset/training/np_training_en.txt"
training_tgt_path = "../dataset/training/np_training_de.txt"
# test_src_path = "../dataset/test/test_en_2014.txt"
# test_tgt_path = "../dataset/test/test_de_2014.txt"
test_src_path = "../dataset/test/test_cost_en.txt"
test_tgt_path = "../dataset/test/test_cost_de.txt"

train_data = PrepareData(src_path = training_src_path, tgt_path = training_tgt_path, is_train = True)
test_data = PrepareData(src_path = test_src_path, tgt_path = test_tgt_path, is_train = False)

test_ins= TestRNN(test_data.filtered_src, test_data.filtered_tgt, train_data.src_word2id, train_data.tgt_word2id, option.reverse)

train_dataset = CustomDataset(src = train_data.filtered_src, tgt = train_data.filtered_tgt, 
                            src_word2id = train_data.src_word2id, tgt_word2id = train_data.tgt_word2id,
                            is_reverse = option.reverse)
test_dataset = CustomDataset(src = test_data.filtered_src, tgt = test_data.filtered_tgt, 
                            src_word2id = train_data.src_word2id, tgt_word2id = train_data.tgt_word2id,
                            is_reverse = option.reverse)

train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

src_vocab_size = len(train_data.src_word2id)
tgt_vocab_size = len(train_data.tgt_word2id)

model = Seq2Seq(attn = option.attn, align = option.align, input_feeding = option.input_feeding, 
                src_vocab_size = src_vocab_size, tgt_vocab_size = tgt_vocab_size, hidden_size = config.dimension, 
                num_layers = config.num_layers, dropout = dropout).to(device)

model.initialization()
with torch.no_grad():
    model.encoder.embedding_layer.weight[config.PAD] = torch.zeros(config.dimension).to(device)
    model.decoder.embedding_layer.weight[config.PAD] = torch.zeros(config.dimension).to(device)

loss_function = nn.CrossEntropyLoss(ignore_index = config.PAD).to(device)
parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.SGD(parameters, lr=config.start_lr)

writer = SummaryWriter(log_dir=f"./runs/{name}")
st = time.time()
train_loss = 0
train_ppl = 0
print("Start training!!")
iter = 0
for epoch in range(config.max_epoch):
    for src, src_len, tgt in train_dataloader:
        # breakpoint()
        src = src.to(device)
        # src_len = src_len.to(device)
        tgt = tgt.to(device)
        
        optimizer.zero_grad()
        predict = model.forward(src, src_len, tgt)
        loss = loss_function(predict, tgt[:,1:])
        # loss = loss_function(predict, tgt[:,1:].reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(parameters, max_norm=config.normalized_gradient)
        optimizer.step()
        
        train_loss += loss.detach().cpu().item()
        train_ppl += torch.exp(loss.detach()).cpu().item()
        
        if iter % 100 == 0:
            train_loss /= 100
            train_ppl /= 100
            print(f"Step: {epoch}-{iter:<10} Train Loss: {train_loss:<10.4f} Train PPL: {train_ppl:<10.2f} Time:{(time.time()-st)/3600:>6.4f} Hour")
            writer.add_scalars('loss', {'train_loss':train_loss}, iter)
            writer.add_scalars('ppl', {'train_ppl':train_ppl}, iter)
            writer.flush()
            train_loss = 0
            train_ppl = 0
            
        if iter % 5000 == 0:
            test_cost = 0
            test_ppl = 0
            num = 0
            model.eval()
            with torch.no_grad():
                for src, src_len, tgt in test_dataloader:
                    src = src.to(device)
                    tgt = tgt.to(device)
                    predict = model.forward(src, src_len, tgt)
                    loss = loss_function(predict, tgt[:,1:])
                    # loss = loss_function(predict, tgt[:,1:].reshape(-1))
                    test_cost += loss.detach().cpu().item()
                    test_ppl += torch.exp(loss.detach()).cpu().item()
                    num += 1
            test_cost /= num
            test_ppl /= num
            print('='*10, f"Step: {epoch}-{iter:<10} Test Loss: {test_cost:<10.4f} Test PPL: {test_ppl:<10.2f} Time:{(time.time()-st)/3600:>6.4f} Hour", '='*10)
            writer.add_scalars('cost', {'test_cost':test_cost}, iter)
            writer.add_scalars('ppl', {'test_ppl':test_ppl}, iter)
            writer.flush()
            model.train()
        iter += 1
    if (epoch+1) >= config.lr_update_point:
        optimizer.param_groups[0]['lr'] *= 0.5
    
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iter': iter,
                }, f"./save_model/{name}_CheckPoint.pth")
    
    # print("Start greedy search!!")
    # greedy_predict = test_ins.greedy_search(model, device)
    # greedy_bleu_score = test_ins.bleu_score(greedy_predict)
    print("Start beam search!!")
    beam_predict = test_ins.beam_search(model, device, beam_size=3)
    beam_bleu_score = test_ins.bleu_score(beam_predict)
    print(' '.join(list(map(lambda x:train_data.tgt_id2word[x], beam_predict[0]))))
    print(' '.join(list(map(lambda x:train_data.tgt_id2word[x], test_ins.tgt[0][1:-1]))))
    print("="*50)
    print(' '.join(list(map(lambda x:train_data.tgt_id2word[x], beam_predict[-1]))))
    print(' '.join(list(map(lambda x:train_data.tgt_id2word[x], test_ins.tgt[-1][1:-1]))))
    print("="*50)
    print(f"{name} beam bleu score : {beam_bleu_score:.2f}")
    # print(f"{name} greedy bleu score : {greedy_bleu_score:.2f}")
    test_ins.perplexity(model, device)
    # test_ins.perplexity_(model, device)
    model.train()
