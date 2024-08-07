import argparse
import time
import os

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader

import config
import numpy as np

from utils import PrepareData, get_tokenized_sen, CustomDataset
from model import Seq2Seq
from collections import Counter
from tqdm import tqdm

#1. load test file
#2. run beam search
#3. compute tokenized bleu score

class TestRNN:
    def __init__(self, filtered_test_src, filtered_test_tgt, train_src_word2id, train_tgt_word2id, is_reverse):
        self.src_word2id = train_src_word2id
        self.tgt_word2id = train_tgt_word2id
        self.src = list(map(lambda x:get_tokenized_sen(x,self.src_word2id,is_reverse), filtered_test_src)) #no padding
        self.tgt = list(map(lambda x:get_tokenized_sen(x,self.tgt_word2id,False), filtered_test_tgt))
        self.test_dataset = CustomDataset(src = filtered_test_src, tgt = filtered_test_tgt, 
                                        src_word2id = train_src_word2id, tgt_word2id = train_tgt_word2id,
                                        is_reverse = is_reverse)
    
    def greedy_search(self, model, device, boundary = None):
        model.eval()
        predict = []
        with torch.no_grad():
            for src_sen in tqdm(self.src[:boundary]):
                outputs = []
                
                encoder_input = torch.LongTensor(src_sen).reshape(1,-1).to(device) # L,
                encoder_input_len = [len(src_sen)]
                
                emb = model.encoder.embedding_layer(encoder_input) #batch size, max lenth, dimension
                emb = model.encoder.dropout(emb)
                packed_emb = pack_padded_sequence(input=emb, lengths=encoder_input_len, batch_first=True, enforce_sorted=False)
                # model.encoder.lstm_layer.flatten_parameters()
                packed_output, (h_n, c_n) = model.encoder.lstm_layer(packed_emb) #it's the result of to compute every step each layers.
                encoder_outputs, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=config.MAX_LENGTH+2)
                decoder_h_0 = torch.clamp(h_n, min=-config.clipForward, max=config.clipForward)
                decoder_c_0 = torch.clamp(c_n, min=-config.clipForward, max=config.clipForward)
                
                decoder_input = torch.LongTensor([2]).reshape(1,-1).to(device) # SOS Token -> 1
                attn_vec = torch.zeros(1,1,config.dimension).to(device)
                
                for time_step in range(config.MAX_LENGTH+1):
                    encoder_input_len = torch.Tensor(encoder_input_len)
                    
                    decoder_output, attn_vec, decoder_h, decoder_c = model.decoder.forward_step(src_len=encoder_input_len, encoder_outputs=encoder_outputs, attn_vec=attn_vec,
                                                                                                time_step=time_step, decoder_input=decoder_input, hidden=decoder_h, cell=decoder_c)
                    decoder_input = decoder_output.topk(1)[1].squeeze(-1).detach()
                    
                    hypo = decoder_input.item()
                    if hypo == config.EOS:
                        break
                    outputs.append(hypo)
                
                predict.append(outputs)
                
        return predict
    
    def length_penalty(self, sentence, alpha = 1.2):
        return ((5+len(sentence))**alpha)/((5+1)**alpha)
    
    def limit_length_repeat_penalty(self, src_length, score, predict_sen):
        min_length = 0.5*src_length
        max_length = 1.5*src_length
        if len(predict_sen) < min_length:
            return 100000
        if len(predict_sen) > max_length:
            return 100000
        if len(predict_sen) >= 5:
            predict_sen = [str(step.item()) for step in predict_sen]
            predict_ngram = Counter(self.get_ngram(4, predict_sen))
            for k, v in predict_ngram.items():
                if v >= 2:
                    return 100000
        return score
    
    def beam_search(self, model, device, beam_size, boundary = None):
        predict = []
        model.eval()
        with torch.no_grad():
            for src_sen in tqdm(self.src[:boundary]):
                stop_flag = False
                encoder_input = torch.LongTensor(src_sen).reshape(1,-1).to(device) # L,
                encoder_input_len = [len(src_sen)]
                
                emb = model.encoder.embedding_layer(encoder_input) #batch size, max lenth, dimension
                emb = model.encoder.dropout(emb)
                packed_emb = pack_padded_sequence(input=emb, lengths=encoder_input_len, batch_first=True, enforce_sorted=False)
                # model.encoder.lstm_layer.flatten_parameters()
                packed_output, (h_n, c_n) = model.encoder.lstm_layer(packed_emb) #it's the result of to compute every step each layers.
                encoder_outputs, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=config.MAX_LENGTH+2)
                decoder_h_0 = torch.clamp(h_n, min=-config.clipForward, max=config.clipForward)
                decoder_c_0 = torch.clamp(c_n, min=-config.clipForward, max=config.clipForward)
                
                decoder_input = torch.LongTensor([2]).reshape(1,-1).to(device) # SOS Token -> 1
                attn_vec = torch.zeros(1,1,config.dimension).to(device)
                
                beam = [(0.0, [decoder_input], decoder_h_0, decoder_c_0, attn_vec)]
                completed_sequences = []
                max_gen_length = round(len(src_sen)*1.5)
                for time_step in range(max_gen_length):
                    new_beam = []
                    for score, sequence, decoder_h, decoder_c, attn_vec in beam:
                        decoder_input = sequence[-1]
                        encoder_input_len = torch.Tensor(encoder_input_len)
                        
                        decoder_output, attn_vec, decoder_h, decoder_c = model.decoder.forward_step(src_len=encoder_input_len, encoder_outputs=encoder_outputs, attn_vec=attn_vec,
                                                                                                    time_step=time_step, decoder_input=decoder_input, hidden=decoder_h, cell=decoder_c)
                        probabilities, candidates = decoder_output.softmax(dim=-1).topk(beam_size)

                        for i in range(beam_size):
                            candidate = candidates.squeeze()[i].unsqueeze(0).unsqueeze(0) #
                            prob = probabilities.squeeze()[i]
                            new_sequence = sequence + [candidate]
                            new_score = (score - torch.log(prob + 1e-7)).item()
                            # new_score /= self.length_penalty(new_sequence)
                            
                            if candidate.item() == 3: #when search the eos token
                                completed_sequences.append((score, sequence, decoder_h, decoder_c, attn_vec))
                                if len(completed_sequences) >= beam_size:
                                    stop_flag = True
                                    break
                            else:
                                new_beam.append((new_score, new_sequence, decoder_h, decoder_c, attn_vec))
                            
                        if stop_flag:
                            break
                        
                    if stop_flag:
                        break
                    
                    beam = sorted(new_beam, key=lambda x:x[0])[:beam_size-len(completed_sequences)]
                    
                completed_sequences.extend(beam)
                completed_sequences = list(map(list, completed_sequences))
                for ind in range(len(completed_sequences)):
                    completed_sequences[ind][0] = self.limit_length_repeat_penalty(len(src_sen), completed_sequences[ind][0], completed_sequences[ind][1])                    
                completed_sequences = sorted(completed_sequences, key=lambda x: x[0]/(self.length_penalty(x[1])))[0]
                best_score, best_sequence, _, _, _ = completed_sequences
                best_sequence = [step.item() for step in best_sequence[1:]]
                predict.append(best_sequence)
                
        return predict
        
    def get_ngram(self, n, sentence):
        ngrams = []
        for i in range(len(sentence)-n+1):
            ngrams += [" ".join(sentence[i:i+n])]
        return ngrams
    
    def bleu_score(self, predict, boundary = None):
        total_bleu_score = 0
        for predict_sentence, tgt_sentence in zip(predict, self.tgt[:boundary]):
            predict_sentence = list(map(str,predict_sentence))
            tgt_sentence = list(map(str,tgt_sentence[1:-1]))
            n_bleu = dict()
            for n in range(1,5):
                correct = 0
                predict_ngram = Counter(self.get_ngram(n, predict_sentence)) #예측
                tgt_ngram = Counter(self.get_ngram(n, tgt_sentence)) #정답
                total = sum(predict_ngram.values()) 
                if total == 0:
                    n_bleu[n] = 0
                    continue
                for pdt_n, pdt_n_c in predict_ngram.items(): #예측
                    if pdt_n in tgt_ngram.keys(): #정답 안에 예측이 있다면
                        tgt_n_c = tgt_ngram[pdt_n] #정답의 개수
                        if tgt_n_c >= pdt_n_c:
                            correct += pdt_n_c #예측을 더함
                        else :
                            correct += tgt_n_c #정답이 더 작으면 정답을 더함
                n_bleu[n] = correct/total
            brevity_penalty = 1
            if len(tgt_sentence) > len(predict_sentence):
                brevity_penalty = np.exp(1 - (len(tgt_sentence)/max(len(predict_sentence),1)))
            bleu = brevity_penalty*np.exp(sum(np.log(max(bs, 1e-7)) for bs in n_bleu.values()) / 4)
            total_bleu_score += bleu
        total_bleu_score /= len(predict)
        total_bleu_score *= 100
        return total_bleu_score
    
    def perplexity(self, model, device):
        test_cost = 0
        test_ppl = 0
        num = 0
        model.eval()
        test_dataloader = DataLoader(self.test_dataset, config.batch_size, shuffle=False)
        loss_function = torch.nn.CrossEntropyLoss(ignore_index = config.PAD).to(device)
        with torch.no_grad():
            for src, src_len, tgt in tqdm(test_dataloader):
                src = src.to(device)
                tgt = tgt.to(device)
                predict = model.forward(src, src_len, tgt)
                loss = loss_function(predict, tgt[:,1:].reshape(-1))
                test_cost += loss.detach().cpu().item()
                test_ppl += torch.exp(loss.detach()).cpu().item()
                num += 1
        test_cost /= num
        test_ppl /= num
        print('='*10, f"Test Loss: {test_cost:<10.4f} Test PPL: {test_ppl:<10.2f}", '='*10)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reverse", action=argparse.BooleanOptionalAction, help='reverse or not') # --reverse True, --no-reverse False
    parser.add_argument("--dropout", action=argparse.BooleanOptionalAction, help='dropout or not') # --dropout True, --no-dropout False
    parser.add_argument("--input_feeding", action=argparse.BooleanOptionalAction, help='input feeding or not')
    parser.add_argument("--attn", choices=['global', 'local_m', 'local_p', 'no'])
    parser.add_argument("--align", choices=['dot', 'general', 'concat', 'location', 'no'])
    parser.add_argument("--name")
    parser.add_argument("--device")
    option = parser.parse_args()

    config.device = option.device
    device = option.device

    # name = "np_v2_base"
    name = option.name

    if option.reverse:
        name += "_reverse"
        print("Reverse ready")

    dropout = 0
    if option.dropout:
        name += "_dropout"
        dropout = config.dropout_rate
        config.max_epoch = 12
        config.lr_update_point = 8
        print(f"{dropout} - Dropout ready")

    if option.input_feeding:
        name += "_infeed"
        
    if option.attn == 'no':
        pass
    else:
        name += "_"+option.attn
        
    if option.align == 'no':
        pass
    else:
        name += "_"+option.align
        
    print(f"System:{name} is ready!!")


    # training_src_path = "../dataset/training/new_training_en.txt"
    # training_tgt_path = "../dataset/training/new_training_de.txt"
    # test_src_path = "../dataset/test/new_test_cost_en.txt"
    # test_tgt_path = "../dataset/test/new_test_cost_de.txt"
    training_src_path = "../dataset/training/np_training_en.txt"
    training_tgt_path = "../dataset/training/np_training_de.txt"
    test_src_path = "../dataset/test/test_cost_en.txt"
    test_tgt_path = "../dataset/test/test_cost_de.txt"

    train_data = PrepareData(src_path = training_src_path, tgt_path = training_tgt_path, is_train = True)
    test_data = PrepareData(src_path = test_src_path, tgt_path = test_tgt_path, is_train = False)
    # with open(test_src_path, "r") as file:
    #     src_lines = file.readlines()
    # with open(test_tgt_path, "r") as file:
    #     tgt_lines = file.readlines()
        
    src_vocab_size = len(train_data.src_word2id)
    tgt_vocab_size = len(train_data.tgt_word2id)
    
    model_info = torch.load(f"./save_model/{name}_CheckPoint.pth", map_location=option.device)
    # model = model_info['model']
    model = Seq2Seq(attn_type = option.attn, align_type = option.align, input_feeding = option.input_feeding, 
                src_vocab_size = src_vocab_size, tgt_vocab_size = tgt_vocab_size, hidden_size = config.dimension, 
                num_layers = config.num_layers, dropout = dropout, is_reverse = option.reverse).to(device)
    model.load_state_dict(model_info['model_state_dict'])
    model.eval()
    
    test_ins= TestRNN(test_data.filtered_src, test_data.filtered_tgt, train_data.src_word2id, train_data.tgt_word2id, option.reverse)
    # print("Start greedy search!!")
    # greedy_predict = test_ins.greedy_search(model, option.device)
    # greedy_bleu_score = test_ins.bleu_score(greedy_predict)
    print("Start beam search!!")
    beam_predict = test_ins.beam_search(model, option.device, beam_size=12)
    beam_bleu_score = test_ins.bleu_score(beam_predict)
    print(' '.join(list(map(lambda x:train_data.tgt_id2word[x], beam_predict[0]))))
    print(' '.join(list(map(lambda x:train_data.tgt_id2word[x], test_ins.tgt[0][1:-1]))))
    print("="*50)
    print(' '.join(list(map(lambda x:train_data.tgt_id2word[x], beam_predict[-1]))))
    print(' '.join(list(map(lambda x:train_data.tgt_id2word[x], test_ins.tgt[-1][1:-1]))))
    print("="*50)
    print(f"{name} beam bleu score : {beam_bleu_score:.2f}")

    print("write ref/hyp file ... ")
    hyp_file_name = "./beam_predict"
    fhyp = open(hyp_file_name, "wb")
    for hyp in beam_predict:
        fhyp.write(' '.join(list(map(lambda x:train_data.tgt_id2word[x], hyp))).encode('utf-8'))
        fhyp.write("\n".encode('utf-8'))
    fhyp.close()

    target_file_name = "./target"
    ftarget = open(target_file_name, "wb")
    for target in test_ins.tgt:
        ftarget.write(' '.join(list(map(lambda x:train_data.tgt_id2word[x], target))).encode('utf-8'))
        ftarget.write("\n".encode('utf-8'))
    ftarget.close()
    
    os.system("./multi-bleu.perl -lc target < beam_predict")
    
    # print(f"{name} greedy bleu score : {greedy_bleu_score:.2f}")
    test_ins.perplexity(model, option.device)