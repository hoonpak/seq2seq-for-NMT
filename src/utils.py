import config

from collections import Counter
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

class PrepareData:
    def __init__(self, src_path, tgt_path, is_train, max_length = config.MAX_LENGTH):
        with open(src_path, "r") as file:
            src_lines = file.readlines()
        with open(tgt_path, "r") as file:
            tgt_lines = file.readlines()
        
        src_length_list = list(map(lambda x: len(x.split()), src_lines))
        tgt_length_list = list(map(lambda x: len(x.split()), tgt_lines))
        
        src_filter_index = self.get_filter_index(src_length_list, max_length)
        tgt_filter_index = self.get_filter_index(tgt_length_list, max_length)
        
        total_filter_index = set(src_filter_index + tgt_filter_index)
        
        self.filtered_src = [sen for i, sen in enumerate(src_lines) if i not in total_filter_index]
        self.filtered_tgt = [sen for i, sen in enumerate(tgt_lines) if i not in total_filter_index]
        
        del src_lines, tgt_lines, src_length_list, tgt_length_list, src_filter_index, tgt_filter_index, total_filter_index
        
        if is_train:
            print("Load src dictionaries...")
            self.src_word2id, self.src_id2word = self.get_dictionary(self.filtered_src)
            # self.src_word2id, self.src_id2word = self.get_dictionary(src_lines)
            print("Load tgt dictionaries...")
            self.tgt_word2id, self.tgt_id2word = self.get_dictionary(self.filtered_tgt)
            # self.tgt_word2id, self.tgt_id2word = self.get_dictionary(tgt_lines)
        
    def get_filter_index(self, length_list, max_length):
        filter_index = []
        for i, s in enumerate(length_list):
            if s > max_length:
                filter_index.append(i)
        return filter_index
    
    def get_dictionary(self, lines):
        word2id = {'<pad>':0, '<unk>':1, '<s>':2, '</s>':3}
        id2word = {0:'<pad>', 1:'<unk>', 2:'<s>', 3:'</s>'}
        
        counter = Counter()
        
        for sen in tqdm(lines):
            for token in sen.split():
                token = token.lower() #
                counter[token] += 1

        print(counter.total())
        
        id = 4
        for word, _ in counter.most_common(config.WORD_FREQUENCY):
            word2id[word] = id
            id2word[id] = word
            id += 1
        return word2id, id2word
    
def get_tokenized_sen(sen, word2id, is_reverse):
    
    tokenized_sen = [2] #<s>
        
    for token in sen.split():
        token = token.lower() #
        if token in word2id:
            tokenized_sen.append(word2id[token])
        else:
            tokenized_sen.append(1) #<unk>
    tokenized_sen.append(3) #</s>
    
    if is_reverse:
        tokenized_sen = list(reversed(tokenized_sen))
            
    return tokenized_sen

class CustomDataset(Dataset):
    def __init__(self, src, tgt, src_word2id, tgt_word2id, is_reverse):
        self.src = src
        self.tgt = tgt
        self.src_word2id = src_word2id
        self.tgt_word2id = tgt_word2id
        self.length = len(self.src)
        self.is_reverse = is_reverse
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        tokenized_src = get_tokenized_sen(self.src[index], self.src_word2id, self.is_reverse) # maximum leng -> 52
        tokenized_tgt = get_tokenized_sen(self.tgt[index], self.tgt_word2id, False)  # maximum leng -> 52
        
        src_length = len(tokenized_src)
        tokenized_src = tokenized_src + [0]*(config.MAX_LENGTH - src_length + 2) # leng -> 52
        tokenized_tgt = tokenized_tgt + [0]*(config.MAX_LENGTH - len(tokenized_tgt) + 2) # leng -> 52
        
        return [torch.LongTensor(tokenized_src), src_length, torch.LongTensor(tokenized_tgt)]