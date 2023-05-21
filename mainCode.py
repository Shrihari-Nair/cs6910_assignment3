#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch.optim as optim
from torch.autograd import Variable
get_ipython().system('pip install wandb')
import wandb
import torch.nn.functional as F
import numpy as np
import csv
device_gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import re

plt.style.use("dark_background")
from IPython.display import clear_output
import random
OUTPUT_CHAR_UPPER_LIMIT = 30

# In[ ]:

# class for Data loading( also for reading , shuffling and batching)
class Data_loading_class(Dataset):
    def __init__(self, f):
        zer = 0
        self.words_in_english, self.words_in_hindi = self.return_the_words(f, Filter_Vocab_HINDI)
        length_of_words_in_english = len(self.words_in_english)
        length_of_words_in_hindi = len(self.words_in_hindi)
        self.index_shuffle = list(range(length_of_words_in_english))
        print("indices shuffled")
        self.for_shuffling()
        self.ssi = zer


    def for_shuffling(self):
        random.shuffle(self.index_shuffle)
    
    def reading(self, f, VOCAB_FILTERING):
        WORDS_ENGLISH = []
        WORDS= []
        WORDS_HINDI = []
        dash = ' - '
        zer = 0
        one = 1 
        
        with open(f, 'r') as file_csv:
            reader_csv = csv.reader(file_csv)                                            # reading the file
            for x in reader_csv:
                temp_eng = x[zer]
                LIST_OF_WORDS_IN_ENG = Filter_Vocab_ENG(temp_eng)                        # filtering out non english letters
                length_of_word_list_english = len(LIST_OF_WORDS_IN_ENG)
                temp_hin = x[one]
                LIST_OF_WORDS_IN_HINDI = VOCAB_FILTERING(temp_hin)                       # filtering out non hindi letters
                length_of_word_list_hindi = len(LIST_OF_WORDS_IN_HINDI)

                if length_of_word_list_english != length_of_word_list_hindi:              # skip unwanted data
                    continue

                for ENG_W in LIST_OF_WORDS_IN_ENG:
                    a = ENG_W
                    #print(a)
                    WORDS_ENGLISH.append(ENG_W)
                for HIN_W in LIST_OF_WORDS_IN_HINDI:
                    b = HIN_W
                    #print(b)
                    WORDS_HINDI.append(HIN_W)
        
        final_english_words = WORDS_ENGLISH
        final_hindi_words = WORDS_HINDI
        WORDS_HINDI = WORDS # 
        #print(WORDS_HINDI)


        return final_english_words, final_hindi_words
    
    def return_the_words(self,f, Filter_Vocab_HINDI):
        e,h = self.reading(f, Filter_Vocab_HINDI)
        return e,h

    def batching(self, size_, list_):
        arr_ = []
        last = self.ssi + size_
        len_of_words_eng = len(self.words_in_english)
        arr = arr_
        zer = 0

        if last >= len_of_words_eng:
            arr_ = [list_[k] for k in self.index_shuffle[zer:last%len_of_words_eng]]                  
            arr = arr_
            last = len_of_words_eng
        return_object = arr + [list_[k] for k in self.index_shuffle[self.ssi : last]]
        return return_object
    
    def extract_batch_for_lang(self, size_, postprocess = True):
        one = 1
        zer = 0
        self.ssi += size_ + one
        store_len = len(self.words_in_english)
        batchHindi_ = self.batching(size_, self.words_in_hindi)
        length_of_words_in_eng_ = store_len
        batchEng_ = self.batching(size_, self.words_in_english)
        
        if self.ssi >= length_of_words_in_eng_:                                    # If one entire epoch is done then shuffle
            length_of_words_in_eng_ = len(self.words_in_english)
            random.shuffle(self.index_shuffle)
            self.ssi = zer 
            
        batchHindi_ = batchHindi_
        batchEng_ = batchEng_
        return batchEng_, batchHindi_



def return_len_of_word(w):
    return len(w)

def hindi_word_representation(index_of_the_letter,hindi_w, device = device_gpu):              # word representation for hindi words 
    zer = 0
    one = 1
    len_of_hindi_word = int(return_len_of_word(hindi_w))
    temp = len_of_hindi_word+one
    hindi_word_rep = torch.zeros([temp, one], dtype=torch.long).to(device)
    for ind, i in enumerate(hindi_w):
        pos = index_of_the_letter[i]
        hindi_word_rep[ind][zer] = pos
    hindi_word_rep[ind+one][zer] = index_of_the_letter[character_padding]
    return hindi_word_rep


def english_word_representation(index_of_the_letter,eng_w, device = device_gpu):             # word representation for english words 
    zer =0
    one =1
    len_of_eng_word = int(return_len_of_word(eng_w))
    temp = len_of_eng_word+one
    len_of_index_of_letter =  len(index_of_the_letter)
    eng_rep = torch.zeros(temp, one,len_of_index_of_letter).to(device)
    for ind, i in enumerate(eng_w):
        ptn = index_of_the_letter[i]
        eng_rep[ind][zer][ptn] = one
    eng_rep[ind+1][zer][index_of_the_letter[character_padding]] = one
    return eng_rep







# In[ ]:

def language_alphabets():                                           # returns hindi and english alphabets
    hi_start = 2304
    hi_end = 2432
    en = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    hi = [chr(z) for z in range(hi_start, hi_end)]
    return en,hi

character_padding = '-PAD-'               # padding 
alphabet_index_eng = {character_padding: 0}
alphabet_index_hindi = {character_padding: 0}


alphabets_in_english,alphabets_in_hindi = language_alphabets()       

for a, b in enumerate(alphabets_in_english):
    c=a
    alphabet_index_eng[b] = c+1                                      # adds index to each english alphabet and is stored as dictionary
for x, y in enumerate(alphabets_in_hindi):
    d=a
    alphabet_index_hindi[y] = d+1                                     # adds index to each hindi alphabet and is stored as dictionary




# In[ ]:
def return_empty_sentence():
    empty_sentence = ''
    return empty_sentence
comma_ = ','
regular_expression_for_non_eng = '[^a-zA-Z ]'
regular_expression_non_english_letters = re.compile(regular_expression_for_non_eng)
empty_sentence = ''
dash = '-'
blank_space = ' '
def Filter_Vocab_HINDI(hindi_sentence):                                                          # Keeping only hindi letters and discarding the rest
    
    hindi_sentence = hindi_sentence.replace(dash,blank_space).replace(comma_, blank_space)       # replacing dash and commas by blankspace
    new_sentence = return_empty_sentence()

    for hindi_character in hindi_sentence:
        if hindi_character == blank_space or hindi_character in alphabet_index_hindi :
            new_sentence += hindi_character
    splitted_sentence_hindi = new_sentence.split()
    #print(splitted_sentence_hindi)
    return splitted_sentence_hindi


def Filter_Vocab_ENG(eng_sentence):                                                              # Keeping only English letters and discarding the rest
    eng_sentence = eng_sentence.replace(dash,blank_space).replace(comma_,blank_space).upper()
    eng_sentence = regular_expression_non_english_letters.sub('', eng_sentence)
    return eng_sentence.split()




# In[ ]:





def load_data():                                            # function to load data files ( loaded only train and test here . can add validation data in place of test)
    dtrain= Data_loading_class('hin_train.csv')
    dtest = Data_loading_class('hin_test.csv')
    return dtrain,dtest

# In[ ]:


data_training,data_testing = load_data()




# Non attetion based encoder class
class NonAttention_EncDec(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, bidirectional, num_layers, num_layers_, celltype, dropout = 0, verbose=False):
        super(NonAttention_EncDec, self).__init__()
        self.celltype = celltype #
        self.hidden_size = hidden_size
        self.output_size = output_size
        dimension = 2
        #
        if celltype == 'gru':
            self.ENCODER_CELL = nn.GRU(input_size, hidden_size,bidirectional = bidirectional,num_layers = num_layers,dropout = dropout)    # creating encoder cell for gru
            self.DECODER_CELL = nn.GRU(output_size, hidden_size,bidirectional = bidirectional,num_layers = num_layers,dropout = dropout)
        elif celltype == 'lstm':
            self.ENCODER_CELL = nn.LSTM(input_size, hidden_size,bidirectional = bidirectional,num_layers = num_layers,dropout = dropout)  # lstm
            self.DECODER_CELL = nn.LSTM(output_size, hidden_size,bidirectional = bidirectional,num_layers = num_layers,dropout = dropout)
        else:
            self.ENCODER_CELL = nn.RNN(input_size, hidden_size,bidirectional = bidirectional,num_layers = num_layers,dropout = dropout)   #rnn
            self.DECODER_CELL = nn.RNN(output_size, hidden_size,bidirectional = bidirectional,num_layers = num_layers,dropout = dropout)
        #
        
        self.hidden_to_output_layer = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=dimension)                      # softmax activation function
        
        self.verbose = verbose
        
    def forward(self, input, max_output_chars = OUTPUT_CHAR_UPPER_LIMIT, device = device_gpu, ground_truth = None):

        one = 1
        if self.celltype == 'lstm':
            o, (hidden,cell) = self.ENCODER_CELL(input)
        else:
            o, hidden = self.ENCODER_CELL(input)

        max_output_chars = OUTPUT_CHAR_UPPER_LIMIT
        # decoder
        state_decoder = hidden
        two = 2
        #
        if self.celltype == 'lstm':
            cell_decoder = cell
        #
        decoder_input = torch.zeros(one, one, self.output_size).to(device)
        ARR_outputs = []

        
        for j in range(OUTPUT_CHAR_UPPER_LIMIT):
            #
            zer =0 
            temp_for_neg = -1
            if self.celltype == 'lstm':
                o, (state_decoder,cell_decoder) = self.DECODER_CELL(decoder_input, (state_decoder,cell_decoder))
            else:
                o, state_decoder = self.DECODER_CELL(decoder_input, state_decoder)
           
            
            o = self.hidden_to_output_layer(state_decoder)
            o = self.softmax(o)
            ARR_outputs.append(o.view(one, temp_for_neg))

            store = o.shape
            index_maximisation = torch.argmax(o, two, keepdim=True)
            if not ground_truth is None:
                index_maximisation = ground_truth[j].reshape(one, one, one)
            ohvector = torch.FloatTensor(store).to(device)
            oneHot = ohvector
            oneHot.zero_()
            oneHot.scatter_(two, index_maximisation, one)
            
            decoder_input = oneHot.detach()
            
        return ARR_outputs





# In[ ]:

# attention based encoder decoder class
class Attention_EncDec(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size,bidirectional,num_layers,num_layers_,celltype,dropout = 0, verbose=False):
        super(Attention_EncDec, self).__init__()
        self.celltype = celltype #
        
        self.hidden_size = hidden_size
        hs_for_att = hidden_size*2

        self.output_size = output_size
        
        #
        if celltype == 'gru':

            self.ENCODER_CELL = nn.GRU(input_size, hidden_size,bidirectional = bidirectional,dropout = dropout)               # enocder cell creation
            self.DECODER_CELL = nn.GRU(hs_for_att, hidden_size,bidirectional = bidirectional,dropout = dropout)
        elif celltype == 'lstm':

            self.ENCODER_CELL = nn.LSTM(input_size, hidden_size,bidirectional = bidirectional,dropout = dropout)
            self.DECODER_CELL = nn.LSTM(hs_for_att, hidden_size,bidirectional = bidirectional,dropout = dropout)
        else:

            self.ENCODER_CELL = nn.RNN(input_size, hidden_size,bidirectional = bidirectional,dropout = dropout)
            self.DECODER_CELL = nn.RNN(hs_for_att, hidden_size,bidirectional = bidirectional,dropout = dropout)
        #
        one = 1
        
        self.U = nn.Linear(self.hidden_size, self.hidden_size)

        self.attention_ = nn.Linear(self.hidden_size, one)
        dimension = 2
        self.W = nn.Linear(self.hidden_size, self.hidden_size)
        self.softmax = nn.LogSoftmax(dim=dimension)                                         # using softmax activation function
        self.o_to_hid = nn.Linear(self.output_size, self.hidden_size)                       # using linear for output to hidden
        self.hidden_to_output_layer = nn.Linear(hidden_size, output_size)
          
        
        self.verbose = verbose
        
    def forward(self, in_value, max_output_chars = OUTPUT_CHAR_UPPER_LIMIT, device = device_gpu, ground_truth = None):
        
        # encoder

        #
        if self.celltype == 'lstm':
            out_of_encoder, (hidden,cell) = self.ENCODER_CELL(in_value)
        else:
            out_of_encoder, hidden = self.ENCODER_CELL(in_value)
        some_val = -1
        one = 1

        out_of_encoder = out_of_encoder.view(some_val, self.hidden_size)
        max_O_char =  max_output_chars

        
        # decoder
        state_decoder = hidden
        VALUE_OUTPUT = []
        decoder_input = torch.zeros(one, one, self.output_size).to(device)

        #
        if self.celltype == 'lstm':
            cell_decoder = cell
        #
        
       
        U = self.U(out_of_encoder)
        

        
        for i in range(max_O_char):
            zer = 0
            W = self.W(state_decoder.view(one, some_val).repeat(out_of_encoder.shape[zer], one))
            V = self.attention_(torch.tanh(U + W))
            temp_var = 1
            temp_var_ = 2
            WEIGHTS_attention = F.softmax(V.view(temp_var, -1), dim = temp_var)             # computation of attention weights

            
            ATT_done = torch.bmm(WEIGHTS_attention.unsqueeze(zer),
                                 out_of_encoder.unsqueeze(zer))
            

            decoder_input = torch.cat(((self.o_to_hid(decoder_input))[0], ATT_done[zer]), 1).unsqueeze(zer)

                


            #
            if self.celltype == 'lstm':
                o, (state_decoder,cell_decoder) = self.DECODER_CELL(decoder_input, (state_decoder,cell_decoder))
            else:
                o, state_decoder = self.DECODER_CELL(decoder_input, state_decoder)
            #
 
                
            o = self.hidden_to_output_layer(state_decoder)
            boolval = True
            o = self.softmax(o)
            VALUE_OUTPUT.append(o.view(1, -1))
            not_gt = not ground_truth
   
            
            index_maximisation = torch.argmax(o, temp_var_, keepdim=boolval)
            if not ground_truth is None:
                index_maximisation = ground_truth[i].reshape(temp_var, temp_var, temp_var)
            oh_v = torch.zeros(o.shape, device=device)                            # one hot vector computation
            oh_v.scatter_(temp_var_, index_maximisation, temp_var) 
            
            decoder_input = oh_v.detach()
            
        return VALUE_OUTPUT


# In[ ]:





# In[ ]:
def return_engwordrep(inp):
    return english_word_representation(alphabet_index_eng,inp)



def workOut(net, inp,char_limit_at_o, device=device_gpu):
    net.eval().to(device)
    word_ohe = return_engwordrep(inp)
    print(word_ohe)
    output = net(word_ohe, char_limit_at_o)
    print(output)
    return output



def accuracy_calculation(model, device = device_gpu):        # function to compute accuracy
    model = model.eval().to(device)
    len_test = len(data_testing)
    zer = 0 
    acc = 0
    one = 1
    for i in range(len_test):
        e, h = data_testing[i]
        hindi_wordrep = hindi_word_representation(alphabet_index_hindi,h, device)
        print(e)
        
        ans_true = 0
        o = workOut(model, e, hindi_wordrep.shape[zer], device)
        print(h)
        
        for index, outpt in enumerate(o):
            val, ind = outpt.topk(one)
            value = val
            hindi_word_position = ind.tolist()[zer]
            temp = hindi_wordrep [index][zer]
            temp3 = hindi_word_position[zer]                  # checking letter by letter . if not equal, the word is considered wrong.
            if temp3 == temp:
                ans_true += one
        temp2 = hindi_wordrep.shape[zer]
        acc += ans_true/temp2
    acc /= len_test
    return acc


# In[ ]:

def calc_loss(o,method , word, bsize_):
    loss = method(o, word) / bsize_
    return loss


def loss_compute_for_training(model, optimizer, method, bsize_, device =device_gpu, teacher_forcing = False):
    
    ZER = 0 
    batchsize = bsize_
    tforce = teacher_forcing
    eng_batch, hindi_batch = data_training.extract_batch_for_lang(bsize_)
    model.train().to(device)
    optimizer.zero_grad()
    
    TOTAL = ZER
    for i in range(batchsize):
        eng_word = return_engwordrep(eng_batch[i])                                                        # calling eng word representation
        hindi_word = hindi_word_representation(alphabet_index_hindi,hindi_batch[i], device)               # calling hindi word representation
        temp = hindi_word.shape[0]
        o = model(eng_word, temp, device, ground_truth = hindi_word if tforce else None)       # calling model with hindi word as ground truth if teacher forcing is in place
        
        for i, o in enumerate(o):
            l = calc_loss(o,method,hindi_word[i],bsize_)         # loss calculation
            TOTAL = TOTAL 
            l.backward(retain_graph = True)                     # back propagating the loss
            TOTAL += l
    value = TOTAL/bsize_
    optimizer.step()
    return value


# In[ ]:

# function to run the model. takes in batch size , momentum , lr as paramters
def model_Run(net,  size_ = 100,num_batches = 1000, learning_r = 0.001,momentum = 0.9, device =device_gpu):
    t = 3
    limit = num_batches//t
    method = nn.NLLLoss(ignore_index = -1)
    net = net.to(device)
    one =1 
    optimizer = optim.Adam(net.parameters(), lr=learning_r)   # adam optimizer used
    freq = 100
    temp = num_batches + one
    loss = np.zeros(temp)
    
    for i in range(num_batches):
        t1 = i+one
        loss[i+1] = (loss[i]*i + loss_compute_for_training(net, optimizer, method, size_, device = device, teacher_forcing = i<limit ))/(t1)  # loss calculation

        if i%freq == 0:
          acc=accuracy_calculation(net)
          print(acc)
          wandb.log({'validation_acc': acc})   # logging  accuracy onto wandb
        wandb.log({'validation_loss': loss[i]})    # logging loss onto wandb
            
    torch.save(net, 'model.pt')
    
    return loss








# In[ ]:



# In[ ]:



def wandb_run():

    wandb.init(project="dl_asg_3_att", resume=True)


    wandb.run.name = f'inp_embed_{wandb.config.input_embedding}enclayer{wandb.config.no_of_enc}declayer{wandb.config.no_of_dec}hidden{wandb.config.hidden_size}cell{wandb.config.cell_type}drop{wandb.config.dropout}'
    # calling the model with hyperparameters combination which wandb chose
    model = Attention_EncDec(   
        input_size = len(alphabet_index_eng),
        hidden_size = wandb.config.hidden_size, 
        output_size = len(alphabet_index_hindi),

        bidirectional = wandb.config.bidirectional,
        num_layers = wandb.config.no_of_enc,
        num_layers_ = wandb.config.no_of_enc,
        celltype = wandb.config.cell_type, 
        dropout = wandb.config.dropout, 

    )
    #model = NonAttention_EncDec(len(alphabet_index_eng), 64, len(alphabet_index_hindi),bidirectional=False,num_layers=2,num_layers_=2,celltype = 'lstm',dropout=0.4)

    device = device = torch.device('cuda' if torch.cuda.is_available() else device_gpu)

    model.to(device_gpu)
    iterations = 2000
    # calling model_run
    val_loss = model_Run(model, size_=64,num_batches = iterations,learning_r=wandb.config.lr)


# In[ ]:




sweep_configuration = {                                                         # hyperparamter config
    'method': 'bayes',
    'name': 'dl_asg_3_att',
    'metric': {
        'goal': 'maximize', 
        'name': 'validation_acc'                                               # metric to maximise validation accuracy
        },
    'parameters': {
        'input_embedding': {'values': [128, 256, 512]},
        'no_of_enc': {'values': [1, 2, 3]},
        'no_of_dec': {'values': [1, 2, 3]},
        'hidden_size': {'values': [64,128, 256, 512]},
        'cell_type': {'values': ['lstm','gru','rnn']},
        'dropout': {'values': [0,0.1,0.2,0.3, 0.4]},
        'bidirectional' : {'values' : [True,False]},
        'lr' : {'values' : [0.001]}
     }
}

wandb.login(key = '64052dababb875ac503401af808849d971fb4177')            # logging into wandb with api key
sweep_id = wandb.sweep(sweep=sweep_configuration, project='dl_asg_3_att')
wandb.agent(sweep_id, function=wandb_run, count=100)
wandb.finish()


# In[ ]:




