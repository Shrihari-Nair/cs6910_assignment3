# cs6910_assignment3
Goal of the assignment:

(i) learn how to model sequence-to-sequence learning problems using Recurrent Neural Networks

(ii) compare different cells such as vanilla RNN, LSTM and GRU 

(iii) understand how attention networks overcome the limitations of vanilla seq2seq models

## Problem Statement
Dataset :  Aksharantar dataset released by AI4Bharat
Dataset contains a word in the native script and its corresponding transliteration in the Latin script (how we type while chatting with our friends on WhatsApp etc). 
So goal is to train a model which takes as input a romanized string (ghar) and produces the corresponding word in Devanagari (घर). 

## The Code
This notebook is structured in such a way that all the cells can be run one after another. Run All Cells command can also be used, but be wary of WandB sweeps at the end.
Dataset from Aksharantar : hin_test.csv , hin_train.csv and hin_valid.csv

### Preprocessing :
The data was extracted from the above csv files. Each alphabet in English and Hindi were given a specific index. Data loading and preprocessing is done by class `Data_loading_class(Dataset)` . 

English and Hindi words were converted into thier corresponding representations by `english_word_representation()` and `hindi_word_representation` functions respectively. 

Later filtering is done by `Filter_Vocab_HINDI()` and `Filter_Vocab_ENG` functions.

### Encoder - Decoder (With and without attention)
The class `NonAttention_EncDec(nn.Module)` is modelled such that it takes initialises an object which takes in all the parameters required to model a non attention based encoder decoder model. The init function looks like this: `    def __init__(self, input_size, hidden_size, output_size, bidirectional, num_layers, num_layers_, celltype, dropout , verbose=False) `.

Parameters:
- input_size – The number of expected features in the input x
- num_layers – Number of recurrent layers (for encoder)
- num_layers_ - Number of recurrent layers (for decoder)
- hidden_size – The number of features in the hidden state h
- dropout – If non-zero, introduces a Dropout layer on the outputs of each GRU layer except the last layer, with dropout probability equal to dropout. Default: 0
- bidirectional – If True, becomes a bidirectional GRU. Default: False

The class `Attention_EncDec(nn.Module)` is modelled such that it takes initialises an object which takes in all the parameters required to model an attention based encoder decoder model. The init function looks like this: ` def __init__(self, input_size, hidden_size, output_size, bidirectional, num_layers, num_layers_, celltype, dropout , verbose=False) `.

The parameter definitions are same.

Optimizer used: Adam

Momentum value : 0.9

Batch size = 64

learning rate = 0.001



### Wandb and Running the model

For running wandb, I have used the configuration shown below:
```
sweep_configuration = {
    'method': 'bayes',
    'name': 'dl_asg_3_att',
    'metric': {
        'goal': 'maximize', 
        'name': 'validation_acc'
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
```
Creating sweep :

` sweep_id = wandb.sweep(sweep_config, project="dl_asg_3_att ")`  

Running sweep : 

`sweep_id = wandb.sweep(sweep=sweep_configuration, project='dl_asg_3_att')`

The `wandb_run()` function will take in certain combination of hyperparameters and call `NonAttention_EncDec(nn.Module)` or `Attention_EncDec(nn.Module)`. 

```
    model = Attention_EncDec(   
        input_size = len(alphabet_index_eng),
        hidden_size = wandb.config.hidden_size, 
        output_size = len(alphabet_index_hindi),
        encoder_embed_dimension =  wandb.config.input_embedding, 
        bidirectional = wandb.config.bidirectional,
        num_layers = wandb.config.no_of_enc,
        num_layers_ = wandb.config.no_of_enc,
        celltype = wandb.config.cell_type, 
        dropout = wandb.config.dropout, 

    )
```
Once we get the model trained, we can call `model_Run()` function to run the model over the desired dataset. In the function `accuracy_calculation()` we can give in the desired dataset ( validation or test ) to run the model. It returns the accuracy of the model. Loss computation is also done inside the `model_Run()` function . Both accuracy and loss is logged into wandb. 

### Train.py
The code can be run using train.py script.


| Name | Default Value | 
| :---: | :-------------: | 
| `-hz`, `--hidden_size` | 256 |
| `-enc`, `--no_of_enc_layers` | 1 |
| `-dec`, `--no_of_enc_layers` | 1 |
| `-cell`, `--celltype` | "lstm", choices=['rnn', 'lstm','gru'] |
| `-drop`, `--dropout` | 0 |
| `-att`, `--attention` | False |
| `-bi`, `--bidirectional` | False |
| `-emb`, `--embedding_size` | 256 |



