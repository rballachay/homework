import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import random
from datasets import load_dataset
from torch.utils.data import DataLoader
import pandas as pd
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##### PROVIDED CODE #####

def tokenize(
    text: "list[str]", max_length: int = None, normalize: bool = True
) -> "list[list[str]]":
    import re
    if normalize:
        regexp = re.compile("[^a-zA-Z ]+")
        # Lowercase, Remove non-alphanum
        text = [regexp.sub("", t.lower()) for t in text]
    return [t.split()[:max_length] for t in text]

def build_index_map(
    word_counts: "dict[str, int]", max_words: int = None
) -> "dict[str, int]":
    sorted_counts = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)
    if max_words:
        sorted_counts = sorted_counts[:max_words-1]
    sorted_words = ["[PAD]"] + [item[0] for item in sorted_counts]
    return {word: ix for ix, word in enumerate(sorted_words)}

# modify build_word_counts for SNLI
# so that it takes into account batch['premise'] and batch['hypothesis']
def build_word_counts(dataloader) -> "dict[str, int]":
    word_counts = {}
    for batch in dataloader:
        for words in tokenize(batch['premise']):
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
        for words in tokenize(batch['hypothesis']):
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
    return word_counts

def tokens_to_ix(
    tokens: "list[list[str]]", index_map: "dict[str, int]"
) -> "list[list[int]]":
    return [
        [index_map[word] for word in words if word in index_map] for words in tokens
    ]


##### END PROVIDED CODE #####

class CharSeqDataloader():
    def __init__(self, filepath, seq_len, examples_per_epoch):

        with open(filepath,'r') as _txt:
            self.data = _txt.read()

        self.unique_chars = list(set(self.data))
        self.vocab_size = len(self.unique_chars)
        self.mappings = self.generate_char_mappings(self.unique_chars)
        self.seq_len = seq_len
        self.examples_per_epoch = examples_per_epoch
    
    def generate_char_mappings(self, uq):
        charmap = {"char_to_idx":{},"idx_to_char":{}}
        for i,char in enumerate(uq):
            charmap["char_to_idx"][char] = i
            charmap["idx_to_char"][i] = char
        return charmap

    def convert_seq_to_indices(self, seq):
        _seq = []
        for s in seq:
            _seq.append(self.mappings['char_to_idx'][s])
        return _seq

    def convert_indices_to_seq(self, seq):
        _seq = []
        for s in seq:
            _seq.append(self.mappings['idx_to_char'][s])
        return _seq

    def get_example(self):
       # can't start less that seq_len from the end
       len_seq = len(self.data)-self.seq_len
       for _ in range(self.examples_per_epoch):
           start_idx = random.randint(0,len_seq)

           seq = self.data[start_idx:start_idx+self.seq_len+1]
           idxs = self.convert_seq_to_indices(seq)

           in_seq = torch.tensor(idxs[:-1]).int()
           target_seq = torch.tensor(idxs[1:]).int()
           yield (in_seq, target_seq)


class CharRNN(nn.Module):
    def __init__(self, n_chars, embedding_size, hidden_size):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_chars = n_chars

        self.embedding_size = embedding_size
        self.embedding_layer = nn.Embedding(self.n_chars, self.embedding_size)

        # your code here
        self.waa = nn.Linear(self.hidden_size,self.hidden_size)
        self.wax = nn.Linear(self.embedding_size,self.hidden_size,bias=False)
        self.wya = nn.Linear(self.hidden_size,self.n_chars)
        
    def rnn_cell(self, i, h):
        h_new = torch.tanh(self.waa(h) + self.wax(i))
        o = self.wya(h_new)
        return o, h_new

    def forward(self, input_seq, hidden = None):
        # init hidden
        if hidden is None:
            hidden=torch.zeros(self.hidden_size)

        preds = []
        for step in range(len(input_seq)):
            i = self.embedding_layer(input_seq[step])
            yt, hidden = self.rnn_cell(i,hidden)
            preds.append(yt)

        preds = torch.stack(preds)
        return preds, hidden


    def get_loss_function(self):
        return nn.CrossEntropyLoss()

    def get_optimizer(self, lr):
        return torch.optim.Adam(self.parameters(),lr=lr)
    
    def sample_sequence(self, starting_char, seq_len, temp=0.5, top_k=None, top_p=None):
        char = starting_char
        hidden=None
        sequence = [char]
        for _ in range(seq_len):
            preds, hidden = self.forward(torch.tensor([char]).unsqueeze(1),hidden)
            preds = preds/temp
            preds = nn.functional.softmax(preds,dim=-1)

            if top_k is not None:
                preds = top_k_filtering(preds,top_k)
            
            if top_p is not None:
                preds = top_p_filtering(preds,top_p)

            char =  int(Categorical(probs=preds).sample())
            sequence.append(char)
        return sequence

class CharLSTM(nn.Module):
    def __init__(self, n_chars, embedding_size, hidden_size):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.n_chars = n_chars

        concat_size = self.embedding_size+self.hidden_size

        self.embedding_layer = nn.Embedding(self.n_chars, self.embedding_size)
        self.forget_gate = nn.Linear(concat_size, hidden_size)
        self.input_gate = nn.Linear(concat_size, hidden_size)
        self.output_gate = nn.Linear(concat_size, hidden_size)
        self.cell_state_layer = nn.Linear(concat_size, hidden_size)
        self.fc_output = nn.Linear(hidden_size,n_chars)

    def forward(self, input_seq, hidden = None, cell = None):
        # init hidden
        if hidden is None:
            hidden=torch.zeros(self.hidden_size)
        if cell is None:
            cell = torch.zeros(self.hidden_size)

        preds = []
        for step in range(len(input_seq)):
            i = self.embedding_layer(input_seq[step]).squeeze()
            yt, hidden, cell = self.lstm_cell(i,hidden,cell)
            preds.append(yt)

        preds = torch.stack(preds)
        return preds, hidden, cell

    def lstm_cell(self, i, h, c):
        combined_input = torch.cat((i, h), dim=0)
        
        forget_gate_output = torch.sigmoid(self.forget_gate(combined_input))
        input_gate_output = torch.sigmoid(self.input_gate(combined_input))
        cell_gate_output = torch.tanh(self.cell_state_layer(combined_input))

        output_gate_output = torch.sigmoid(self.output_gate(combined_input))
        
        cell_state = forget_gate_output * c + input_gate_output * cell_gate_output
        hidden_state = output_gate_output * torch.tanh(cell_state)

        output = self.fc_output(hidden_state)
        
        return output, hidden_state, cell_state

    def get_loss_function(self):
        return nn.CrossEntropyLoss()

    def get_optimizer(self, lr):
        return torch.optim.Adam(self.parameters(),lr=lr)
    
    def sample_sequence(self, starting_char, seq_len, temp=0.5, top_k=None, top_p=None):
        char = starting_char
        hidden=None
        cell=None
        sequence = [char]
        for _ in range(seq_len):
            preds, hidden, cell = self.forward(torch.tensor([char]).unsqueeze(1),hidden, cell)
            preds = preds/temp
            preds = nn.functional.softmax(preds,dim=-1)

            if top_k is not None:
                preds = top_k_filtering(preds,top_k)
            
            if top_p is not None:
                preds = top_p_filtering(preds,top_p)

            char =  int(Categorical(probs=preds).sample()[0])
            sequence.append(char)
        return sequence


def top_k_filtering(logits, top_k=40):
    logits_out = torch.zeros(logits.shape, dtype=logits.dtype)
    topk_values,topk_indices = torch.topk(logits,top_k)
    logits_out.scatter_(1, topk_indices, topk_values)
    logits_out =logits_out / logits_out.sum(dim=1).unsqueeze(-1)
    logits_out[logits_out==0] = -np.inf
    logits_out[logits_out==-0] = -np.inf
    return logits_out

def top_p_filtering(logits, top_p=0.9):
    logits_out = torch.zeros(logits.shape)
    _logits=nn.functional.softmax(logits,dim=-1)
    sorted, indices = torch.sort(_logits,descending=True)
    idx = (sorted.cumsum(axis=1)>top_p).cumsum(axis=1)<=1
    idx = idx.gather(1, indices.argsort(1))
    logits_out[idx] = logits[idx]
    logits_out =logits_out / logits_out.sum(dim=1).unsqueeze(-1)
    logits_out[logits_out==0] = -np.inf
    logits_out[logits_out==-0] = -np.inf
    return logits_out

def train(model, dataset, lr, out_seq_len, num_epochs):

    # code to initialize optimizer, loss function
    running_loss = []
    total_loss = []

    optimizer = model.get_optimizer(lr)
    loss_fn = model.get_loss_function()

    for epoch in range(num_epochs):
        model.train()
        for in_seq, out_seq in dataset.get_example():
            preds, *_ = model.forward(in_seq,None)

            loss = loss_fn(preds,out_seq.long())
            optimizer.zero_grad()

            loss.backward()

            # Adjust learning weights
            optimizer.step()

            running_loss.append(int(loss))

        # print info every X examples
        #print(f"Epoch {epoch}. Running loss so far: {(np.mean(running_loss)):.8f}")

        #print("\n-------------SAMPLE FROM MODEL-------------")

        # code to sample a sequence from your model randomly
        model.eval()
        with torch.no_grad():
            char = random.sample(dataset.unique_chars, len(dataset.unique_chars))[0]
            idx = dataset.mappings['char_to_idx'][char]
            seq = model.sample_sequence(idx, out_seq_len)
            seq_char = ''.join(dataset.convert_indices_to_seq(seq))
            #print(seq_char)

        total_loss.append(np.mean(running_loss))
        #print("\n------------/SAMPLE FROM MODEL/------------")
        running_loss = []
        
    print("\n------------/RESULTS FROM MODEL/------------")
    for i in range(5):
        print(f"\n------------/SAMPLE NUMBER {i+1}/------------")
        char = random.sample(dataset.unique_chars, len(dataset.unique_chars))[0]
        idx = dataset.mappings['char_to_idx'][char]
        
        print(f"Seed character: {char}\n")
        for temp in [0.1,0.5,0.9]:
            print(f"\n------------/FOR TEMPERATURE = {temp}/------------")
            seq = model.sample_sequence(idx, out_seq_len, temp=temp)
            seq_char = ''.join(dataset.convert_indices_to_seq(seq))
            print(seq_char)

    print(f"Seed character: {char}\n")
    preds, *_ = model.forward(torch.tensor([idx]),None)
    preds = preds/temp
    preds = nn.functional.softmax(preds,dim=-1)
    top_k = top_k_filtering(preds,5)
    top_k = [int(i) for i in np.argwhere(top_k!=-np.inf)[1]]
    top_k = dataset.convert_indices_to_seq(top_k)
    
    top_p = top_p_filtering(preds,0.5)
    top_p = [int(i) for i in np.argwhere(top_p!=-np.inf)[1]]
    top_p = dataset.convert_indices_to_seq(top_p)
    print(f"Top K = {', '.join(top_k)}")
    print(f"Top P = {', '.join(top_p)}")
    
    return total_loss # return model optionally

def run_char_rnn(data_path, num_epochs):
    hidden_size = 512
    embedding_size = 300
    seq_len = 100
    lr = 1e-3
    epoch_size = 32 # one epoch is this # of examples
    out_seq_len = 200

    # code to initialize dataloader, model
    dataset = CharSeqDataloader(data_path,seq_len,epoch_size)
    model = CharRNN(dataset.vocab_size,embedding_size,hidden_size)

    loss = train(model, dataset, lr=lr, 
                out_seq_len=out_seq_len, 
                num_epochs=num_epochs)
    return loss

def run_char_lstm(data_path, num_epochs):
    hidden_size = 512
    embedding_size = 300
    seq_len = 100
    lr = 1e-3
    epoch_size = 32
    out_seq_len = 200

    # code to initialize dataloader, model
    dataset = CharSeqDataloader(data_path,seq_len,epoch_size)
    model = CharLSTM(dataset.vocab_size,embedding_size,hidden_size)

    loss = train(model, dataset, lr=lr, 
                out_seq_len=out_seq_len, 
                num_epochs=num_epochs)
    return loss


def fix_padding(batch_premises, batch_hypotheses):
    premise_tensors = [torch.tensor(lst) for lst in batch_premises]
    premise_tensors_r = [torch.tensor(lst[::-1]) for lst in batch_premises]

    premise_padded_tensor = nn.utils.rnn.pad_sequence(premise_tensors, batch_first=True, padding_value=0)
    premise_padded_tensor_r = nn.utils.rnn.pad_sequence(premise_tensors_r, batch_first=True, padding_value=0)

    hypo_tensors = [torch.tensor(lst) for lst in batch_hypotheses]
    hypo_tensors_r = [torch.tensor(lst[::-1]) for lst in batch_hypotheses]

    hypo_padded_tensor = nn.utils.rnn.pad_sequence(hypo_tensors, batch_first=True, padding_value=0)
    hypo_padded_tensor_r = nn.utils.rnn.pad_sequence(hypo_tensors_r, batch_first=True, padding_value=0)
    return premise_padded_tensor, hypo_padded_tensor, premise_padded_tensor_r, hypo_padded_tensor_r 


def create_embedding_matrix(word_index, emb_dict, emb_dim):
    n_words = len(word_index)
    embeddings = np.zeros((n_words,emb_dim))
    for word, i in word_index.items():
        embeddings[i,:] = emb_dict.get(word,np.zeros(emb_dim))
    return torch.from_numpy(embeddings).float()

def evaluate(model, dataloader, index_map):
    y = []
    y_hat = []
    for batch in dataloader:
        premise = tokens_to_ix(tokenize(batch['premise']),index_map)
        hypothesis = tokens_to_ix(tokenize(batch['hypothesis']),index_map)
        preds = model.forward(premise,hypothesis).detach().to('cpu')
        preds = list(torch.argmax(preds, dim=1))

        y_hat.extend(preds)
        y.extend(list(batch['label'].detach().to('cpu')))

    return sum(1 for x,y in zip(y,y_hat) if x == y) / len(y)


class UniLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, num_classes):
        super(UniLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.embedding_layer = nn.Embedding(self.vocab_size, self.hidden_dim, padding_idx=0)

        # your code here
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.int_layer = nn.Linear(2*self.hidden_dim,self.hidden_dim)
        self.out_layer = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, a, b):
        (a, b, a_r, b_r)  = fix_padding(a, b)
        a = self.embedding_layer(a.to(device))
        b = self.embedding_layer(b.to(device))
        _, (_, a) = self.lstm(a)
        _, (_, b) = self.lstm(b)

        # just want the very last of the sequence
        ab = torch.cat((a[-1], b[-1]), dim=-1)
        ab = self.int_layer(ab)
        ab = nn.functional.relu(ab)
        ab = self.out_layer(ab)
        return ab


class ShallowBiLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, num_classes):
        super(ShallowBiLSTM, self).__init__()
        self.hidden_dim = hidden_dim//2
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.embedding_layer = nn.Embedding(self.vocab_size, self.hidden_dim, padding_idx=0)

        # your code here
        self.lstm_forward = nn.LSTM(self.hidden_dim,self.hidden_dim//2, num_layers, batch_first=True)
        self.lstm_backward = nn.LSTM(self.hidden_dim,self.hidden_dim//2, num_layers, batch_first=True)

        self.int_layer = nn.Linear(2*self.hidden_dim,self.hidden_dim)
        self.out_layer = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, a, b):
        (a, b, a_r, b_r)  = fix_padding(a, b)
        a = self.embedding_layer(a.to(device))
        b = self.embedding_layer(b.to(device))
        a_r = self.embedding_layer(a_r.to(device))
        b_r = self.embedding_layer(b_r.to(device))
        _, (_,a) = self.lstm_forward(a)
        _, (_,b) = self.lstm_forward(b)
        _, (_,a_r) = self.lstm_backward(a_r)
        _, (_,b_r) = self.lstm_backward(b_r)
        # just want the very last of the sequence
        ab = torch.cat((a[-1], a_r[-1], b[-1], b_r[-1]), dim=1)
        ab = self.int_layer(ab)
        ab = nn.functional.relu(ab)
        ab = self.out_layer(ab)
        return ab

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, num_classes):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.embedding_layer = nn.Embedding(self.vocab_size, self.hidden_dim, padding_idx=0)

        # your code here
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.int_layer = nn.Linear(4*self.hidden_dim,self.hidden_dim)
        self.out_layer = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, a, b):
        (a, b, a_r, b_r)  = fix_padding(a, b)
        a = self.embedding_layer(a.to(device))
        b = self.embedding_layer(b.to(device))
        _, (_, a) = self.lstm(a)
        _, (_, b) = self.lstm(b)

        # just want the very last of the sequence
        ab = torch.cat((a[-1], a[-2], b[-1],b[-2]), dim=-1)
        ab = self.int_layer(ab)
        ab = nn.functional.relu(ab)
        ab = self.out_layer(ab)
        return ab


def run_snli(model, use_glove=True):
    dataset = load_dataset("snli")
    glove = pd.read_csv('./data/glove.6B.100d.txt', sep=" ", quoting=3, header=None, index_col=0)

    train_filtered = dataset['train'].filter(lambda ex: ex['label'] != -1)
    valid_filtered = dataset['validation'].filter(lambda ex: ex['label'] != -1)
    test_filtered =  dataset['test'].filter(lambda ex: ex['label'] != -1)

    # code to make dataloaders
    dataloader_train = DataLoader(
        train_filtered,
        batch_size=320,
        shuffle=True,
    )

    dataloader_valid = DataLoader(
        valid_filtered,
        batch_size=320,
        shuffle=True,
    )

    dataloader_test = DataLoader(
        test_filtered,
        batch_size=320,
        shuffle=True,
    )


    word_counts = build_word_counts(dataloader_train)
    index_map = build_index_map(word_counts)

    emb_dict = {word:embed.values for word,embed in glove.iterrows()}
    glove_embeddings = create_embedding_matrix(index_map,emb_dict,glove.shape[1])

    # training code
    model = model(len(index_map),100, 1, 3)

    if use_glove:
        model.embedding_layer.from_pretrained(glove_embeddings, freeze=False, padding_idx=0)

    # move the model
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-2)

    results = []
    running_loss=[]
    current_acc = -np.inf
    patience = 0
    for epoch in range(15):
        model.train()
        loss = []
        for batch in dataloader_train:
            premise = tokens_to_ix(tokenize(batch['premise']),index_map)
            hypothesis = tokens_to_ix(tokenize(batch['hypothesis']),index_map)
            label = batch['label'].to(device)
            preds = model.forward(premise,hypothesis)
            loss = loss_fn(preds, label)

            optimizer.zero_grad()
            loss.backward()
            print(loss)

            # Adjust learning weights
            optimizer.step()

            running_loss.append(int(loss))

        print("STARTING EVAL\n\n")
        model.eval()
        with torch.no_grad():
            valid_acc = evaluate(model,dataloader_valid,index_map)
            test_acc = evaluate(model,dataloader_test,index_map)     
        
        results.append({
            'train':np.mean(running_loss),
            'test':test_acc,
            'valid':valid_acc,
            'epoch':epoch+1
        })

        if valid_acc<current_acc:
            patience+=1
            if patience>1:
                print("Validation accuracy has gone down. Breaking out of loop")
                break
        else:
            patience = 0

        current_acc = valid_acc

    return pd.DataFrame(results)



def run_snli_lstm(glove=True):
    model_class = UniLSTM
    return run_snli(model_class, glove)

def run_snli_bilstm(glove=True):
    model_class = ShallowBiLSTM # fill in the classs name of the model (to initialize within run_snli)
    return run_snli(model_class, glove)


def __part_1():
    data_path = ["./data/shakespeare.txt","./data/sherlock.txt"]
    num_epochs=100
    results = {'epoch':[],'loss':[],'data':[],'model':[]}
    for data in data_path:
        print(f"\n\n------------/STARTING RNN WITH DATA {data}/------------")
        loss = run_char_rnn(data,num_epochs)
        results['epoch'].extend(np.arange(len(loss)))
        results['loss'].extend(loss)
        results['data'].extend([data.split('/')[-1].replace('.txt','')]*len(loss))
        results['model'].extend(['rnn']*len(loss))

        print(f"\n\n------------/STARTING LSTM WITH DATA {data}/------------")
        loss = run_char_lstm(data,num_epochs)
        results['epoch'].extend(np.arange(len(loss)))
        results['loss'].extend(loss)
        results['data'].extend([data.split('/')[-1].replace('.txt','')]*len(loss))
        results['model'].extend(['lstm']*len(loss))

    results =pd.DataFrame(results)
    sns.set_theme()
    plot = sns.lineplot(data=results,x='epoch',y='loss',hue='data',style="model")
    fig = plot.get_figure()
    fig.savefig(f'results/char_rnn_lstm_results.png',dpi=200)


def __part_2():
    import os
    all_results = []

    path = f'results/snli_bilstm_results.csv'

    if not os.path.exists(path):
        results = run_snli(BiLSTM, True)
        results.to_csv(f'results/snli_bilstm_results.csv')
    else:
        results = pd.read_csv(path)
        results['model'] = 'lstm/bilstm'
        results['glove'] = True

    all_results.append(results)

    for glove in [True, False]:
        
        path = f'results/snli_lstm_results{"_glove" if glove else ""}.csv'
        if not os.path.exists(path):
            results = run_snli_lstm(glove)
            results.to_csv(path)
        else:
            results = pd.read_csv(path)
            results['model'] = 'lstm'
            results['glove'] = glove

        all_results.append(results)
        
        path = f'results/snli_bilstm_shallow_results{"_glove" if glove else ""}.csv'
        if not os.path.exists(path):
            results = run_snli_bilstm(glove)
            results.to_csv(path)
        else:
            results = pd.read_csv(path)
            results['model'] = 'bilstm'
            results['glove'] = glove

        all_results.append(results)

    all_results = pd.concat(all_results,axis=0).reset_index(drop=True)
    all_results = pd.melt(all_results, id_vars=['epoch','model','glove'], 
                          value_vars=['valid','test'],var_name='stage',value_name='accuracy')
    sns.set_theme()
    plot = sns.relplot(data=all_results,x='epoch',y='accuracy',hue='model',style='glove',col='stage',kind='line')
    plot.fig.savefig('results/accuracy_for_lstm_models.png',dpi=200)


if __name__ == '__main__':
    #__part_1()
    __part_2()