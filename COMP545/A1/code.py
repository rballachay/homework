from typing import Union, Iterable, Callable
import random

import torch.nn as nn
import torch

BATCH_SIZE = 64
SHUFFLE=True
EMBED_DIM = 100
MAX_WORDS = 10000

def load_datasets(data_directory: str) -> Union[dict, dict]:
    """
    Reads the training and validation splits from disk and load
    them into memory.

    Parameters
    ----------
    data_directory: str
        The directory where the data is stored.

    Returns
    -------
    train: dict
        The train dictionary with keys 'premise', 'hypothesis', 'label'.
    validation: dict
        The validation dictionary with keys 'premise', 'hypothesis', 'label'.
    """
    import json
    import os

    with open(os.path.join(data_directory, "train.json"), "r") as f:
        train = json.load(f)

    with open(os.path.join(data_directory, "validation.json"), "r") as f:
        valid = json.load(f)

    return train, valid


def tokenize(
    text: "list[str]", max_length: int = None, normalize: bool = True
) -> "list[list[str]]":
    """
    Tokenize the text into individual words (nested list of string),
    where the inner list represent a single example.

    Parameters
    ----------
    text: list of strings
        Your cleaned text data (either premise or hypothesis).
    max_length: int, optional
        The maximum length of the sequence. If None, it will be
        the maximum length of the dataset.
    normalize: bool, default True
        Whether to normalize the text before tokenizing (i.e. lower
        case, remove punctuations)
    Returns
    -------
    list of list of strings
        The same text data, but tokenized by space.

    Examples
    --------
    >>> tokenize(['Hello, world!', 'This is a test.'], normalize=True)
    [['hello', 'world'], ['this', 'is', 'a', 'test']]
    """
    import re

    if normalize:
        regexp = re.compile("[^a-zA-Z ]+")
        # Lowercase, Remove non-alphanum
        text = [regexp.sub("", t.lower()) for t in text]

    return [t.split()[:max_length] for t in text]


def build_word_counts(token_list: "list[list[str]]") -> "dict[str, int]":
    """
    This builds a dictionary that keeps track of how often each word appears
    in the dataset.

    Parameters
    ----------
    token_list: list of list of strings
        The list of tokens obtained from tokenize().

    Returns
    -------
    dict of {str: int}
        A dictionary mapping every word to an integer representing the
        appearance frequency.

    Notes
    -----
    If you have  multiple lists, you should concatenate them before using
    this function, e.g. generate_mapping(list1 + list2 + list3)
    """
    word_counts = {}

    for words in token_list:
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

    return word_counts


def build_index_map(
    word_counts: "dict[str, int]", max_words: int = None
) -> "dict[str, int]":
    """
    Builds an index map that converts a word into an integer that can be
    accepted by our model.

    Parameters
    ----------
    word_counts: dict of {str: int}
        A dictionary mapping every word to an integer representing the
        appearance frequency.
    max_words: int, optional
        The maximum number of words to be included in the index map. By
        default, it is None, which means all words are taken into account.

    Returns
    -------
    dict of {str: int}
        A dictionary mapping every word to an integer representing the
        index in the embedding.
    """

    sorted_counts = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)
    if max_words:
        sorted_counts = sorted_counts[:max_words-1]
    
    sorted_words = ["[PAD]"] + [item[0] for item in sorted_counts]

    return {word: ix for ix, word in enumerate(sorted_words)}


def tokens_to_ix(
    tokens: "list[list[str]]", index_map: "dict[str, int]"
) -> "list[list[int]]":
    """
    Converts a nested list of tokens to a nested list of indices using
    the index map.

    Parameters
    ----------
    tokens: list of list of strings
        The list of tokens obtained from tokenize().
    index_map: dict of {str: int}
        The index map from build_index_map().

    Returns
    -------
    list of list of int
        The same tokens, but converted into indices.

    Notes
    -----
    Words that have not been seen are ignored.
    """
    return [
        [index_map[word] for word in words if word in index_map] for words in tokens
    ]


### 1.1 Batching, shuffling, iteration
def build_loader(
    data_dict: dict, batch_size: int = 64, shuffle: bool = False
) -> Callable[[], Iterable[dict]]:
    # get the length of the data dict, then get the indices 
    n_items = len(data_dict['premise'])

    # get the indexes batched up, randomizing order with pytorch
    all_idx = torch.arange(n_items)
    if shuffle:
        all_idx = all_idx[torch.randperm(all_idx.size()[0])]
    batch_idx = all_idx.split(batch_size)
    def loader():
        for batch in batch_idx:
            _loaded = {}
            for key in data_dict.keys():
                _loaded[key] = []
                for idx in batch:
                    _loaded[key].append(data_dict[key][int(idx)])
            yield _loaded
    return loader


### 1.2 Converting a batch into inputs
def convert_to_tensors(text_indices: "list[list[int]]") -> torch.Tensor:
    max_rows = max([len(batch) for batch in text_indices])
    padded = [batch + [0] * (max_rows - len(batch)) for batch in text_indices]
    return torch.tensor(padded)


### 2.1 Design a logistic model with embedding and pooling
def max_pool(x: torch.Tensor) -> torch.Tensor:
    x, _ = torch.max(x, dim=1)
    return x


class PooledLogisticRegression(nn.Module):
    def __init__(self, embedding: nn.Embedding):
        super().__init__()

        self.embedding = embedding 
        self.sigmoid = nn.Sigmoid()
        self.layer_pred = nn.Linear(EMBED_DIM*2,1)

    # DO NOT CHANGE THE SECTION BELOW! ###########################
    # # This is to force you to initialize certain things in __init__
    def get_layer_pred(self):
        return self.layer_pred

    def get_embedding(self):
        return self.embedding

    def get_sigmoid(self):
        return self.sigmoid

    # DO NOT CHANGE THE SECTION ABOVE! ###########################

    def forward(self, premise: torch.Tensor, hypothesis: torch.Tensor) -> torch.Tensor:
        emb = self.get_embedding()
        layer_pred = self.get_layer_pred()
        sigmoid = self.get_sigmoid()

        # embed
        _premise = max_pool(emb(premise))
        _hypothesis = max_pool(emb(hypothesis))

        _cat = torch.cat((_premise,_hypothesis),axis=1)
        _linear = layer_pred(_cat)
        return sigmoid(_linear).reshape(premise.shape[0])
        

### 2.2 Choose an optimizer and a loss function
def assign_optimizer(model: nn.Module, **kwargs) -> torch.optim.Optimizer:
    return torch.optim.SGD(model.parameters(), **kwargs)


def bce_loss(y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    return -(y_pred.log()*y + (1-y)*(1-y_pred).log()).mean()


### 2.3 Forward and backward pass
def forward_pass(model: nn.Module, batch: dict, device="cpu"):
    # Every data instance is an input + label pair
    x_premise = convert_to_tensors(batch['premise'])
    x_hypothesis = convert_to_tensors(batch['hypothesis'])
    y_labels = torch.tensor(batch['label'])
    return model(x_premise,x_hypothesis),y_labels
    


def backward_pass(
    optimizer: torch.optim.Optimizer, y: torch.Tensor, y_pred: torch.Tensor
) -> torch.Tensor:
    # TODO: Your code here
    # Compute the loss and its gradients
    loss = bce_loss(y,y_pred)
    loss.backward()

    # Adjust learning weights
    optimizer.step()
    return loss


### 2.4 Evaluation
def f1_score(y: torch.Tensor, y_pred: torch.Tensor, threshold=0.5,epsilon = 1e-7) -> torch.Tensor:
    y_pred = (y_pred>threshold).to(torch.float32) # take threshold & convert to float
    tp = (y * y_pred).sum()
    fp = ((1 - y) * y_pred).sum()
    fn = (y * (1 - y_pred)).sum()

    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    return f1

### 2.5 Train loop
def eval_run(
    model: nn.Module, loader: Callable[[], Iterable[dict]], device: str = "cpu"
):
    f1_scores = []
    running_vloss = 0.0
        
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, data in enumerate(loader()):
            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            y_labels,y = forward_pass(model,data)
            vloss = bce_loss(y, y_labels)
            running_vloss += vloss

            _score = f1_score(y,y_labels)
            f1_scores.append(_score)

    avg_vloss = running_vloss / (i + 1)
    return avg_vloss, torch.mean(torch.tensor(f1_scores))


def train_loop(
    model: nn.Module,
    train_loader,
    valid_loader,
    optimizer,
    n_epochs: int = 3,
    device: str = "cpu", # running on mac, ignore
):
    f1_scores = []
    running_loss = 0.
    last_loss = 0.

    for epoch in range(1, n_epochs+1):

        model.train(True)
        for i, data in enumerate(train_loader()):

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            y_labels,y = forward_pass(model,data)
            loss = backward_pass(optimizer,y, y_labels)

            # Gather data and report
            running_loss += loss.item()
            if i % 100 == 99:
                last_loss = running_loss / 100 # loss per batch
                print(f'batch {i+1} loss: {last_loss:.3f}')
                running_loss = 0.

        avg_vloss,f1_score = eval_run(model,valid_loader)
        print(f'Epoch {epoch}, LOSS train {last_loss:.3f} valid {avg_vloss:.3f}, f1 score {f1_score:.3f}')
        f1_scores.append(f1_score)
    return f1_scores



### 3.1
class ShallowNeuralNetwork(nn.Module):
    def __init__(self, embedding: nn.Embedding, hidden_size: int):
        super().__init__()

        self.embedding = embedding 
        self.sigmoid = nn.Sigmoid()
        self.activation = nn.ReLU()

        # linear layers
        self.ff_layer = nn.Linear(EMBED_DIM*2,hidden_size)
        self.layer_pred = nn.Linear(hidden_size,1)

    # DO NOT CHANGE THE SECTION BELOW! ###########################
    # # This is to force you to initialize certain things in __init__
    def get_ff_layer(self):
        return self.ff_layer

    def get_layer_pred(self):
        return self.layer_pred

    def get_embedding(self):
        return self.embedding

    def get_sigmoid(self):
        return self.sigmoid

    def get_activation(self):
        return self.activation

    # DO NOT CHANGE THE SECTION ABOVE! ###########################

    def forward(self, premise: torch.Tensor, hypothesis: torch.Tensor) -> torch.Tensor:
        emb = self.get_embedding()
        layer_pred = self.get_layer_pred()
        sigmoid = self.get_sigmoid()
        ff_layer = self.get_ff_layer()
        act = self.get_activation()

        # embed
        _premise = max_pool(emb(premise))
        _hypothesis = max_pool(emb(hypothesis))

        _cat = torch.cat((_premise,_hypothesis),axis=1)
        _linear = ff_layer(_cat)
        _relu = act(_linear)
        _final = layer_pred(_relu)
        return sigmoid(_final).reshape(premise.shape[0])


### 3.2
class DeepNeuralNetwork(nn.Module):
    def __init__(self, embedding: nn.Embedding, hidden_size: int, num_layers: int = 2):
        super().__init__()

        self.embedding = embedding 
        self.sigmoid = nn.Sigmoid()
        self.activation = nn.ReLU()

        # linear layers
        self.ff_layers = nn.ModuleList([
            nn.Linear(2*EMBED_DIM,hidden_size) if i==0 else nn.Linear(hidden_size, hidden_size) for i in range(num_layers)
        ])
        self.layer_pred = nn.Linear(hidden_size,1)
        self.num_layers = num_layers

    # DO NOT CHANGE THE SECTION BELOW! ###########################
    # # This is to force you to initialize certain things in __init__
    def get_ff_layers(self):
        return self.ff_layers

    def get_layer_pred(self):
        return self.layer_pred

    def get_embedding(self):
        return self.embedding

    def get_sigmoid(self):
        return self.sigmoid

    def get_activation(self):
        return self.activation

    # DO NOT CHANGE THE SECTION ABOVE! ###########################

    def forward(self, premise: torch.Tensor, hypothesis: torch.Tensor) -> torch.Tensor:
        emb = self.get_embedding()
        layer_pred = self.get_layer_pred()
        sigmoid = self.get_sigmoid()
        ff_layers = self.get_ff_layers()
        act = self.get_activation()

        # embed
        _premise = max_pool(emb(premise))
        _hypothesis = max_pool(emb(hypothesis))

        _cat = torch.cat((_premise,_hypothesis),axis=1)
        
        _linear=_cat
        for hidden_layer in ff_layers:
            _linear = act(hidden_layer(_linear))

        _final = layer_pred(_linear)
        return sigmoid(_final).reshape(premise.shape[0])


if __name__ == "__main__":
    # If you have any code to test or train your model, do it BELOW!

    # Seeds to ensure reproducibility
    random.seed(2022)
    torch.manual_seed(2022)

    # If you use GPUs, use the code below:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prefilled code showing you how to use the helper functions
    train_raw, valid_raw = load_datasets("data")

    train_tokens = {
        "premise": tokenize(train_raw["premise"], max_length=64),
        "hypothesis": tokenize(train_raw["hypothesis"], max_length=64),
    }

    valid_tokens = {
        "premise": tokenize(valid_raw["premise"], max_length=64),
        "hypothesis": tokenize(valid_raw["hypothesis"], max_length=64),
    }

    word_counts = build_word_counts(
        train_tokens["premise"]
        + train_tokens["hypothesis"]
        + valid_tokens["premise"]
        + valid_tokens["hypothesis"]
    )
    index_map = build_index_map(word_counts, max_words=MAX_WORDS)

    train_indices = {
        "label": train_raw["label"],
        "premise": tokens_to_ix(train_tokens["premise"], index_map),
        "hypothesis": tokens_to_ix(train_tokens["hypothesis"], index_map)
    }

    valid_indices = {
        "label": valid_raw["label"],
        "premise": tokens_to_ix(valid_tokens["premise"], index_map),
        "hypothesis": tokens_to_ix(valid_tokens["hypothesis"], index_map)
    }

    # 1.1
    train_loader = build_loader(train_indices,BATCH_SIZE, SHUFFLE)
    valid_loader = build_loader(valid_indices,BATCH_SIZE, SHUFFLE)

    # 1.2
    batch = next(train_loader())
    y_premise = convert_to_tensors(batch['premise'])
    y_hypothesis = convert_to_tensors(batch['hypothesis'])
    y = torch.tensor(batch['label'])

    # 2.1
    embedding = nn.Embedding(MAX_WORDS, EMBED_DIM)
    model = PooledLogisticRegression(embedding)

    # 2.2
    optimizer = assign_optimizer(model, lr=1e-3, momentum=0.9)

    # 2.3
    y_pred = model(y_premise, y_hypothesis)
    loss = bce_loss(y,y_pred)

    # 2.4
    score = f1_score(y,y_pred)

    # 2.5
    n_epochs = 2

    embedding = embedding
    model =  PooledLogisticRegression(embedding)
    optimizer = assign_optimizer(model,lr=1e-3, momentum=0.9)

    #scores = train_loop(model,train_loader,valid_loader,optimizer,n_epochs)

    # 3.1
    embedding = embedding
    model = ShallowNeuralNetwork(embedding, 16)
    optimizer = assign_optimizer(model, lr=1e-2, momentum=0.9)

    #scores = train_loop(model,train_loader,valid_loader,optimizer,n_epochs)

    # 3.2
    embedding = embedding
    model = DeepNeuralNetwork(embedding, 16, 4)
    optimizer = assign_optimizer(model, lr=1e-2, momentum=0.9)

    scores = train_loop(model,train_loader,valid_loader,optimizer,n_epochs)
