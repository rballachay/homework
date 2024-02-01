import csv
import json
import itertools
import random
from typing import Union, Callable

import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


# ########################## PART 1: PROVIDED CODE ##############################
def load_datasets(data_directory: str) -> "Union[dict, dict]":
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


def tokenize_w2v(
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
        sorted_counts = sorted_counts[: max_words - 1]

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


def collate_cbow(batch):
    """
    Collate function for the CBOW model. This is needed only for CBOW but not skip-gram, since
    skip-gram indices can be directly formatted by DataLoader. For more information, look at the
    usage at the end of this file.
    """
    sources = []
    targets = []

    for s, t in batch:
        sources.append(s)
        targets.append(t)

    sources = torch.tensor(sources, dtype=torch.int64)
    targets = torch.tensor(targets, dtype=torch.int64)

    return sources, targets


def train_w2v(model, optimizer, loader, device):
    """
    Code to train the model. See usage at the end.
    """
    model.train()

    for x, y in tqdm(loader, miniters=20, leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        y_pred = model(x)

        loss = F.cross_entropy(y_pred, y)
        loss.backward()

        optimizer.step()

    return loss


class Word2VecDataset(torch.utils.data.Dataset):
    """
    Dataset is needed in order to use the DataLoader. See usage at the end.
    """

    def __init__(self, sources, targets):
        self.sources = sources
        self.targets = targets
        assert len(self.sources) == len(self.targets)

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        return self.sources[idx], self.targets[idx]


# ########################## PART 2: PROVIDED CODE ##############################
def load_glove_embeddings(file_path: str) -> "dict[str, np.ndarray]":
    """
    Loads trained GloVe embeddings downloaded from:
        https://nlp.stanford.edu/projects/glove/
    """
    word_to_embedding = {}
    with open(file_path, "r") as f:
        for line in f:
            word, raw_embeddings = line.split()[0], line.split()[1:]
            embedding = np.array(raw_embeddings, dtype=np.float64)
            word_to_embedding[word] = embedding
    return word_to_embedding


def load_professions(file_path: str) -> "list[str]":
    """
    Loads profession words from the BEC-Pro dataset. For more information on BEC-Pro,
    see:
        https://arxiv.org/abs/2010.14534
    """
    with open(file_path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # Skip the header.
        professions = [row[1] for row in reader]
    return professions


def load_gender_attribute_words(file_path: str) -> "list[list[str]]":
    """
    Loads the gender attribute words from: https://aclanthology.org/N18-2003/
    """
    with open(file_path, "r") as f:
        gender_attribute_words = json.load(f)
    return gender_attribute_words


def compute_partitions(XY: "list[str]") -> "list[tuple]":
    """
    Computes all of the possible partitions of X union Y into equal sized sets.

    Parameters
    ----------
    XY: list of strings
        The list of all target words.

    Returns
    -------
    list of tuples of strings
        List containing all of the possible partitions of X union Y into equal sized
        sets.
    """
    return list(itertools.combinations(XY, len(XY) // 2))


def p_value_permutation_test(
    X: "list[str]",
    Y: "list[str]",
    A: "list[str]",
    B: "list[str]",
    word_to_embedding: "dict[str, np.array]",
) -> float:
    """
    Computes the p-value for a permutation test on the WEAT test statistic.

    Parameters
    ----------
    X: list of strings
        List of target words.
    Y: list of strings
        List of target words.
    A: list of strings
        List of attribute words.
    B: list of strings
        List of attribute words.
    word_to_embedding: dict of {str: np.array}
        Dict containing the loaded GloVe embeddings. The dict maps from words
        (e.g., 'the') to corresponding embeddings.

    Returns
    -------
    float
        The computed p-value for the permutation test.
    """
    # Compute the actual test statistic.
    s = weat_differential_association(X, Y, A, B, word_to_embedding, weat_association)

    XY = X + Y
    partitions = compute_partitions(XY)

    total = 0
    total_true = 0
    for X_i in partitions:
        # Compute the complement set.
        Y_i = [w for w in XY if w not in X_i]

        s_i = weat_differential_association(
            X_i, Y_i, A, B, word_to_embedding, weat_association
        )

        if s_i > s:
            total_true += 1
        total += 1

    p = total_true / total

    return p


# ######################## PART 1: YOUR WORK STARTS HERE ########################


def build_current_surrounding_pairs(indices: "list[int]", window_size: int = 2):
    # take every
    currents = []
    surroundings = []
    for idx in range(window_size, len(indices) - window_size):
        currents.append(indices[idx])

        _surr = []
        for i in range(-window_size, window_size + 1):
            if i == 0:
                continue
            _surr.append(indices[idx + i])
        surroundings.append(_surr)
    return surroundings, currents


def expand_surrounding_words(
    ix_surroundings: "list[list[int]]", ix_current: "list[int]"
):
    surrounding_expanded = [item for sublist in ix_surroundings for item in sublist]
    current_expanded = [
        elem for elem in ix_current for _ in range(len(ix_surroundings[0]))
    ]
    return surrounding_expanded, current_expanded


def cbow_preprocessing(indices_list: "list[list[int]]", window_size: int = 2):
    sources = []
    targets = []
    for i_list in indices_list:
        surr, curr = build_current_surrounding_pairs(i_list, window_size)
        sources.extend(surr)
        targets.extend(curr)

    # raise Exception(sources[0], targets[0])
    return sources, targets


def skipgram_preprocessing(indices_list: "list[list[int]]", window_size: int = 2):
    sources = []
    targets = []
    for i_list in indices_list:
        surr, curr = build_current_surrounding_pairs(i_list, window_size)
        surr, curr = expand_surrounding_words(surr, curr)
        sources.extend(surr)
        targets.extend(curr)
    return sources, targets


class SharedNNLM:
    def __init__(self, num_words: int, embed_dim: int):
        """
        SkipGram and CBOW actually use the same underlying architecture,
        which is a simplification of the NNLM model (no hidden layer)
        and the input and output layers share the same weights. You will
        need to implement this here.

        Notes
        -----
          - This is not a nn.Module, it's an intermediate class used
            solely in the SkipGram and CBOW modules later.
          - Projection does not have a bias in word2vec
        """

        # TODO vvvvvv
        self.embedding = nn.Embedding(num_words, embed_dim)
        self.projection = nn.Linear(embed_dim, num_words, bias=False)

        # TODO ^^^^^
        self.bind_weights()

    def bind_weights(self):
        """
        Bind the weights of the embedding layer with the projection layer.
        This mean they are the same object (and are updated together when
        you do the backward pass).
        """
        emb = self.get_emb()
        proj = self.get_proj()

        proj.weight = emb.weight

    def get_emb(self):
        return self.embedding

    def get_proj(self):
        return self.projection


class SkipGram(nn.Module):
    """
    Use SharedNNLM to implement skip-gram. Only the forward() method differs from CBOW.
    """

    def __init__(self, num_words: int, embed_dim: int = 100):
        """
        Parameters
        ----------
        num_words: int
            The number of words in the vocabulary.
        embed_dim: int
            The dimension of the word embeddings.
        """
        super().__init__()

        self.nnlm = SharedNNLM(num_words, embed_dim)
        self.emb = self.nnlm.get_emb()
        self.proj = self.nnlm.get_proj()

    def forward(self, x: torch.Tensor):
        _x = self.emb(x)
        return self.proj(_x)


class CBOW(nn.Module):
    """
    Use SharedNNLM to implement CBOW. Only the forward() method differs from SkipGram,
    as you have to sum up the embedding of all the surrounding words (see paper for details).
    """

    def __init__(self, num_words: int, embed_dim: int = 100):
        """
        Parameters
        ----------
        num_words: int
            The number of words in the vocabulary.
        embed_dim: int
            The dimension of the word embeddings.
        """
        super().__init__()

        self.nnlm = SharedNNLM(num_words, embed_dim)
        self.emb = self.nnlm.get_emb()
        self.proj = self.nnlm.get_proj()

    def forward(self, x: torch.Tensor):
        _x = self.emb(x)
        _x = self.proj(_x)
        return _x.sum(axis=1)


def compute_topk_similar(
    word_emb: torch.Tensor, w2v_emb_weight: torch.Tensor, k
) -> list:
    cos = torch.nn.CosineSimilarity(dim=1)
    sims = cos(w2v_emb_weight, word_emb)

    # drop the first item because it is the word itself
    return torch.topk(sims.flatten(), k + 1).indices.tolist()[1:]


@torch.no_grad()
def retrieve_similar_words(
    model: nn.Module,
    word: str,
    index_map: "dict[str, int]",
    index_to_word: "dict[int, str]",
    k: int = 5,
) -> "list[str]":
    word_idx = index_map[word]

    # get the weights from the embedding
    weight = model.emb.weight

    word_emb = weight[word_idx, :]

    similar = compute_topk_similar(word_emb, weight, k)

    return [index_to_word[i] for i in similar]


@torch.no_grad()
def word_analogy(
    model: nn.Module,
    word_a: str,
    word_b: str,
    word_c: str,
    index_map: "dict[str, int]",
    index_to_word: "dict[int, str]",
    k: int = 5,
) -> "list[str]":
    # get the weights from the embedding
    model.eval()

    a = model.emb.weight[index_map[word_a]]
    b = model.emb.weight[index_map[word_b]]
    c = model.emb.weight[index_map[word_c]]
    d = a - b + c

    similar = compute_topk_similar(d, model.emb.weight, k)
    return [index_to_word[i] for i in similar]


# ######################## PART 2: YOUR WORK STARTS HERE ########################


def compute_gender_subspace(
    word_to_embedding: "dict[str, np.array]",
    gender_attribute_words: "list[tuple[str, str]]",
    n_components: int = 1,
) -> np.array:

    data = []
    for word_a, word_b in gender_attribute_words:
        mean = (word_to_embedding[word_a] + word_to_embedding[word_b]) / 2
        embed_a = word_to_embedding[word_a] - mean
        embed_b = word_to_embedding[word_b] - mean

        # add a + b
        data.append(embed_a)
        data.append(embed_b)

    data = np.stack(data, axis=0)
    pca = PCA(n_components=n_components)
    pca.fit(data)
    return pca.components_


def project(a: np.array, b: np.array) -> "tuple[float, np.array]":
    dot_product = np.dot(a, b)
    magnitude_squared = np.dot(b, b)
    scale = dot_product / magnitude_squared
    projection = scale * b
    return scale, projection


def compute_profession_embeddings(
    word_to_embedding: "dict[str, np.array]", professions: "list[str]"
) -> "dict[str, np.array]":
    profession_embeddings = {}
    for profession in professions:
        embed = []
        for sub in profession.split():
            embed.append(word_to_embedding[sub])
        embed = np.stack(embed, axis=0).mean(axis=0)
        profession_embeddings[profession] = embed
    return profession_embeddings


def compute_extreme_words(
    words: "list[str]",
    word_to_embedding: "dict[str, np.array]",
    gender_subspace: np.array,
    k: int = 10,
    max_: bool = True,
) -> "list[str]":

    scales = []
    for word in words:
        embed = word_to_embedding[word]
        scale, _ = project(embed, np.squeeze(gender_subspace))
        scales.append((word, scale))
    scales.sort(key=lambda x: x[1], reverse=max_)
    final_words = [i for i, _ in scales]
    return final_words[:k]


def cosine_similarity(a: np.array, b: np.array) -> float:
    dot_product = np.dot(a, b)
    norm_A = np.linalg.norm(a)
    norm_B = np.linalg.norm(b)
    similarity = dot_product / (norm_A * norm_B)
    return similarity


def compute_direct_bias(
    words: "list[str]",
    word_to_embedding: "dict[str, np.array]",
    gender_subspace: np.array,
    c: float = 0.25,
):
    N = len(words)
    gender_subspace = np.squeeze(gender_subspace)

    similars = []
    for word in words:
        embedding = word_to_embedding[word]
        sim = cosine_similarity(embedding, gender_subspace)
        similars.append(sim)

    direct_sum = np.sum(np.abs(similars) ** c)
    return direct_sum / N


def weat_association(
    w: str, A: "list[str]", B: "list[str]", word_to_embedding: "dict[str, np.array]"
) -> float:

    emb_w = word_to_embedding[w]

    a_sims = []
    for a in A:
        emb_a = word_to_embedding[a]
        sim = cosine_similarity(emb_w, emb_a)
        a_sims.append(sim)
    a_sims = np.stack(a_sims, axis=0).mean(axis=0)

    b_sims = []
    for b in B:
        emb_b = word_to_embedding[b]
        sim = cosine_similarity(emb_w, emb_b)
        b_sims.append(sim)
    b_sims = np.stack(b_sims, axis=0).mean(axis=0)
    return a_sims - b_sims


def weat_differential_association(
    X: "list[str]",
    Y: "list[str]",
    A: "list[str]",
    B: "list[str]",
    word_to_embedding: "dict[str, np.array]",
    weat_association_func: Callable,
) -> float:

    x_weats = []
    for x in X:
        weat = weat_association_func(x, A, B, word_to_embedding)
        x_weats.append(weat)
    x_sum = np.sum(x_weats)

    y_weats = []
    for y in Y:
        weat = weat_association_func(y, A, B, word_to_embedding)
        y_weats.append(weat)
    y_sum = np.sum(y_weats)
    return x_sum - y_sum


def debias_word_embedding(
    word: str, word_to_embedding: "dict[str, np.array]", gender_subspace: np.array
) -> np.array:
    word_emb = word_to_embedding[word]
    _, vector = project(word_emb, np.squeeze(gender_subspace))
    return word_emb - vector


def hard_debias(
    word_to_embedding: "dict[str, np.array]",
    gender_attribute_words: "list[str]",
    n_components: int = 1,
) -> "dict[str, np.array]":
    gender_subspace = compute_gender_subspace(
        word_to_embedding, gender_attribute_words, n_components
    )

    debiased_embeddings = {}
    for word in word_to_embedding:
        embedding = debias_word_embedding(word, word_to_embedding, gender_subspace)
        debiased_embeddings[word] = embedding
    return debiased_embeddings


if __name__ == "__main__":
    random.seed(2022)
    torch.manual_seed(2022)
    """
    # Parameters (you can change them)
    sample_size = 250  # Change this if you want to take a subset of data for testing
    batch_size = 64
    n_epochs = 2
    num_words = 50000

    # Load the data
    #data_path = "../input/a1-data"  # Use this for kaggle
    data_path = "data"  # Use this if running locally

    # If you use GPUs, use the code below:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ###################### PART 1: TEST CODE ######################
    print("=" * 80)
    print("Running test code for part 1")
    print("-" * 80)

    # Prefilled code showing you how to use the helper functions
    train_raw, valid_raw = load_datasets(data_path)
    if sample_size is not None:
        for key in ["premise", "hypothesis", "label"]:
            train_raw[key] = train_raw[key][:sample_size]
            valid_raw[key] = valid_raw[key][:sample_size]

    full_text = (
        train_raw["premise"]
        + train_raw["hypothesis"]
        + valid_raw["premise"]
        + valid_raw["hypothesis"]
    )

    # Process into indices
    tokens = tokenize_w2v(full_text)

    word_counts = build_word_counts(tokens)
    word_to_index = build_index_map(word_counts, max_words=num_words)
    index_to_word = {v: k for k, v in word_to_index.items()}

    text_indices = tokens_to_ix(tokens, word_to_index)

    # Train CBOW
    sources_cb, targets_cb = cbow_preprocessing(text_indices, window_size=2)
    loader_cb = DataLoader(
        Word2VecDataset(sources_cb, targets_cb),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_cbow,
    )

    model_cb = CBOW(num_words=len(word_to_index), embed_dim=200).to(device)
    optimizer = torch.optim.Adam(model_cb.parameters())

    for epoch in range(n_epochs):
        loss = train_w2v(model_cb, optimizer, loader_cb, device=device).item()
        print(f"Loss at epoch #{epoch}: {loss:.4f}")

    # Training Skip-Gram

    # TODO: your work here
    model_sg = SkipGram(num_words=len(word_to_index), embed_dim=200).to(device)

    # RETRIEVE SIMILAR WORDS
    word = "man"

    similar_words_cb = retrieve_similar_words(
        model=model_cb,
        word=word,
        index_map=word_to_index,
        index_to_word=index_to_word,
        k=5,
    )

    similar_words_sg = retrieve_similar_words(
        model=model_sg,
        word=word,
        index_map=word_to_index,
        index_to_word=index_to_word,
        k=5,
    )

    print(f"(CBOW) Words similar to '{word}' are: {similar_words_cb}")
    print(f"(Skip-gram) Words similar to '{word}' are: {similar_words_sg}")

    # COMPUTE WORDS ANALOGIES
    a = "man"
    b = "woman"
    c = "girl"

    analogies_cb = word_analogy(
        model=model_cb,
        word_a=a,
        word_b=b,
        word_c=c,
        index_map=word_to_index,
        index_to_word=index_to_word,
    )
    analogies_sg = word_analogy(
        model=model_sg,
        word_a=a,
        word_b=b,
        word_c=c,
        index_map=word_to_index,
        index_to_word=index_to_word,
    )

    print(f"CBOW's analogies for {a} - {b} + {c} are: {analogies_cb}")
    print(f"Skip-gram's analogies for {a} - {b} + {c} are: {analogies_sg}")
    """
    # ###################### PART 1: TEST CODE ######################

    # Prefilled code showing you how to use the helper functions
    word_to_embedding = load_glove_embeddings("data/glove/glove.6B.300d.txt")

    professions = load_professions("data/professions.tsv")

    gender_attribute_words = load_gender_attribute_words(
        "data/gender_attribute_words.json"
    )

    # === Section 2.1 ===
    gender_subspace = compute_gender_subspace(
        word_to_embedding, gender_attribute_words, 1
    )

    # === Section 2.2 ===
    a = word_to_embedding["man"]
    b = word_to_embedding["boy"]
    scalar_projection, vector_projection = project(a, b)

    # === Section 2.3 ===
    profession_to_embedding = compute_profession_embeddings(
        word_to_embedding, professions
    )

    # === Section 2.4 ===
    positive_profession_words = compute_extreme_words(
        professions, profession_to_embedding, gender_subspace, max_=True
    )
    negative_profession_words = compute_extreme_words(
        professions, profession_to_embedding, gender_subspace, max_=False
    )

    print(f"Max profession words: {positive_profession_words}")
    print(f"Min profession words: {negative_profession_words}")

    # === Section 2.5 ===
    direct_bias_professions = compute_direct_bias(
        professions, profession_to_embedding, gender_subspace
    )

    # === Section 2.6 ===

    # Prepare attribute word sets for testing
    A = ["male", "man", "boy", "brother", "he", "him", "his", "son"]
    B = ["female", "woman", "girl", "sister", "she", "her", "hers", "daughter"]

    # Prepare target word sets for testing
    X = ["doctor", "mechanic", "engineer"]
    Y = ["nurse", "artist", "teacher"]

    word = "doctor"
    weat_association = weat_association
    _weat_differential_association = weat_differential_association(
        X, Y, A, B, word_to_embedding, weat_association
    )

    # === Section 3.1 ===
    debiased_word_to_embedding = debias_word_embedding(
        word, word_to_embedding, gender_subspace
    )
    #debiased_profession_to_embedding = hard_debias(
    #    profession_to_embedding, gender_attribute_words, 1
    #)

    # === Section 3.2 ===
    debiased_profession_to_embedding = {}
    for word in profession_to_embedding:
        embedding = debias_word_embedding(word, profession_to_embedding, gender_subspace)
        debiased_profession_to_embedding[word] = embedding

    direct_bias_professions_biased = compute_direct_bias(professions,profession_to_embedding,gender_subspace)
    direct_bias_professions_debiased = compute_direct_bias(professions,debiased_profession_to_embedding,gender_subspace)

    print(f"DirectBias Professions (biased): {direct_bias_professions_biased:.2f}")
    print(f"DirectBias Professions (debiased): {direct_bias_professions_debiased:.2f}")

    X = [
        "math",
        "algebra",
        "geometry",
        "calculus",
        "equations",
        "computation",
        "numbers",
        "addition",
    ]

    Y = [
        "poetry",
        "art",
        "dance",
        "literature",
        "novel",
        "symphony",
        "drama",
        "sculpture",
    ]

    debiased_word_to_embedding = hard_debias(
        word_to_embedding, gender_attribute_words, 1
    )

    #p_value_biased = p_value_permutation_test(X,Y,A,B,word_to_embedding)
    #p_value_debiased = p_value_permutation_test(X,Y,A,B,debiased_word_to_embedding)

    #print(f"p-value biased: {p_value_biased:.2f}")
    #print(f"p-value debiased: {p_value_debiased:.2f}")

    from sklearn.manifold import TSNE
    import seaborn as sns
    import pandas as pd

    max_words = compute_extreme_words(
    words=professions,
    word_to_embedding=profession_to_embedding,
    gender_subspace=gender_subspace,
    k=10,
    max_=True
    )

    min_words = compute_extreme_words(
    words=professions,
    word_to_embedding=profession_to_embedding,
    gender_subspace=gender_subspace,
    k=10,
    max_=False
    )

    embeddings = []
    for word in max_words+min_words:
        embeddings.append(profession_to_embedding[word])
    embeddings = np.stack(embeddings,axis=0)

    X_embedded = TSNE(n_components=2, learning_rate='auto',
                      init='random', perplexity=3).fit_transform(embeddings)
    
    data = pd.DataFrame({
        '1st Dimension':list(X_embedded[:,0]),
        '2nd Dimension':list(X_embedded[:,1]),
        'Direction':['max']*10 + ['min']*10

    })

    sns.set_theme()
    plot = sns.scatterplot(data, x='1st Dimension', y='2nd Dimension',hue='Direction')
    fig = plot.get_figure()
    fig.savefig("tsne_gender_bias.png") 