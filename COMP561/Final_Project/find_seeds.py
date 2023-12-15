"""
COMP561 Proejct - Finding Seed
"""

import random
import numpy as np
from collections import defaultdict
from utils import cachewrapper


def get_query(
    genome: str,
    confidence_values: np.ndarray,
    query_length: int,
    nucleotides: list,
    seed: int,
):
    # random.seed(seed)
    query_sequence = ""
    start_pos = random.randint(0, len(genome) - query_length)

    for i in range(start_pos, start_pos + query_length):
        nucleotide_probs = [
            confidence_values[i] if n == genome[i] else (1 - confidence_values[i]) / 3
            for n in nucleotides
        ]
        query_sequence += np.random.choice(nucleotides, p=nucleotide_probs)

    # Randomly insert 1/50nucleotides
    for _ in range(random.randint(1, int(query_length / 10))):
        insert_pos = random.randint(0, len(query_sequence))
        query_sequence = (
            query_sequence[:insert_pos]
            + random.choice(nucleotides)
            + query_sequence[insert_pos:]
        )

    # Randomly delete 1/50 nucleotides
    for _ in range(random.randint(1, int(query_length / 10))):
        if len(query_sequence) == 0:
            break
        del_pos = random.randint(0, len(query_sequence) - 1)
        query_sequence = query_sequence[:del_pos] + query_sequence[del_pos + 1 :]
    return query_sequence


@cachewrapper("results/genome_seeds.json")
def get_genome_dict(
    genome: str, probabilities: np.ndarray, window_size: int, nucs: list, thres: float
):
    table = np.zeros((len(genome), len(nucs)))
    mapping = {"A": 0, "T": 1, "C": 2, "G": 3}
    genome_idx = np.array([mapping[chr] for chr in genome])
    x_axis = np.arange(len(genome))

    probabilities[probabilities == 1] = 1 - 1e-10
    table[x_axis, genome_idx] = probabilities

    other_prob = (1 - probabilities) / 3
    first_col = genome_idx != 0
    table[first_col, 0] = other_prob[first_col]
    second_col = genome_idx != 1
    table[second_col, 1] = other_prob[second_col]

    third_col = genome_idx != 2
    table[third_col, 2] = other_prob[third_col]

    fourth_col = genome_idx != 3
    table[fourth_col, 3] = other_prob[fourth_col]

    seeds = defaultdict(list)

    cur_result = table[0]
    for i in range(1, window_size):
        cur_result = cur_result[..., None]

        dim = tuple([1] * i + [4])
        prob1 = np.reshape(table[i], dim)
        cur_result = cur_result @ prob1

    record = cur_result > thres

    indices = np.nonzero(record)
    all_indices = np.transpose(indices)
    for j in range(all_indices.shape[0]):
        seeds[tuple(all_indices[j])].append(0)

    length = table.shape[0]
    for i in range(window_size, length):
        dim = tuple([4] + [1] * (window_size - 1))
        div = np.reshape(table[i - 11], dim)
        cur_result = (cur_result / div)[0]

        cur_result = cur_result[..., None]
        dim = tuple([1] * 10 + [4])
        prob1 = np.reshape(table[i], dim)
        cur_result = cur_result @ prob1

        record = cur_result > thres
        indices = np.nonzero(record)
        all_indices = np.transpose(indices)

        for j in all_indices:
            seeds[tuple(j)].append(i - 10)

    new_seeds = dict()
    mapping = {"A": 0, "T": 1, "C": 2, "G": 3}
    rev_mapping = {mapping[i]: i for i in mapping}
    count = 0
    for i in seeds:
        seq = "".join([rev_mapping[j] for j in i])
        new_seeds[seq] = seeds[i]
        count += len(seeds[i])

    return new_seeds


def get_query_dict(
    genome_dict: str,
    query_sequence: str,
    window_size: int,
):
    seeds = {}

    for i in range(len(query_sequence) - window_size + 1):
        window = query_sequence[i : i + window_size]
        if window in genome_dict:
            if window not in seeds:
                seeds[window] = []
            for j in genome_dict[window]:
                seeds[window].append((i, j))

    return seeds
