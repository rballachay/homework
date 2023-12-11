"""
COMP561 Proejct - Finding Seed
"""

import random
import numpy as np
import json


def get_query(
    genome: str,
    confidence_values: np.ndarray,
    query_length: int,
    nucleotides: list,
    seed: int,
):
    random.seed(seed)
    query_sequence = ""
    start_pos = random.randint(0, len(genome) - query_length)

    for i in range(start_pos, start_pos + query_length):
        nucleotide_probs = [
            confidence_values[i] if n == genome[i] else (1 - confidence_values[i]) / 3
            for n in nucleotides
        ]
        query_sequence += np.random.choice(nucleotides, p=nucleotide_probs)
    return query_sequence


def get_genome_dict(genome: str, window_size: int):
    genome_dict = {}

    for i in range(len(genome) - window_size + 1):
        subseq = "".join(genome[i : i + window_size])
        if subseq not in genome_dict:
            genome_dict[subseq] = []
        genome_dict[subseq].append(i)
    return genome_dict


def get_query_dict(
    genome: str,
    genome_dict: str,
    query_sequence: str,
    confidence_values: np.ndarray,
    window_size: int,
    threshold: float,
):
    seeds = {}

    for j in range(len(query_sequence) - window_size + 1):
        query_subseq = query_sequence[j : j + window_size]
        if query_subseq in genome_dict:
            for i in genome_dict[query_subseq]:
                joint_prob = np.prod(
                    [
                        confidence_values[k]
                        if genome[k] == query_sequence[j + k - i]
                        else (1 - confidence_values[k]) / 3
                        for k in range(i, i + window_size)
                    ]
                )
                if joint_prob > threshold:
                    if query_subseq not in seeds:
                        seeds[query_subseq] = []
                    seeds[query_subseq].append((j, i))

    return seeds
