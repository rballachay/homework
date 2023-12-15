import numpy as np
import itertools
from utils import cachewrapper
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
from enum import IntEnum


class AlignmentPlotter:
    def __init__(self):
        self.first = True

    def run(self, H, path, a, b):
        if not self.first:
            return

        self.first = False
        self.H = H
        self.path = path
        self.a = [" "] + list(a)
        self.b = [" "] + list(b)

    def plot(self):
        fig, ax = plt.subplots(figsize=(20, 20))
        data = pd.DataFrame(self.H, index=self.a, columns=self.b)

        data_copy = data.copy()
        vals = data_copy.values
        vals[:, :] = 0
        for i, j in self.path:
            vals[i, j] = 1
        data_copy.loc[:] = vals

        sns.heatmap(data, ax=ax, cbar=False)
        cmap1 = mpl.colors.ListedColormap(["c"])
        sns.heatmap(
            data_copy,
            mask=vals == 0,
            cmap=cmap1,
            cbar=False,
            ax=ax,
        )
        fig.savefig("smith_waterman_alignment.png", dpi=100)


def needleman_wunsch(x: str, y: str, probs: np.ndarray, gap=1):
    nx = len(x)
    ny = len(y)
    # Optimal score at each possible pair of characters.
    F = np.zeros((nx + 1, ny + 1))
    F[:, 0] = np.linspace(0, -nx * gap, nx + 1)
    F[0, :] = np.linspace(0, -ny * gap, ny + 1)
    # Pointers to trace through an optimal aligment.
    P = np.zeros((nx + 1, ny + 1))
    P[:, 0] = 3
    P[0, :] = 4
    # Temporary scores.
    t = np.zeros(3)
    for i in range(nx):
        for j in range(ny):
            # if we have a matching base, we can use the probability of the base from x, otherwise
            # use the non-dominant base
            prob = probs[i]
            if x[i] != y[j]:
                prob = (1 - prob) / 3

            score = scoring_function(prob)
            t[0] = F[i, j] + score
            t[1] = F[i, j + 1] - gap
            t[2] = F[i + 1, j] - gap
            tmax = np.max(t)
            F[i + 1, j + 1] = tmax
            if t[0] == tmax:
                P[i + 1, j + 1] += 2
            if t[1] == tmax:
                P[i + 1, j + 1] += 3
            if t[2] == tmax:
                P[i + 1, j + 1] += 4
    # Trace through an optimal alignment.
    i = nx
    j = ny
    rx = []
    ry = []
    while i > 0 or j > 0:
        if P[i, j] in [2, 5, 6, 9]:
            rx.append(x[i - 1])
            ry.append(y[j - 1])
            i -= 1
            j -= 1
        elif P[i, j] in [3, 5, 7, 9]:
            rx.append(x[i - 1])
            ry.append("-")
            i -= 1
        elif P[i, j] in [4, 6, 7, 9]:
            rx.append("-")
            ry.append(y[j - 1])
            j -= 1
    # Reverse the strings.
    rx = "".join(rx)[::-1]
    ry = "".join(ry)[::-1]
    return "\n".join([rx, ry])


def smith_waterman(seq1, seq2, probs, gap=2):
    # Assigning the constant values for the traceback
    class Trace(IntEnum):
        STOP = 0
        LEFT = 1
        UP = 2
        DIAGONAL = 3

    # Generating the empty matrices for storing scores and tracing
    row = len(seq1) + 1
    col = len(seq2) + 1
    matrix = np.zeros(shape=(row, col), dtype=np.float)
    tracing_matrix = np.zeros(shape=(row, col), dtype=np.int)

    # Initialising the variables to find the highest scoring cell
    max_score = -np.inf
    max_index = (-1, -1)

    # Calculating the scores for all cells in the matrix
    for i in range(1, row):
        for j in range(1, col):
            # Calculating the diagonal score (match score)
            if seq1[i - 1] == seq2[j - 1]:
                prob = probs[i - 1]
            else:
                prob = (1 - probs[i - 1]) / 3

            diagonal_score = matrix[i - 1, j - 1] + scoring_function(prob)

            # Calculating the vertical gap score
            vertical_score = matrix[i - 1, j] - gap

            # Calculating the horizontal gap score
            horizontal_score = matrix[i, j - 1] - gap

            # Taking the highest score
            matrix[i, j] = max(0, diagonal_score, vertical_score, horizontal_score)

            # Tracking where the cell's value is coming from
            if matrix[i, j] == 0:
                tracing_matrix[i, j] = Trace.STOP

            elif matrix[i, j] == horizontal_score:
                tracing_matrix[i, j] = Trace.LEFT

            elif matrix[i, j] == vertical_score:
                tracing_matrix[i, j] = Trace.UP

            elif matrix[i, j] == diagonal_score:
                tracing_matrix[i, j] = Trace.DIAGONAL

            # Tracking the cell with the maximum score
            if matrix[i, j] >= max_score:
                max_index = (i, j)
                max_score = matrix[i, j]

    # Initialising the variables for tracing
    aligned_seq1 = ""
    aligned_seq2 = ""
    current_aligned_seq1 = ""
    current_aligned_seq2 = ""
    (max_i, max_j) = max_index

    traceback_path = []
    # Tracing and computing the pathway with the local alignment
    while tracing_matrix[max_i, max_j] != Trace.STOP:
        traceback_path.append((max_i, max_j))
        if tracing_matrix[max_i, max_j] == Trace.DIAGONAL:
            current_aligned_seq1 = seq1[max_i - 1]
            current_aligned_seq2 = seq2[max_j - 1]
            max_i = max_i - 1
            max_j = max_j - 1

        elif tracing_matrix[max_i, max_j] == Trace.UP:
            current_aligned_seq1 = seq1[max_i - 1]
            current_aligned_seq2 = "-"
            max_i = max_i - 1

        elif tracing_matrix[max_i, max_j] == Trace.LEFT:
            current_aligned_seq1 = "-"
            current_aligned_seq2 = seq2[max_j - 1]
            max_j = max_j - 1

        aligned_seq1 = aligned_seq1 + current_aligned_seq1
        aligned_seq2 = aligned_seq2 + current_aligned_seq2

    # Reversing the order of the sequences
    aligned_seq1 = aligned_seq1[::-1]
    aligned_seq2 = aligned_seq2[::-1]

    return aligned_seq1, aligned_seq2, matrix, traceback_path


# @cachewrapper("results/full_sequence_alignment.json")
def align_sequences(
    genome: str,
    query: str,
    probs: np.ndarray,
    ungapped_hsps: dict,
    gap_penalty: float,
    search_distance: int,
    alignment_method: str = "local",
):
    results = {}
    #plotter = AlignmentPlotter()
    for seed, pairs in ungapped_hsps.items():
        for i_query, i_genome, i_len in pairs:

            if alignment_method == "global":
                genome_sub = genome[i_genome : i_genome + i_len]
                probs_sub = probs[i_genome : i_genome + i_len]
                query_sub = query[i_query : i_query + i_len]
                pairs, score = needleman_wunsch(
                    genome_sub, query_sub, probs, gap_penalty
                )
            else:
                # search from 50 bases before the match to 50 bases after
                query_start_on_genome = i_genome - i_query
                start_idx = query_start_on_genome - search_distance
                end_idx = query_start_on_genome + len(query) + search_distance
                genome_sub = genome[start_idx:end_idx]
                probs_sub = probs[start_idx:end_idx]
                query_sub = query
                pairs, score, H, traceback_path = smith_waterman(
                    genome_sub, query_sub, probs_sub, gap_penalty
                )
                #plotter.run(H, traceback_path, genome_sub, query_sub)

            results[seed] = {
                "score": score,
                "aligned": pairs.split("\n"),
                "seqs": [genome_sub, query_sub],
                "probs": list(probs_sub),
            }

            # only want to run plotter once
            #plotter.plot()

    return results


def scoring_function(prob: float):
    if prob == 0:
        prob = 1e-10
    score = np.emath.logn(4, prob) + 1
    return score
