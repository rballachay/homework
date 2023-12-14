import numpy as np
import itertools
from utils import cachewrapper


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


def smith_waterman(a: str, b: str, probs: np.ndarray, gap_cost=2):
    def matrix(a, b, probs, gap_cost=2):
        H = np.zeros((len(a) + 1, len(b) + 1), np.float)

        for i, j in itertools.product(range(1, H.shape[0]), range(1, H.shape[1])):
            if a[i - 1] == b[j - 1]:
                prob = probs[i - 1]
            else:
                prob = (1 - probs[i - 1]) / 3

            score = scoring_function(prob)
            match = H[i - 1, j - 1] + score
            delete = H[i - 1, j] - gap_cost
            insert = H[i, j - 1] - gap_cost
            H[i, j] = max(match, delete, insert)
        return H

    def traceback(H, b, b_="", old_i=0):
        # flip H to get index of **last** occurrence of H.max() with np.argmax()
        H_flip = np.flip(np.flip(H, 0), 1)
        i_, j_ = np.unravel_index(H_flip.argmax(), H_flip.shape)
        i, j = np.subtract(
            H.shape, (i_ + 1, j_ + 1)
        )  # (i, j) are **last** indexes of H.max()
        if H[i, j] == 0:
            return b_, j
        b_ = b[j - 1] + "-" + b_ if old_i - i > 1 else b[j - 1] + b_
        return traceback(H[0:i, 0:j], b, b_, i)

    a, b = a.upper(), b.upper()
    H = matrix(a, b, probs, gap_cost)
    max_score = H.max()
    b_, pos = traceback(H, b)
    a_ = a[pos : pos + len(b_)]
    return "\n".join([a_, b_]), max_score


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
                start_idx = i_genome - search_distance - i_query
                end_idx = i_genome + i_len + search_distance
                genome_sub = genome[start_idx:end_idx]
                probs_sub = probs[start_idx:end_idx]
                query_sub = query
                pairs, score = smith_waterman(
                    genome_sub, query_sub, probs_sub, gap_penalty
                )

            results[seed] = {
                "score": score,
                "aligned": pairs.split("\n"),
                "seqs": [genome_sub, query_sub],
                "probs": list(probs_sub),
            }

    return results


def scoring_function(prob: float):
    if prob == 0:
        prob = 1e-10
    score = np.emath.logn(4, prob) + 1
    return score
