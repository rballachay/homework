import numpy as np


def nw_extension(
    genome: str,
    query: str,
    probs: np.ndarray,
    ungapped_hsps: dict,
    gap_penalty: float,
):
    for seed, pairs in ungapped_hsps.items():
        print(f"Gapped extension of seed {seed}")
        for i_query, i_genome, i_len in pairs:
            genome_sub = genome[i_genome : i_genome + i_len]
            query_sub = query[i_query : i_query + i_len]
            pairs = needleman_wunsch(genome_sub, query_sub, probs, gap_penalty)


def scoring_function(prob: float):
    score = np.emath.logn(4, prob) + 1
    if score < -1:
        score = -1
    return score


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
