import math
import numpy as np
from utils import cachewrapper


@cachewrapper("results/ungapped_high_scoring_pairs.json")
def get_ungapped_hsps(
    genome: str,
    seeds: dict,
    probabilities: np.ndarray,
    query_sequence: str,
    min_len_hsp: int,
):
    hsp_dict = {}

    for seed, positions in seeds.items():
        for pos in positions:
            query_pos, genome_pos = pos
            score = 0
            for i in range(len(seed)):
                nucleotide = seed[i]
                p = (
                    probabilities[genome_pos + i]
                    if genome[genome_pos + i] == nucleotide
                    else (1 - probabilities[genome_pos + i]) / 3
                )
                if p == 0:
                    p += 1e-10
                    score += math.log(p, 4) + 1

            hsp = seed
            max_score = score
            dropoff_threshold = 30

            # Extend right
            for i in range(query_pos + len(seed), len(query_sequence)):
                idx_genome = genome_pos + i - query_pos
                if idx_genome == len(genome):
                    print("Reached end of genome, terminating right extension")
                    break
                p = (
                    probabilities[idx_genome]
                    if query_sequence[i] == genome[idx_genome]
                    else (1 - probabilities[idx_genome]) / 3
                )
                if p == 0:
                    p += 1e-10
                score += math.log(p, 4) + 1
                if p == 0:
                    print("we have zero prob")
                p += 1e-10
                if score > max_score:
                    max_score = score
                    hsp += query_sequence[i]
                elif max_score - score > dropoff_threshold:
                    break

            score = max_score

            # Extend left
            hsp_start = query_pos
            for i in range(query_pos - 1, -1, -1):
                p = (
                    probabilities[genome_pos + i - query_pos]
                    if query_sequence[i] == genome[genome_pos + i - query_pos]
                    else (1 - probabilities[genome_pos + i - query_pos]) / 3
                )
                if p == 0:
                    p += 1e-10
                score += math.log(p, 4) + 1
                if score > max_score:
                    max_score = score
                    hsp = query_sequence[i] + hsp
                    hsp_start = i
                elif (max_score - score) > dropoff_threshold:
                    break

            # don't consider high scoring pairs that are under length
            # threshold - waste of time
            if len(hsp) < min_len_hsp:
                continue

            if seed not in hsp_dict:
                hsp_dict[seed] = []
            occurrence = (hsp_start, genome_pos - (query_pos - hsp_start), len(hsp))
            if occurrence not in hsp_dict[seed]:
                hsp_dict[seed].append(occurrence)

    return hsp_dict
