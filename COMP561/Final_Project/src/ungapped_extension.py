import math
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from src.utils import scoring_function


class UngappedPlotter:
    def __init__(self):
        self.scores = []
        self.seqs = []
        self.locs = []
        self.seed_len = None

    def new(self):
        self.scores.append([])
        self.locs.append([])

    def forward(self, score, idx):
        self.scores[-1].append(score)
        self.locs[-1].append(idx)

    def backward(self, score, idx):
        self.scores[-1].insert(0, score)
        self.locs[-1].insert(0, idx)

    def add_seq(self, seqs):
        self.seqs.append(seqs)

    def plot(self):
        _lens = np.array([len(i) for i in self.scores])
        max_i = np.random.choice(np.flatnonzero(_lens == _lens.max()))
        scores = np.array(self.scores[max_i])
        genome, query = self.seqs[max_i]
        locs = self.locs[max_i]
        plt.figure(dpi=200)
        plt.plot(locs, scores, "k")
        plt.xlabel("Position in Query Sequence (L=1000)")
        plt.ylabel("Ungapped Sequence Match Score")
        # plt.savefig("results/ungapped_alignment.png")


# @cachewrapper("results/ungapped_high_scoring_pairs.json")
def get_ungapped_hsps(
    genome: str,
    seeds: dict,
    probabilities: np.ndarray,
    query_sequence: str,
    scoring_func: callable,
):
    hsp_dict = {}
    plotter = UngappedPlotter()
    for seed, positions in seeds.items():
        for pos in positions:
            plotter.new()
            query_pos, genome_pos = pos
            score = 0

            seed_score = []
            for i in range(len(seed)):
                nucleotide = seed[i]
                p = (
                    probabilities[genome_pos + i]
                    if genome[genome_pos + i] == nucleotide
                    else (1 - probabilities[genome_pos + i]) / 3
                )
                score += scoring_function(p)
                seed_score.append(score)

            for i in range(len(seed)):
                plotter.forward(np.average(seed_score), query_pos + i)

            seed_score = copy(score)
            hsp = seed
            temp_hsp = seed
            max_score = score
            dropoff_threshold = 30
            final_scores = []

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
                score += scoring_function(p)
                plotter.forward(score, i)

                if (max_score - score) < 0:
                    if temp_hsp != hsp:
                        hsp = temp_hsp
                    max_score = score
                    hsp += query_sequence[i]
                    temp_hsp += query_sequence[i]
                elif (max_score - score) >= 0 and (
                    max_score - score
                ) <= dropoff_threshold:
                    temp_hsp += query_sequence[i]
                elif (max_score - score) > dropoff_threshold:
                    break

            final_scores.append(max_score)
            score = seed_score
            max_score = seed_score
            temp_hsp = hsp

            # Extend left
            hsp_start = query_pos
            for i in range(query_pos - 1, -1, -1):
                # print(query_sequence[i],genome[genome_pos + i - query_pos])
                p = (
                    probabilities[genome_pos + i - query_pos]
                    if query_sequence[i] == genome[genome_pos + i - query_pos]
                    else (1 - probabilities[genome_pos + i - query_pos]) / 3
                )
                score += scoring_function(p)
                plotter.backward(score, i)

                if (max_score - score) < 0:
                    if hsp != temp_hsp:
                        hsp = temp_hsp
                    max_score = score
                    hsp = query_sequence[i] + hsp
                    temp_hsp = query_sequence[i] + temp_hsp
                    hsp_start = i
                elif (max_score - score) >= 0 and (
                    max_score - score
                ) <= dropoff_threshold:
                    temp_hsp = query_sequence[i] + temp_hsp
                elif (max_score - score) > dropoff_threshold:
                    break

            final_scores.append(max_score)
            gs_ = int(genome_pos - (query_pos - hsp_start))
            plotter.add_seq(
                (
                    genome[gs_ : gs_ + len(hsp)],
                    query_sequence[hsp_start : hsp_start + len(hsp)],
                )
            )

            # don't consider high scoring pairs that are under our
            # designated score thresold
            if not scoring_func(len(hsp), len(hsp), max(final_scores)):
                continue

            if hsp not in hsp_dict:
                hsp_dict[hsp] = []

            occurrence = (hsp_start, gs_, len(hsp))
            if occurrence not in hsp_dict[hsp]:
                hsp_dict[hsp].append(occurrence)

    return hsp_dict, plotter
