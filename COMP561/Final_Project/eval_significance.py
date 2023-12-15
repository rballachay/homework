import pandas as pd
import numpy as np
import random
from difflib import SequenceMatcher
from blast import BLAST
from sequence_alignment import smith_waterman
import os
import scipy.stats as ss


class GumbelSignificance:
    def __init__(self, blast: BLAST, genome: str, probabilities: np.ndarray):
        self.blast = blast
        self.genome = genome
        self.probabilities = probabilities

        self._gumbel_parameters = None

        # this will only be set in the case that the monte
        # carlo simulation is run
        self._monte_carlo_results = None
        self.monte_carlo_path = "results/monte_carlo_results.csv"

        # here are the lengths we are going to use to evaluate
        # our gumbel parameters
        self.lens = [50, 100, 250, 500, 1000]

        # this is the number of simulations we are going to run
        # for each of our lengths in order to estimate mean of gumbel
        self.n_sims = 100

    def add_significance(self, results):
        gumbel_parameters = self.gumbel_parameters

        # mu = log(K*m*n)/sigma
        for key, content in results.items():
            seqs = content["seqs"]
            m_n = len(seqs[0]) * len(seqs[1])
            loc = np.log(gumbel_parameters["K"] * m_n) / gumbel_parameters["sigma"]
            scale = gumbel_parameters["sigma"]
            dist = ss.gumbel_r(loc=loc, scale=scale)
            content["p-value"] = dist.pdf(content["score"])

    @property
    def gumbel_parameters(self):
        """Gumbel parameters need to be created before any of this is done"""
        if self._gumbel_parameters is None:
            self._gumbel_parameters = self.__get_gumbel_parameters()
        return self._gumbel_parameters

    def __get_gumbel_parameters(
        self,
    ):
        monte_carlo_results = self.__run_monte_carlo_simulation()
        y = []
        x = []
        sigmas = []
        for (m, n), _monte_carlo_results in monte_carlo_results.groupby(["m", "n"]):
            loc, scale = ss.gumbel_r.fit(_monte_carlo_results["score"].values)
            y.append(np.exp(loc * scale))
            x.append(m * n)
            sigmas.append(scale)

        K, _, _, _ = np.linalg.lstsq(
            np.array(x).reshape(len(x), -1), np.array(y).reshape(len(y), -1)
        )
        sigma = np.average(sigmas)
        K = K[0][0]
        return {"K": K, "sigma": sigma}

    def __run_monte_carlo_simulation(self):
        if os.path.exists(self.monte_carlo_path):
            return pd.read_csv(self.monte_carlo_path)
        # initialize the monte carlo results
        self._monte_carlo_results = []
        for query_len in self.lens:
            # run blast to get the match of what we would expect for the sequence
            results = self.blast.run(self.genome, self.probabilities, query_len, False)

            # pick the max score
            max_score_content = [
                v for _, v in sorted(results.items(), key=lambda item: item[1]["score"])
            ][-1]

            max_score = max_score_content["score"]
            genome_seq, query_seq = max_score_content["seqs"]
            probs = max_score_content["probs"]

            # randomly shuffle the genome and get the max score
            for i in range(self.n_sims):
                genome_, probs_ = unison_shuffled_copies(genome_seq, probs)
                _, score, _, _ = smith_waterman(
                    "".join(genome_), query_seq, np.array(probs_)
                )

                self._monte_carlo_results.append(
                    {
                        "score": score,
                        "m": len(genome_),
                        "n": len(query_seq),
                        "max_score": max_score,
                        "sim": i + 1,
                    }
                )
        results = pd.DataFrame(self._monte_carlo_results)
        results.to_csv(self.monte_carlo_path, index=False)
        return results


def unison_shuffled_copies(a, b):
    a = list(a)
    b = list(b)
    c = list(zip(a, b))
    random.shuffle(c)
    a, b = zip(*c)
    return list(a), list(b)
