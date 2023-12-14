import pandas as pd
import numpy as np
import random
from difflib import SequenceMatcher
from blast import BLAST
from sequence_alignment import smith_waterman


class GumbelSignificance:
    def __init__(self, blast: BLAST, genome: str, probabilities: np.ndarray):
        self.blast = blast
        self.genome = genome
        self.probabilities = probabilities

        self._gumbel_parameters = None

        # this will only be set in the case that the monte
        # carlo simulation is run
        self._monte_carlo_results = None

        # here are the lengths we are going to use to evaluate
        # our gumbel parameters
        self.lens = [50, 100, 250, 500, 1000]

        # this is the number of simulations we are going to run
        # for each of our lengths in order to estimate mean of gumbel
        self.n_sims = 100

    def get_significance(self):
        gumbel_parameters = self.gumbel_parameters

    @property
    def gumbel_parameters(self):
        """Gumbel parameters need to be created before any of this is done"""
        if self._gumbel_parameters is None:
            self._gumbel_parameters = self.__run_monte_carlo_simulation()
        return self._gumbel_parameters

    def __run_monte_carlo_simulation(self):
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
                _, score = smith_waterman("".join(genome_), query_seq, np.array(probs_))

                self._monte_carlo_results.append(
                    {
                        "score": score,
                        "m": len(genome_),
                        "n": len(query_seq),
                        "max_score": max_score,
                        "sim": i + 1,
                    }
                )
        pd.DataFrame(self._monte_carlo_results).to_csv(
            "monte_carlo_results.csv", index=False
        )


def unison_shuffled_copies(a, b):
    a = list(a)
    b = list(b)
    c = list(zip(a, b))
    random.shuffle(c)
    a, b = zip(*c)
    return list(a), list(b)


def evaluate_alignments(scored_hsps: dict):
    """Evaluate the p-value of the alignments
    scored_hsps has structure like:
        'hsp_seed': {
                'score':score,
                'aligned':pairs.split('\n'),
                'seqs':[genome_sub,query_sub],
                'probs':list(probs_sub) }
    """

    # we don't want to run the same analysis multiple times on essentially
    # the exact same sequences. In order to only evaluate those


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

    pass


def fit_gumbel_dist():
    pass
