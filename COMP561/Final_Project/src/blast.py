from src.find_seeds import get_query, get_genome_dict, get_query_dict
from src.ungapped_extension import get_ungapped_hsps
from src.sequence_alignment import align_sequences
import numpy as np


class BLAST:
    def __init__(
        self,
        window_size: int,
        nucleotides: list,
        threshold: float,
        random_seed: int,
        gap_penalty: int,
        search_distance: int,
        plot: bool = True,
    ):

        self.window_size = window_size
        self.nucleotides = nucleotides
        self.threshold = threshold
        self.random_seed = random_seed
        self.gap_penalty = gap_penalty
        self.search_distance = search_distance
        self.plot = plot

    def run(
        self,
        genome: str,
        max_conf: np.ndarray,
        query_length: int,
        scoring_func: callable,
    ):
        # get 11-len fragments from query
        genome_dict = get_genome_dict(
            genome, max_conf, self.window_size, self.nucleotides, self.threshold
        )

        # get query string from genome
        query = get_query(
            genome, max_conf, query_length, self.nucleotides, self.random_seed
        )

        # get 11-len fragments from query
        query_dict = get_query_dict(genome_dict, query, self.window_size)

        # get dictionary of all high-scoring pairs
        ungapped_hsps, plotter = get_ungapped_hsps(
            genome, query_dict, max_conf, query, scoring_func
        )

        if self.plot:
            fig = plotter.plot()

        # align sequences locally and get score
        scored_hsps = align_sequences(
            genome,
            query,
            max_conf,
            ungapped_hsps,
            self.gap_penalty,
            self.search_distance,
        )
        return scored_hsps
