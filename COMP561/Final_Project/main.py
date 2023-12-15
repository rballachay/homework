import argparse
from pathlib import Path
import numpy as np
from blast import BLAST
from eval_significance import GumbelSignificance

# most important parameter
QUERY_LENGTH = 50

## config
NUCLEOTIDES = ["A", "C", "G", "T"]
RANDOM_SEED = 18
WINDOW_SIZE = 11
THRESHOLD = 0.6 ** WINDOW_SIZE
GAP_PENALTY = 2
MIN_LEN_HSP = 20  # minimum length of high-scoring pair to be considered for gapped hsp
SEARCH_DISTANCE = 50  # search distance for local alignment


def main(fasta: Path, confidence: Path):
    # load in raw data
    genome, max_conf = load_genome(fasta, confidence)

    blast = BLAST(
        WINDOW_SIZE,
        NUCLEOTIDES,
        THRESHOLD,
        RANDOM_SEED,
        MIN_LEN_HSP,
        GAP_PENALTY,
        SEARCH_DISTANCE,
    )

    # gumbel_sig = GumbelSignificance(blast, genome, max_conf)
    # gumbel_sig.get_significance()

    # run blast
    blast.run(genome, max_conf, QUERY_LENGTH)

    # evaluate significance of the scores.
    # run_monte_carlo(scored_hsps)


def load_genome(fasta: Path, confidence: Path):
    with open(fasta.resolve(), "r") as file:
        genome = file.read()

    max_conf = np.loadtxt(confidence.resolve())

    return genome, max_conf


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Run modified BLAST with probabilities")
    parser.add_argument(
        "-fasta",
        help="Fasta file, sequence",
        type=Path,
        default=Path("data/ch22.maf.ancestors.42000000.complete.boreo.fa.txt"),
    )
    parser.add_argument(
        "-confidence",
        help="Text file, confidence of best base",
        type=Path,
        default=Path("data/ch22.maf.ancestors.42000000.complete.boreo.conf.txt"),
    )
    args = parser.parse_args()

    main(args.fasta, args.confidence)
