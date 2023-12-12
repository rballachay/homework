import argparse
from pathlib import Path
import numpy as np
from find_seeds import get_query, get_genome_dict, get_query_dict
from ungapped_extension import get_ungapped_hsps
from needleman_wunsch import nw_extension

## config
NUCLEOTIDES = ["A", "C", "G", "T"]
QUERY_LENGTH = 500
RANDOM_SEED = 18
WINDOW_SIZE = 11
THRESHOLD = 0.6 ** WINDOW_SIZE
GAP_PENALTY = 2
MIN_LEN_HSP = 15


def main(fasta: Path, confidence: Path):
    # load in raw data
    genome, max_conf = load_genome(fasta, confidence)

    # get query string from genome
    query = get_query(genome, max_conf, QUERY_LENGTH, NUCLEOTIDES, RANDOM_SEED)

    # get 11-len fragments from query
    genome_dict = get_genome_dict(genome, max_conf, WINDOW_SIZE, NUCLEOTIDES, THRESHOLD)

    # get 11-len fragments from query
    query_dict = get_query_dict(genome_dict, query, WINDOW_SIZE)

    ungapped_hsps = get_ungapped_hsps(genome, query_dict, max_conf, query, MIN_LEN_HSP)

    nw_extension(genome, query, max_conf, ungapped_hsps, GAP_PENALTY)


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
