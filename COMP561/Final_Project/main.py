import argparse
from pathlib import Path
import numpy as np
from find_seeds import get_query, get_genome_dict, get_query_dict

## config
NUCLEOTIDES = ["A", "C", "G", "T"]
QUERY_LENGTH = 3000
RANDOM_SEED = 18
WINDOW_SIZE = 11
THRESHOLD = 0.6 ** WINDOW_SIZE
GAP_PENALTY = -2


def main(fasta: Path, confidence: Path):
    # load in raw data
    genome, max_conf = load_genome(fasta, confidence)

    # get query string from genome
    query = get_query(genome, max_conf, QUERY_LENGTH, NUCLEOTIDES, RANDOM_SEED)

    # get 11-len fragments from query
    genome_dict = get_genome_dict(genome, WINDOW_SIZE)

    # get 11-len fragments from query
    query_dict = get_query_dict(
        genome, genome_dict, query, max_conf, WINDOW_SIZE, THRESHOLD
    )

    print(query_dict)


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
