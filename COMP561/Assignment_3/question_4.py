import pandas as pd
import numpy as np
from collections import Counter
import yaml
import os
from utils.viterbiAlgorithm import Config, ViterbiAlgorithm


def parse_fasta_file(fasta):
    full_seq = ""
    sequences = {}
    key = None
    with open(fasta, "r") as f:
        for line in f:
            line = line.replace("\n", "")
            if line.startswith(">"):
                key = line.split(" ")[0][1:]
                sequences[key] = ""
            else:
                sequences[key] += line
                full_seq += line
    return sequences, full_seq


def rle(inarray):
    """run length encoding. Partial credit to R rle function.
    Multi datatype arrays catered for including non Numpy
    returns: tuple (runlengths, startpositions, values)"""
    ia = np.asarray(inarray)  # force numpy
    n = len(ia)
    y = ia[1:] != ia[:-1]  # pairwise unequal (string safe)
    i = np.append(np.where(y), n - 1)  # must include last element posi
    z = np.diff(np.append(-1, i))  # run lengths
    p = np.cumsum(np.append(0, z))[:-1]  # positions
    return (z, p, ia[i])


def generate_viterbi_config(
    gff: pd.DataFrame, gff_lens: pd.DataFrame, cfg: str, sequences: str
):
    if os.path.exists(cfg):
        with open(cfg, "r") as _obj:
            loaded_cfg = yaml.safe_load(_obj)
    else:
        # forward coding genes
        forward_genes = gff[
            (gff["type"] == "CDS") & (gff["seqid"] != "###") & (gff["strand"] == "+")
        ].reset_index(drop=True)

        coding_len = float(
            (forward_genes["end"] - forward_genes["start"]).mean(skipna=False)
        )

        print(f"Average coding region length: {coding_len:.0f}")

        all_exons = []
        all_introns = []
        all_gff_seqs = []
        total_len = 0
        counter = 0
        for row in gff_lens.itertuples():
            seq = np.zeros(int(row.len))
            _genes = forward_genes[forward_genes["seqid"] == row.seqid]
            for subseq in _genes.itertuples():
                assert subseq.end < row.len
                assert subseq.start < subseq.end
                seq[int(subseq.start) : int(subseq.end)] = 1

            len_seqs, start_idxs, values = rle(seq)
            all_exons.extend(len_seqs[(values == 0)].tolist())
            all_introns.extend(len_seqs[(values == 1)].tolist())

            all_gff_seqs.extend(seq.tolist())
            total_len += row.len

        noncode_len = float(np.average(all_exons))
        print(f"Average non-coding region length: {noncode_len:.0f}")

        codons = [
            full_seq[i : i + 3]
            for i in range(0, len(full_seq), 3)
            if (all_gff_seqs[i] == 1)
        ]
        noncoding_chars = [
            full_seq[i] for i in range(len(full_seq)) if all_gff_seqs[i] == 0
        ]

        codon_dict = dict(Counter(codons))

        # remove all the stop codons
        for _codon in {"TAA", "TGA", "TAG"}:
            codon_dict[_codon] = 0

        total = sum(codon_dict.values())
        codon_dict = {k: v / total for k, v in codon_dict.items()}

        noncoding_bases = dict(Counter(noncoding_chars))
        total = sum(noncoding_bases.values())
        noncoding_bases = {k: v / total for k, v in noncoding_bases.items()}

        loaded_cfg = {
            "noncode_len": noncode_len,
            "coding_len": coding_len,
            "coding_bases": codon_dict,
            "noncoding_bases": noncoding_bases,
        }

        with open(cfg, "w") as outfile:
            yaml.dump(loaded_cfg, outfile, default_flow_style=False)

    return loaded_cfg


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="Viterbi Gene Finding", description="Find Genes using HMMs"
    )

    parser.add_argument(
        "--gff", type=str, default="data/Vibrio_cholerae.GFC_11.37.gff3"
    )
    parser.add_argument(
        "--fasta", type=str, default="data/Vibrio_cholerae.GFC_11.dna.toplevel.fa"
    )
    parser.add_argument("--cfg", type=str, default="data/viterbi_config.yaml")

    args = parser.parse_args()

    gff = pd.read_csv(args.gff, sep="\t", header=None, skiprows=range(157))
    gff.columns = [
        "seqid",
        "source",
        "type",
        "start",
        "end",
        "score",
        "strand",
        "phase",
        "attributes",
    ]

    gff_lens = pd.read_csv(
        args.gff,
        delim_whitespace=True,
        skiprows=lambda x: x not in range(1, 152),
        header=None,
    )
    gff_lens = gff_lens[[1, 3]]
    gff_lens.columns = ["seqid", "len"]

    fasta_dict, full_seq = parse_fasta_file(args.fasta)

    config_dict = generate_viterbi_config(gff, gff_lens, args.cfg, full_seq)

    config = Config(
        config_dict["coding_bases"],
        config_dict["noncoding_bases"],
        config_dict["coding_len"],
        config_dict["noncode_len"],
    )

    viterbi = ViterbiAlgorithm(config)

    final_data = []
    # treat each sequence differently
    for name, seq in fasta_dict.items():
        _, coding_regions = viterbi.parse_sequence(seq)
        if coding_regions:
            for start, end in coding_regions:
                # note that the bases aren't zero-indexed here
                final_data.append(
                    {
                        "seqid": name,
                        "start": start + 1,
                        "end": end + 1,
                    }
                )

    final_df = pd.DataFrame(final_data)
    final_df["source"] = "ena"
    final_df["type"] = "CDS"
    final_df["score"] = "."
    final_df["strand"] = "+"
    final_df["phase"] = 0
    final_df["attributes"] = "."
    final_df.to_csv("Vibrio_cholerae_viterbi_genes.csv", index=False)
