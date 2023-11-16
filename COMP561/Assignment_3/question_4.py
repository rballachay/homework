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


def generate_viterbi_config(gff: pd.DataFrame, gff_lens: pd.DataFrame, cfg: str):
    if os.path.exists(cfg):
        with open(cfg, "r") as _obj:
            loaded_cfg = yaml.safe_load(_obj)
    else:
        # forward coding genes
        forward_genes = gff[
            (gff["type"] == "CDS") & (gff["seqid"] != "###") & (gff["strand"] == "+")
        ].reset_index(drop=True)
        # forward_genes['id'] = forward_genes['seqid'].str[-5:].astype(int)
        coding_len = float((forward_genes["end"] - forward_genes["start"]).mean())

        print(f"Average coding region length: {coding_len:.0f}")

        all_exons = []
        all_gff_seqs = []
        for row in gff_lens.itertuples():
            seq = np.ones(row.len)
            for subseq in forward_genes[
                forward_genes["seqid"] == row.seqid
            ].itertuples():
                seq[int(subseq.start) : int(subseq.end)] = np.nan

            cumsum = pd.Series(seq).cumsum().fillna(method="pad")
            reset = -cumsum[pd.Series(seq).isnull()].diff().fillna(cumsum)
            result = (
                pd.Series(seq).where(pd.Series(seq).notnull(), reset).cumsum().values
            )
            end_of_frame = (
                (result - np.concatenate([result[1:], [0]], axis=0)) == result
            ) & (result > 0)
            exons_lens = end_of_frame.astype(int) * result
            exons_lens[exons_lens == 0] = np.nan
            all_exons.extend(exons_lens[~np.isnan(exons_lens)].tolist())

            all_gff_seqs.extend(seq.tolist())

        noncode_len = float(np.average(all_exons))
        print(f"Average non-coding region length: {noncode_len:.0f}")

        coding_chars = [
            full_seq[i] for i in range(len(full_seq)) if np.isnan(all_gff_seqs[i])
        ]
        noncoding_chars = [
            full_seq[i] for i in range(len(full_seq)) if ~np.isnan(all_gff_seqs[i])
        ]

        coding_bases = dict(Counter(coding_chars))
        total = sum(coding_bases.values())
        coding_bases = {k: v / total for k, v in coding_bases.items()}

        noncoding_bases = dict(Counter(noncoding_chars))
        total = sum(noncoding_bases.values())
        noncoding_bases = {k: v / total for k, v in noncoding_bases.items()}

        loaded_cfg = {
            "noncode_len": noncode_len,
            "coding_len": coding_len,
            "coding_bases": coding_bases,
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

    config_dict = generate_viterbi_config(gff, gff_lens, args.cfg)

    config = Config(
        config_dict["coding_bases"],
        config_dict["noncoding_bases"],
        config_dict["coding_len"],
        config_dict["noncode_len"],
    )

    viterbi = ViterbiAlgorithm(config)

    # treat each sequence differently
    for name, seq in fasta_dict.items():
        viterbi.parse_sequence(seq)
