import yaml
from utils.viterbiAlgorithm import Config, ViterbiAlgorithm


def main():
    cfg = "data/viterbi_config.yaml"

    with open(cfg, "r") as _obj:
        config_dict = yaml.safe_load(_obj)

    config = Config(
        config_dict["coding_bases"],
        config_dict["noncoding_bases"],
        config_dict["coding_len"],
        config_dict["noncode_len"],
    )

    viterbi = ViterbiAlgorithm(config)

    states = viterbi.parse_sequence(
        "CTAGCTCGATAGCTCGCTCGATAGGGCTCTCGAAGCTCGTGAATGCCGTGACGTACGGAGCGATTAGATAGATGGCTAGAGGCCGTGCGAGCAGGGTAAGGAGCGTGAGCAAATGATCAGGCGATGCGCTTTAACGGTCAAGGACGTACGCCCAGCGAAGAGCGTATGTGGACGATGATTTA"
    )
    print(states)


if __name__ == "__main__":
    main()
