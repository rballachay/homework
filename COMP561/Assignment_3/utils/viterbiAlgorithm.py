import numpy as np
import itertools
import pandas as pd
from collections import Counter


class Config:
    # this isn't referenced directly, just bear this in mind
    states = ["GENE", "INTER", "START", "STOP"]

    # these are all directly referenced
    start_codons = {"ATG"}
    stop_codons = {"TAA", "TGA", "TAG"}
    all_codons = set(
        ["".join(i) for i in itertools.product(["A", "C", "T", "G"], repeat=3)]
    )

    initial_states = {"GENE": -np.inf, "INTER": 0, "START": -np.inf, "STOP": -np.inf}

    def __init__(self, trans_in: dict, trans_out: dict, l_in: float, l_out: float):
        self.emissions = self.__make_emission_dict(trans_in, trans_out)
        self.transitions = self.__make_transition_dict(l_in, l_out)

    @classmethod
    def __make_emission_dict(cls, trans_in: dict, trans_out: dict):
        codons_p_in = np.array(list(trans_in.values()))
        codons_p_in = np.log(codons_p_in / codons_p_in.sum())

        codons_p_out = np.array([trans_out[i] for i in "ACTG"])
        codons_p_out = np.log(codons_p_out / codons_p_out.sum())

        # create start codon array
        start_codons = np.zeros(len(cls.all_codons))

        for _codon in cls.start_codons:
            start_codons[list(cls.all_codons).index(_codon)] = 1

        start_codons = list(np.log(start_codons / start_codons.sum()))

        # create stop codon array
        stop_codons = np.zeros(len(cls.all_codons))

        for _codon in cls.stop_codons:
            stop_codons[list(cls.all_codons).index(_codon)] = 1

        stop_codons = list(np.log(stop_codons / stop_codons.sum()))

        return {
            "GENE": dict(zip(trans_in.keys(), codons_p_in)),
            "INTER": dict(zip("ACTG", codons_p_out)),
            "START": dict(zip(cls.all_codons, start_codons)),
            "STOP": dict(zip(cls.all_codons, stop_codons)),
        }

    @classmethod
    def __make_transition_dict(cls, l_in: float, l_out: float):
        _zeros = np.zeros(len(cls.states))
        # remember that we are keeping order from states
        gene_trans = _zeros.copy()
        gene_trans[0] = 1 - 3 / l_in  # gene to gene is 1-3/999
        gene_trans[-1] = 3 / l_in  # gene to stop is 3/999
        gene_trans = np.log(gene_trans / gene_trans.sum())

        inter_trans = _zeros.copy()
        inter_trans[1] = 1 - 1 / l_out
        inter_trans[2] = 1 / l_out
        inter_trans = np.log(inter_trans / inter_trans.sum())

        start_trans = _zeros.copy()
        start_trans[0] = 1  # guaranteed transition from start to gene
        start_trans = np.log(start_trans / start_trans.sum())

        stop_trans = _zeros.copy()
        stop_trans[1] = 1  # guaranteed transition from stop to intergene
        stop_trans = np.log(stop_trans / stop_trans.sum())

        return {
            "GENE": gene_trans,
            "INTER": inter_trans,
            "START": start_trans,
            "STOP": stop_trans,
        }


class ViterbiAlgorithm:
    def __init__(self, config: Config):
        self.config = config

        self.n_states = len(self.config.states)

        # initialize empty properties
        self.dp_prob = None
        self.dp_path = None
        self.in_phase = None

        self.n_bases = None
        self.state_array = None
        self.coding_regions = None

    def parse_sequence(self, seq: str):
        self.n_bases = len(seq)

        # initialize our three dynamic programming tables
        self.dp_prob = np.full((self.n_states, self.n_bases), -np.inf)
        self.dp_path = np.full((self.n_states, self.n_bases), -1)
        self.in_phase = np.ones((self.n_states, self.n_bases))
        self.state_array = []
        self.coding_regions = []

        # run the forward + populate the three arrays initialized above
        self.__viterbi_forward(seq)

        # run the sequence backward
        self.__viterbi_backward(seq)

        return self.state_array.copy(), self.coding_regions.copy()

    def __viterbi_forward(self, seq):
        """Rules for codon length:
        If you're in a gene or a stop codon, the stride length always has to be 3,
        meaning you always have to be coming from a position 3 bases ahead of you.
        If you go from a stop gene to intergene, you need to account for the fact
        that you are transitioning from a codon of length 3 + need to skip the 2
        next bases.
        """

        # run initialization
        for i in range(self.n_states):
            self.dp_prob[i, 0] = self.config.initial_states[self.config.states[i]]

        for i in range(1, self.n_bases):
            observed_base = seq[i]
            observed_codon = seq[i : i + 3]

            _state_probs = np.full(self.n_states, -np.inf)
            for j, _state in enumerate(["GENE", "INTER", "START", "STOP"]):
                if _state == "INTER":
                    _prob = self.config.emissions[_state][observed_base]
                else:
                    if len(observed_codon) == 3:
                        _prob = self.config.emissions[_state][observed_codon]
                    else:
                        _prob = -np.inf

                _state_probs[j] = _prob

            for j in range(self.n_states):
                # don't update current spot in table if we're not in phase
                if not self.in_phase[j, i]:
                    continue

                # there's no point filling this value if the probability
                # of the state is zero
                if _state_probs[j] == -np.inf:
                    continue

                # we know that state j will be in position j of the transition dict
                _transition = np.array(
                    [_arr[j] for _, _arr in self.config.transitions.items()]
                )

                # we have to normalize because the transition dict is set up in the
                # other direction, but the idea is that we need to know the probability
                # of coming to the state we are currently at from the other states,
                # which should sum to one
                # print(_transition)
                # _transition = _transition/_transition.sum()
                _state_transition = _transition + self.dp_prob[:, i - 1]
                # _state_transition = np.multiply(_state_transition, self.in_phase[:, i])
                _state_transition[self.in_phase[:, i] == False] = -np.inf
                _state_transition = _state_transition + _state_probs[j]

                # random tie breaking of all zeros will always give preference to coming from intergene
                if np.all(_state_transition == -np.inf):
                    continue
                else:
                    idx = np.argmax(_state_transition)

                if idx == 3:
                    print(f"I am {j} at {i}, coming from {idx}")

                # current probability  is the current state multiplied by most likely parent
                _prob_max = _state_transition[idx]

                # this means that we are transitioning into a three-base reading frame,
                # and must ignore the next two bases when we transition from here
                if j != 1 and _prob_max > -np.inf:
                    # you cannot transition from the current stage
                    self.in_phase[j, i : i + 3] = False
                    self.dp_path[j, i : i + 3] = idx
                    self.dp_prob[j, i : i + 3] = _prob_max
                    # print(f"I am {j} at {i}, coming from {idx}")
                # don't overwrite if it isn't in phase
                else:
                    self.dp_path[j, i] = idx
                    self.dp_prob[j, i] = _prob_max

                if j == 1 and idx != 1:
                    print("Transitioning from gene to non-gene")

            # normalize by the sum of probabilities for that step
            # self.dp_prob[:,i] = self.dp_prob[:,i]/self.dp_prob[:,i].sum()

    def __viterbi_backward(self, seq):
        # start at the very end of the sequence
        i = self.n_bases - 1

        # get the max probability in the last column
        start_idx = np.argmax(self.dp_prob[:, -1])
        idx = start_idx

        # in the case that the thing doesn't end with intergene, which
        # I guess is possible, start this gene at the end
        if idx != 1:
            stop_gene = i + 1
        while i >= 0:
            # stop index
            if idx == 3:
                stop_gene = i + 1
            elif idx == 2:
                self.coding_regions.append((i - 2, stop_gene))
            next_idx = self.dp_path[idx, i]
            if idx == 1:
                element = seq[i]
                i -= 1
            else:
                element = seq[i - 2 : i + 1]
                i -= 3
            idx = next_idx
            self.state_array.append(element)
        # self.state_array.reverse()


class GeneResults:
    """Class for parsing results of annotated genes and
    comparing annotated results and results from viterbi
    algorithm.
    """

    def __init__(self, annotated: pd.DataFrame, viterbi: pd.DataFrame):
        viterbi = viterbi.copy().reindex(sorted(viterbi.columns), axis=1)
        annotated = annotated.copy().reindex(sorted(annotated.columns), axis=1)

        # check
        assert (annotated.columns == viterbi.columns).all()

        self.annotated = annotated.copy()
        self.viterbi = viterbi.copy()

        self.comparison = None
        self.missed_df = None
        self.partial_df = None

        __unique_seq_ids = set(
            self.annotated.seqid.unique().tolist()
            + self.viterbi.seqid.unique().tolist()
        )
        self.unique_seq_ids = set(sorted(list(__unique_seq_ids)))

    def compare(self) -> pd.DataFrame:
        """Run comparison in steps"""
        # get all the distinct sequences in both dataframes
        __comparison_dicts = []
        __partial = []
        __missed = []

        for seq_id in self.unique_seq_ids:
            _viterbi = (
                self.viterbi[self.viterbi.seqid == seq_id].copy().reset_index(drop=True)
            )
            _annotated = (
                self.annotated[self.annotated.seqid == seq_id]
                .copy()
                .reset_index(drop=True)
            )

            # use the viterbi results as the basis and compare to annotated
            _results_annotated, _missed_i, _partial_i = self.__compare_seq(
                _viterbi, _annotated
            )
            __comparison_dicts.append(_results_annotated)
            __missed.append(_missed_i)
            __partial.append(_partial_i)

        self.comparison = self.__format_comparison(__comparison_dicts.copy())

        # going to return these here even though they aren't the final results.
        # will do analysis in another object
        self.missed_df = pd.concat(__missed).reset_index(drop=True)
        self.partial_df = pd.concat(__partial).reset_index(drop=True)

        return self.comparison.copy(), self.missed_df.copy(), self.partial_df.copy()

    def __compare_seq(self, viterbi: pd.DataFrame, annotated: pd.DataFrame):
        viterbi_genes = viterbi[["start", "end"]].values
        annotated_genes = annotated[["start", "end"]].values

        # essentially like a loop subtracting each row of each array from each row of the other array
        # we first subtract every row of second array, then first array
        # so each index i is n_viterbi*i + n_annotated
        difference = (viterbi_genes[:, np.newaxis] - annotated_genes).reshape(
            -1, viterbi_genes.shape[1]
        )

        # missed genes here is a list of indexes in annotated_genes that correspond to
        # genes that weren't detected by the viterbi algorithm
        missed_idx, partial_idx = self.__get_missed_genes(
            difference, annotated_genes.shape[0]
        )

        missed_genes = viterbi.iloc[missed_idx]
        partial_genes = viterbi.iloc[partial_idx]

        # check the matching
        matches_both = (difference.sum(axis=1) == 0).sum(axis=0)
        matches_start = (difference[:, 0] == 0).sum(axis=0) - matches_both
        matches_end = (difference[:, 1] == 0).sum(axis=0) - matches_both

        return (
            {
                "match_start": matches_start,
                "match_end": matches_end,
                "matches_both": matches_both,
                "n_annotated_genes": annotated_genes.shape[0],
                "n_viterbi_genes": viterbi_genes.shape[0],
            },
            missed_genes,
            partial_genes,
        )

    @staticmethod
    def __format_comparison(comparison_dicts: list):
        comparison_df = pd.DataFrame(comparison_dicts)

        results = {}
        for divisor in {"annotated", "viterbi"}:
            sub_results = {}
            for numerator in {"match_start", "match_end", "matches_both"}:
                value = (
                    comparison_df[numerator] / comparison_df[f"n_{divisor}_genes"]
                ).mean(axis=0)
                sub_results[numerator] = float(f"{value:.3f}")

            results[divisor] = sub_results
        return results

    @staticmethod
    def __get_missed_genes(difference: np.ndarray, n_annotated: int):
        """Each n_annotated rows corresponds to the nth element of the n_viterbi
        array subtracted vs the entire annotated array.
        """
        matching = difference == 0
        if not n_annotated:
            return np.array([]), np.array([])

        # this will tell us if the gene is
        annotated_info = (
            matching.reshape(-1, n_annotated, matching.shape[-1])
            .sum(axis=1)
            .sum(axis=1)
        )

        missed_idx = np.where(annotated_info == 0)
        partial_idx = np.where(annotated_info == 1)
        return missed_idx, partial_idx


class MissedGeneAnalysis:
    def __init__(
        self,
        missed_df: pd.DataFrame,
        partial_df: pd.DataFrame,
        annotated: pd.DataFrame,
        genes_dict: dict,
        codons_dict: dict,
    ):
        self.missed_df = missed_df.copy()
        self.partial_df = partial_df.copy()
        self.annotated = annotated.copy()
        self.genes_dict = genes_dict.copy()
        self.codons_dict = codons_dict

    def analyze(self):
        # compare the length of missed, partially missed and all annotated genes
        l_missed, l_partial, l_all = self.__get_lengths()
        print(
            f"Avg len missed = {l_missed:.3f}, len partial = {l_partial:.3f}, len total = {l_all:.3f}"
        )
        missed_freq, partial_freq, annotated_freq = self.__get_codon_freqs()
        print(
            f"Avg missed codon prob = {missed_freq:.5f}, avg partial codon prob = {partial_freq:.5f}, avg annotated codon freq = {annotated_freq:.5f}"
        )

    def __get_lengths(self):
        l_missed = (self.missed_df["end"] - self.missed_df["start"]).mean()
        l_partial = (self.partial_df["end"] - self.partial_df["start"]).mean()
        l_annotated = (self.annotated["end"] - self.annotated["start"]).mean()
        return l_missed, l_partial, l_annotated

    def __get_codon_freqs(self):
        missed_freq = self.__get_codons(
            self.missed_df, self.genes_dict, self.codons_dict
        )
        partial_freq = self.__get_codons(
            self.partial_df, self.genes_dict, self.codons_dict
        )
        annotated_freq = self.__get_codons(
            self.annotated, self.genes_dict, self.codons_dict
        )
        return missed_freq, partial_freq, annotated_freq

    @staticmethod
    def __get_codons(
        gene_df: pd.DataFrame, seq_dict: dict, codons_dict: dict, offset=1
    ):
        codons_all = []
        i = 0
        for seqid in seq_dict.keys():
            _seq = seq_dict[seqid]
            genes = gene_df[gene_df["seqid"] == seqid]
            for gene in genes.itertuples():
                codons = chunks(_seq[int(gene.start) - 1 + 3 : int(gene.end) - 3], 3)
                codons_all.extend(list(codons))

        codons_counted = dict(Counter(codons_all))

        total = sum(codons_counted.values())
        codons_counted = {k: v / total for k, v in codons_counted.items()}

        codons = [codons_counted[i] * codons_dict[i] for i in codons_counted.keys()]
        return sum(codons)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        _yield = lst[i : i + n]
        if len(_yield) != n:
            continue
        yield _yield
