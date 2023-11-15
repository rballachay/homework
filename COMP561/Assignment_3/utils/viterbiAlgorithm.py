from typing import NamedTuple
import numpy as np
import itertools


class Config:
    # this isn't referenced directly, just bear this in mind
    states = ["GENE", "INTER", "START", "STOP"]

    # these are all directly referenced
    start_codons = {"ATG"}
    stop_codons = {"TAA", "TGA", "TAG"}
    all_codons = set(
        ["".join(i) for i in itertools.product(["A", "C", "T", "G"], repeat=3)]
    )
    initial_states = {"GENE": 0, "INTER": 1, "START": 0, "STOP": 0}

    def __init__(self, trans_in: dict, trans_out: dict, l_in: float, l_out: float):
        self.emissions = self.__make_emission_dict(trans_in, trans_out)
        self.transitions = self.__make_transition_dict(l_in, l_out)

    @classmethod
    def __make_emission_dict(cls, trans_in: dict, trans_out: dict):
        codons_p_in = np.array(
            [
                trans_in[codon[0]] * trans_in[codon[1]] * trans_in[codon[2]]
                if codon not in cls.stop_codons
                else 0
                for codon in cls.all_codons
            ]
        )
        codons_p_in = list(codons_p_in / codons_p_in.sum())

        codons_p_out = [trans_out[i] for i in "ACTG"]

        # create start codon array
        start_codons = np.zeros(len(cls.all_codons))

        for _codon in cls.start_codons:
            start_codons[list(cls.all_codons).index(_codon)] = 1

        start_codons = list(start_codons / start_codons.sum())

        # create stop codon array
        stop_codons = np.zeros(len(cls.all_codons))

        for _codon in cls.stop_codons:
            stop_codons[list(cls.all_codons).index(_codon)] = 1

        stop_codons = list(stop_codons / stop_codons.sum())

        return {
            "GENE": dict(zip(cls.all_codons, codons_p_in)),
            "INTER": dict(zip("ACTG", codons_p_out)),
            "START": dict(zip(cls.all_codons, start_codons)),
            "STOP": dict(zip(cls.all_codons, stop_codons)),
        }

    @classmethod
    def __make_transition_dict(cls, l_in: float, l_out: float):
        _zeros = np.zeros(len(cls.states))
        # remember that we are keeping order from states
        gene_trans = _zeros.copy()
        gene_trans[0] = 1 - 3 / l_in  # gene to gene is 3/999
        gene_trans[-1] = 3 / l_in

        inter_trans = _zeros.copy()
        inter_trans[1] = 1 - 1 / l_out
        inter_trans[2] = 1 / l_out

        start_trans = _zeros.copy()
        start_trans[0] = 1  # guaranteed transition from start to gene

        stop_trans = _zeros.copy()
        stop_trans[1] = 1  # guaranteed transition from stop to intergene

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

    def parse_sequence(self, seq: str):
        self.n_bases = len(seq)

        # initialize our three dynamic programming tables
        self.dp_prob = np.zeros((self.n_states, self.n_bases))
        self.dp_path = np.zeros((self.n_states, self.n_bases))
        self.in_phase = np.ones((self.n_states, self.n_bases))

        self.__viterbi(seq)

    def __viterbi(self, seq):
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

            _state_probs = np.zeros(self.n_states)
            for j, _state in enumerate(["GENE", "INTER", "START", "STOP"]):
                if _state == "INTER":
                    _prob = self.config.emissions[_state][observed_base]
                else:
                    # if len(observed_codon)==3:
                    _prob = self.config.emissions[_state][observed_codon]
                    # else:
                    # _prob = 0

                _state_probs[j] = _prob

            for j in range(self.n_states):
                # if the current in_phase is false, we don't want to touch this
                if not self.in_phase[j, i]:
                    continue

                # we know that state j will be in position j of the transition dict
                _transition = np.array(
                    [_arr[j] for _, _arr in self.config.transitions.items()]
                )

                _state_transition = np.multiply(_transition, self.dp_prob[:, i - 1])

                # random tie breaking of all zeros will always give preference to coming from intergene
                if not np.any(_state_transition):
                    idx = 1
                else:
                    idx = np.argmax(_state_transition)

                # current probability  is the current state multiplied by most likely parent
                _prob_max = _state_transition[idx] * _state_probs[j]

                # this means that we are transitioning into a three-base reading frame,
                # and must ignore the next two bases when we transition from here
                if idx != 1:
                    print(
                        f"The most likely path for {self.config.states[j]} here is coming from a {self.config.states[idx]}! We are now shifting the reading frame into step size 3, {_prob_max}"
                    )

                    self.in_phase[j, i : i + 3] = False

                    self.dp_path[j, i + 3] = idx
                    self.dp_prob[j, i + 3] = _prob_max
                else:
                    self.dp_path[j, i] = idx
                    self.dp_prob[j, i] = _prob_max

            self.dp_prob[:, i] = self.dp_prob[:, i] / (self.dp_prob[:, i].sum())
            print(self.dp_prob[:, i])
