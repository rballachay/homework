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
            "GENE": trans_in.copy(),
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
        assert sum(gene_trans) == 1

        inter_trans = _zeros.copy()
        inter_trans[1] = 1 - 1 / l_out
        inter_trans[2] = 1 / l_out
        assert sum(inter_trans) == 1

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
        self.state_array = None
        self.coding_regions = None

    def parse_sequence(self, seq: str):
        self.n_bases = len(seq)

        # initialize our three dynamic programming tables
        self.dp_prob = np.zeros((self.n_states, self.n_bases))
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

            _state_probs = np.zeros(self.n_states)
            for j, _state in enumerate(["GENE", "INTER", "START", "STOP"]):
                if _state == "INTER":
                    _prob = self.config.emissions[_state][observed_base]
                else:
                    if len(observed_codon) == 3:
                        _prob = self.config.emissions[_state][observed_codon]
                    else:
                        _prob = 0

                _state_probs[j] = _prob

            for j in range(self.n_states):
                # don't update current spot in table if we're not in phase
                if not self.in_phase[j, i]:
                    continue

                # there's no point filling this value if the probability
                # of the state is zero
                if _state_probs[j] == 0:
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
                _state_transition = np.multiply(_transition, self.dp_prob[:, i - 1])
                _state_transition = np.multiply(_state_transition, self.in_phase[:, i])
                _state_transition = np.multiply(_state_transition, _state_probs[j])

                # if idx==

                # random tie breaking of all zeros will always give preference to coming from intergene
                if not np.any(_state_transition):
                    continue
                else:
                    idx = np.argmax(_state_transition)

                if idx == 3:
                    print(f"I am {j} at {i}, coming from {idx}")
                # current probability  is the current state multiplied by most likely parent
                _prob_max = _state_transition[idx]

                # this means that we are transitioning into a three-base reading frame,
                # and must ignore the next two bases when we transition from here
                if j != 1 and _prob_max > 0:
                    # you cannot transition from the current stage
                    self.in_phase[j, i : i + 3] = False
                    self.dp_path[j, i : i + 3] = idx
                    self.dp_prob[j, i : i + 3] = _prob_max
                # don't overwrite if it isn't in phase
                else:
                    self.dp_path[j, i] = idx
                    self.dp_prob[j, i] = _prob_max

                if j == 1 and idx != 1:
                    print("Transitioning from gene to non-gene")

            # some random number to keep the thing going
            if not i % 3:
                if sum(self.dp_prob[:, i]) < 1:
                    self.dp_prob *= 100

            if sum(self.dp_prob[:, i]) == 0:
                raise Exception("Probability reached zero, must re-start")
            if np.isnan(self.dp_prob[:, i]).any():
                raise Exception("Probability reached infinity!!!")

            # normalize by the sum of probabilities for that step
            # self.dp_prob[:,i] = self.dp_prob[:,i]/self.dp_prob[:,i].sum()

    def __viterbi_backward(self, seq):
        # start at the very end of the sequence
        i = self.n_bases - 1

        # get the max probability in the last column
        start_idx = np.argmax(self.dp_prob[:, -1])
        idx = start_idx
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
