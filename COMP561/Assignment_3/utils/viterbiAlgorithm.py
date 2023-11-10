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
    initial_states = {"GENE":0,"INTER":1,"START":0, "STOP":0}

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

        codons_p_out = {trans_out[i] for i in "ACTG"}

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
            "INTER": codons_p_out,
            "START": dict(zip(cls.all_codons, start_codons)),
            "STOP": dict(zip(cls.all_codons, stop_codons)),
        }
    
    @classmethod
    def __make_transition_dict(cls, l_in:float, l_out:float):
        _zeros = np.zeros(len(cls.states))
        # remember that we are keeping order from states
        gene_trans = _zeros.copy()
        gene_trans[0] = 1-3/l_in # gene to gene is 3/999
        gene_trans[-1] = 3/l_in

        inter_trans = _zeros.copy()
        inter_trans[1] = 1-1/l_out
        inter_trans[2] = 1/l_out

        start_trans = _zeros.copy()
        start_trans[0] = 1 #guaranteed transition to gene from start

        stop_trans = _zeros.copy()
        stop_trans[1] = 1 #guaranteed transition from stop to intergene

        return {"GENE":gene_trans,"INTER":inter_trans,"START":start_trans,"STOP":stop_trans}


class ViterbiAlgorithm:
    def __init__(self, config:Config):
        self.config = config

        self.n_states = len(self.config.states)

        # initialize empty properties
        self.dp_prob = None
        self.dp_path = None

        self.n_bases = None
    
    def parse_sequence(self, seq:str):
        self.n_bases = len(seq)

        # initialize our two dynamic programming tables
        _dp_mat = np.zeros((self.n_states,self.n_bases))
        self.dp_prob = _dp_mat.copy()
        self.dp_path = _dp_mat.copy()

        self.__viterbi()

    def __viterbi(self, seq):

        # run initialization 
        for i in range(self.n_states):
            self.dp_prob[i,0] = self.config.initial_states[self.config.states[i]]

        for i in range(1,self.n_bases):
            for j in range(self.n_states):
                state_name = self.config.states[j]
                
                # get the emission probability of the current state
                _emission = seq

