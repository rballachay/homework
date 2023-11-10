import numpy as np
from scipy import stats
import pandas as pd


class GWASRunner:
    def __init__(self, population: np.ndarray, pheno: np.ndarray, verbose: bool):
        self.population = population.copy()
        self.pheno = pheno
        self.verbose = verbose

        self.p_values = np.zeros(population.shape[0])
        self.corrected_p = np.zeros(population.shape[0])
        self.allele_count = np.sum(self.population, axis=1)

    def run_chi2(self):
        disease_counts = self.__allele_counter(self.allele_count, self.pheno)
        healthy_counts = self.__allele_counter(self.allele_count, ~self.pheno)

        counts_final = np.stack([disease_counts, healthy_counts], axis=0)
        for i in range(counts_final.shape[-1]):
            try:
                res = stats.chi2_contingency(counts_final[..., i])
                self.p_values[i] = res[1]
            except Exception as e:
                self.print(f"Failed running chi2 for SNP{i}, setting p to 1")
                self.p_values[i] = 1

        # bonferroni corrected p values
        self.corrected_p = self.p_values * self.p_values.shape[0]

    def make_snp_table(self, corrected_p: float, snp_ids: list):
        self.disease_probs = np.zeros((self.allele_count.shape[0], 3))

        # create phenotype array to use as values
        pheno_array = np.tile(self.pheno, (self.allele_count.shape[0], 1)).astype(float)

        for i in range(3):
            # first, need to make a mask of hetero individuals.
            # completely ignore where we don't have the disease
            allele = (self.allele_count == i).astype(float)
            allele[allele == 0] = np.nan

            self.disease_probs[:, i] = np.nanmean(
                np.multiply(pheno_array, allele), axis=1
            )

        self.disease_odds_hetero = self.disease_probs[:, 1] / self.disease_probs[:, 0]
        self.disease_odds_homo_alt = self.disease_probs[:, 2] / self.disease_probs[:, 0]

        self.table = pd.DataFrame(
            {
                "SNP ID": snp_ids,
                "Uncorrected p-value": self.p_values,
                "Correct p-value": self.corrected_p,
                "Disease odds ratio for heterozygous individuals": self.disease_odds_hetero,
                "Disease odds ratio for homozygous alternate individuals": self.disease_odds_homo_alt,
            }
        )

        return (
            self.table[self.table["Correct p-value"] < corrected_p]
            .copy()
            .reset_index(drop=True)
        )

    @staticmethod
    def __allele_counter(allele_count: np.ndarray, pheno: np.ndarray):
        count_arr = allele_count[:, pheno]

        # this is the number of individuals that has each allele count, 0, 1 + 2
        zero_count = np.sum((count_arr == 0), axis=1)
        one_count = np.sum((count_arr == 1), axis=1)
        two_count = np.sum((count_arr == 2), axis=1)
        allele_counts_snp = np.stack([zero_count, one_count, two_count], axis=0)
        return allele_counts_snp

    def print(self, message: str):
        if self.verbose:
            print(message)
