import pandas as pd
import numpy as np
from utils.utils import parse_initial_pop
from utils.gwasRunner import GWASRunner
import os

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="Run GWAS", description="Run Genetic-Wide Association Study"
    )

    parser.add_argument("--gwas", type=str, default="data/gwas_population.vcf")
    parser.add_argument("--pheno", type=str, default="data/gwas_phenotypes.txt")
    parser.add_argument(
        "--snp_table", type=str, default="results/snp_table_results.csv"
    )

    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    df = pd.read_csv(args.gwas, sep="\t")

    # this could also be read from gwas but this is simpler
    snp_ids = [f"SNP{i}" for i in range(len(df.index))]

    pheno = pd.read_csv(args.pheno, sep="\t", header=None)[1].values.astype(bool)

    # population is (10000, 2, 1000) numpy array but we want (10000,1000), where
    # the value is the sum over the axis 1
    population = parse_initial_pop(df)

    runner = GWASRunner(population, pheno, False)
    runner.run_chi2()

    print(
        f"A total of {np.sum(runner.p_values<0.05)} SNPs have p_value lower than 0.05"
    )
    print(f"We expect {0.05*population.shape[0]:.0f} SNPs to be randomly significant")

    # make the SNP table
    table = runner.make_snp_table(0.05, snp_ids)
    # table.to_csv(args.snp_table, index=False)
