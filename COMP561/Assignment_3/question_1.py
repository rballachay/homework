import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from utils.popSimulation import PopulationSimulation
from utils.utils import parse_initial_pop
from functools import partial


def random_fitness(population: np.ndarray, masked_idx: int = None):
    "Choose fitness randomly, with opportunity to ignore one parent"
    n_population = population.shape[-1]
    _population = np.ones(n_population)

    if masked_idx:
        _population[masked_idx] = 0
        n_population -= 1

    pop_probs = _population / n_population
    return np.random.choice(np.arange(population.shape[-1]), p=pop_probs)


def SNP_N_fitness(population: np.ndarray, snp_arr: np.ndarray, masked_idx: int = None):
    """Takes in the population, and checks the number of counts of each SNP, multiplying
    them by the snp_arr, which represents the advantage given by having 1 count of each
    snp.
    """
    _snp_pop = np.matmul(np.sum(population, axis=1).T, snp_arr)
    _population = np.ones(population.shape[-1]) + _snp_pop

    if masked_idx:
        _population[masked_idx] = 0

    pop_probs = _population / np.sum(_population)
    return np.random.choice(np.arange(population.shape[-1]), p=pop_probs)


class Problems:
    AVG_FREQUENCY_20GEN = "results/avg_frequency_20gens.png"
    EXTINCT_FIXATE_1000 = "results/extinction_fixation_1000.png"
    EXTINCTION_SNP42_SELECT = "results/extinction_snp42_select.png"
    EXTINCTION_SNP42_DELETE = "results/extinction_snp42_delete.png"

    @classmethod
    def problem_b(cls, population):
        simulation = PopulationSimulation(population, 1, True)
        simulation.simulate(fitness_function=random_fitness)
        simulation.summarize()

    @classmethod
    def problem_d(cls, population):
        simulation = PopulationSimulation(population, 20, False)
        simulation.simulate(fitness_function=random_fitness)
        simulation.summarize()

        fig, ax = plt.subplots()
        for i in range(100):
            plt.plot(range(1, 21), simulation.avg_freq[i, :])

        ax.set_xlabel("Generation")
        ax.set_ylabel("Alternate Allele Frequency")
        fig.savefig(cls.AVG_FREQUENCY_20GEN)

    @classmethod
    def problem_e(cls, population):
        simulation = PopulationSimulation(population, 1000, True)
        simulation.simulate(fitness_function=random_fitness)
        simulation.summarize()

        fig, ax = plt.subplots()
        ax.plot(range(1, 1001), simulation.extinction, "g--")

        ax2 = ax.twinx()
        ax2.plot(range(1, 1001), simulation.fixation, "b--")

        ax.set_xlabel("Generation")
        ax.set_ylabel("Extinction Rate", color="g")
        ax2.set_ylabel("Fixation Rate", color="b")
        fig.savefig(cls.EXTINCT_FIXATE_1000)

    def problem_f(population):
        snp_arr = np.zeros(population.shape[0])
        snp_arr[42] = 0.5

        simulation = PopulationSimulation(population, 1, True)

        avgs = np.zeros(100)
        for i in range(len(avgs)):
            snp_n_fitness = partial(SNP_N_fitness, snp_arr=snp_arr)
            simulation.simulate(fitness_function=snp_n_fitness)
            simulation.summarize()
            avgs[i] = simulation.count_end[42]

        print(f"After {len(avgs)} sims, percentage extinct = {np.average(avgs==0)}")

    @classmethod
    def problem_g(cls, population):
        # reduce the number of snps
        _population = population[:100, ...]
        simulation = PopulationSimulation(_population, 100, False)

        snp_arr = np.zeros(_population.shape[0])
        snp_arr[42] = 0.5

        counts = np.zeros((100, 100))
        for i in range(counts.shape[0]):
            snp_n_fitness = partial(SNP_N_fitness, snp_arr=snp_arr)
            simulation.simulate(fitness_function=snp_n_fitness)

            # count of snp_n over 100 generations
            snp_n = simulation.count_snp_n(42)
            counts[i, :] = snp_n

        extinction = (counts == 0).sum(axis=0) / 100
        fixation = (counts == 200).sum(axis=0) / 100

        fig, ax = plt.subplots()
        ax.plot(range(1, 101), extinction, "g--")

        ax2 = ax.twinx()
        ax2.plot(range(1, 101), fixation, "b--")

        ax.set_xlabel("Generation")
        ax.set_ylabel("Extinction Rate", color="g")
        ax2.set_ylabel("Fixation Rate", color="b")
        fig.savefig(cls.EXTINCTION_SNP42_SELECT)

    @classmethod
    def problem_h(cls, population):
        # reduce the number of snps
        _population = population[:100, ...]
        simulation = PopulationSimulation(_population, 100, False)

        snp_arr = np.zeros(_population.shape[0])
        snp_arr[42] = -0.11

        counts = np.zeros((100, 100))
        for i in range(counts.shape[0]):
            snp_n_fitness = partial(SNP_N_fitness, snp_arr=snp_arr)
            simulation.simulate(fitness_function=snp_n_fitness)

            # count of snp_n over 100 generations
            snp_n = simulation.count_snp_n(42)
            counts[i, :] = snp_n

        extinction = (counts == 0).sum(axis=0) / 100
        fixation = (counts == 200).sum(axis=0) / 100

        fig, ax = plt.subplots()
        ax.plot(range(1, 101), extinction, "g--")

        ax2 = ax.twinx()
        ax2.plot(range(1, 101), fixation, "b--")

        ax.set_xlabel("Generation")
        ax.set_ylabel("Extinction Rate", color="g")
        ax2.set_ylabel("Fixation Rate", color="b")
        fig.savefig(cls.EXTINCTION_SNP42_DELETE)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="Population Simulation", description="Simulate population"
    )

    parser.add_argument("--file", type=str, default="data/initial_population.vcf")

    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    # read file and run program
    df = pd.read_csv(args.file, sep="\t")

    # population is (10,000, 2, 100) numpy array
    population = parse_initial_pop(df)

    # Problems.problem_b(population)

    # Problems.problem_d(population)

    # Problems.problem_e(population)

    # Problems.problem_f(population)

    # Problems.problem_g(population)

    # Problems.problem_h(population)
