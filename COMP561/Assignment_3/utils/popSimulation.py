import numpy as np


class PopulationSimulation:
    def __init__(self, population: np.ndarray, n_gen: int, verbose: bool):
        self._initial_population = population
        self.L, _, self.N = population.shape
        self.n_gen = n_gen
        self.verbose = verbose

        self.simulation = None

    def simulate(self, fitness_function):
        population = self._initial_population.copy()

        # array with shape (N_snps, N_alleles, N_inviduals, N_generations)
        self.simulation = np.zeros([population.shape[0], self.n_gen])

        for gen in range(self.n_gen):
            generation = np.zeros_like(population)
            for i in range(self.N):
                # return the index of parent 1 + 2. note that this doesn't guarantee
                # that it contains our allele even if the allele of that index is selected.
                # this is because we don't know yet if that strand will be selected.
                idx_p1 = fitness_function(population, masked_idx=None)
                idx_p2 = fitness_function(population, masked_idx=idx_p1)
                child = self.reproduce(population[..., idx_p1], population[..., idx_p2])
                generation[..., i] = child.copy()
            self.print(
                f"Finished generation {gen+1} of simulation, allele 1 has count of: {np.sum(generation[0,...])}"
            )
            population = generation.copy()

            self.simulation[..., gen] = np.sum(generation.copy(), axis=(1, 2))

    def summarize(self):
        # sum count of allele over both strands, all individuals
        _sum = self.simulation.copy()
        self.count_end = self.simulation.copy()
        x = _sum.T
        max_generation = np.where(
            np.count_nonzero(x, axis=0) == 0,
            0,
            (x.shape[0]) - np.argmin(x[::-1, :] == 0, axis=0),
        )
        self.avg_life = np.average(max_generation)
        self.per_extinct = np.average(x[-1, :] == 0)
        self.print(
            f"The average life of each allele in {self.n_gen} generations is {self.avg_life}"
        )
        self.print(
            f"The percentage of extinct alleles after {self.n_gen} generations is: {self.per_extinct}"
        )

        self.avg_freq = self.simulation / (
            self._initial_population.shape[1] * self._initial_population.shape[2]
        )

        self.extinction = np.average(_sum == 0, axis=0)
        self.fixation = np.average(_sum == self.N * 2, axis=0)

    def count_snp_n(self, snp_id: int):
        # sum the counts over both strands + all individuals
        return self.simulation[snp_id, :]

    def reproduce(self, mom, dad):
        mom_gamete = self.__meiosis(self.L, mom.copy())
        dad_gamete = self.__meiosis(self.L, dad.copy())
        return np.stack(
            [mom_gamete[:, self.rand_strand], dad_gamete[:, self.rand_strand]], axis=-1
        )

    @staticmethod
    def __meiosis(L, arr):
        k = np.random.randint(L, size=1)[0]
        temp_slice = arr[k:, 0].copy()
        arr[k:, 0] = arr[k:, 1].copy()
        arr[k:, 1] = temp_slice
        return arr

    @property
    def rand_strand(self):
        return np.random.choice([0, 1], p=[0.5, 0.5])

    def print(self, message: str):
        if self.verbose:
            print(message)
