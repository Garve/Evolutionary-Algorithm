import numpy as np
from abc import ABC, abstractmethod


class Individual(ABC):
    def __init__(self, value=None, **kwargs):
        if value is not None:
            self.value = value
        else:
            self.value = self._random_init(**kwargs)

    @abstractmethod
    def __add__(self, other):
        pass

    @abstractmethod
    def mutate(self, rate):
        pass

    @abstractmethod
    def _random_init(self, **kwargs):
        pass


class OneDimensionalOptimization(Individual):
    def __add__(self, other):
        return OneDimensionalOptimization(0.5 * self.value + 0.5 * other.value)

    def mutate(self, rate=1):
        self.value += np.random.normal(0, rate)

    def _random_init(self, bound):
        return np.random.uniform(-bound, bound)


class ThreeDimensionalOptimization(Individual):
    def __add__(self, other):
        return ThreeDimensionalOptimization(0.5 * self.value + 0.5 * other.value)

    def mutate(self, rate=1):
        self.value += np.random.normal(0, rate, 3)

    def _random_init(self, bound):
        return np.random.uniform(-bound, bound, 3)


class TSP(Individual):
    def __add__(self, other):
        self_head = self.value[:len(self.value) // 2].copy()
        self_tail = self.value[len(self.value) // 2:].copy()
        other_tail = other.value[len(other.value) // 2:].copy()

        mapping = {other_tail[i]: self_tail[i] for i in range(len(self_tail))}

        for i in range(len(self_head)):
            while self_head[i] in other_tail:
                self_head[i] = mapping[self_head[i]]

        return TSP(np.hstack([self_head, other_tail]))

    def mutate(self, rate=1):
        for _ in range(rate):
            i, j = np.random.choice(range(len(self.value)), 2, replace=False)
            self.value[i], self.value[j] = self.value[j], self.value[i]

    def _random_init(self, n_cities, **kwargs):
        return np.random.choice(range(n_cities), n_cities, replace=False)


class Pool:
    def __init__(self, individual_class, size, fitness, **kwargs):
        self.size = size
        self.fitness = fitness
        self.individuals = [individual_class(**kwargs) for _ in range(size)]
        self.individuals.sort(key=lambda x: self.fitness(x))

    def replace(self, new_individuals):
        self.individuals.extend(new_individuals)
        self.individuals.sort(key=lambda x: self.fitness(x))
        self.individuals = self.individuals[-self.size:]

    def get_parents(self, n_offsprings):
        mothers = self.individuals[-2 * n_offsprings::2]
        fathers = self.individuals[-2 * n_offsprings + 1::2]

        return mothers, fathers


class Evolution:
    def __init__(self, individual_class, pool_size, fitness, **kwargs):
        self.pool = Pool(individual_class, pool_size, fitness, **kwargs)

    def _step(self, n_offsprings, rate=1):
        mothers, fathers = self.pool.get_parents(n_offsprings)
        offsprings = []

        for mother, father in zip(mothers, fathers):
            offspring = mother + father
            offspring.mutate(rate)
            offsprings.append(offspring)

        self.pool.replace(offsprings)
