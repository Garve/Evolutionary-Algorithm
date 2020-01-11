import numpy as np
from evo import Evolution, TSP
import matplotlib.pyplot as plt

n_cities = 20
cities = np.random.randint(0, n_cities ** 2, size=(n_cities, 2))
matrix = []
for city in cities:
    row = []
    for city_ in cities:
        row.append(np.linalg.norm(city - city_))
    matrix.append(row)
distances = np.array(matrix)


def fitness(tsp):
    res = 0
    for i in range(len(tsp.value)):
        res += distances[tsp.value[i], tsp.value[(i + 1) % len(tsp.value)]]
    return -res


e = Evolution(TSP, 1000, fitness, n_cities=n_cities)
hist = []

for i in range(100):
    hist.append(e.pool.fitness(e.pool.individuals[-1]))
    e._step(100)

plt.plot(hist)
plt.show()

plt.figure(figsize=(12, 8))
plt.scatter(x=cities[:, 0], y=cities[:, 1])
solution = e.pool.individuals[-1]
for i in range(len(cities)):
    plt.plot([cities[solution.value[i]][0], cities[solution.value[(i + 1) % len(cities)]][0]],
             [cities[solution.value[i]][1], cities[solution.value[(i + 1) % len(cities)]][1]], 'r')
plt.show()
