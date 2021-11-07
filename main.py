import random
import numpy as np
import time
import math
# import maplotlib.pyplot as plt

BIT_SIZE = 6
POPULATION_SIZE = 10
SIMULATION_RUNS = 5


class Chromosome:
    def __init__(self, bit: str, gen: int):
        self.bit: str = bit
        self.gen: int = gen
        self.fitness: float = 0
        self.x: int = 0
        self.value: int = 0
        self.parentX: Chromosome = None
        self.parentY: Chromosome = None

    def set_fitness(self, fitness: float):
        self.fitness: float = fitness

    def set_value(self, value: int):
        self.value: int = value

    def set_x(self, x: int):
        self.x: int = x

    def set_parent(self, x, y):
        self.parentX = x
        self.parentY = y

    def __repr__(self) -> str:
        parent = ""
        if self.parentX != None:
            parent = ', p1: {}, p2: {}'.format(
                self.parentX.bit, self.parentY.bit)
        return 'bit: {}, x: {:>3}, gen: {:>3}, fitness: {:.5f}, value: {:6} {}'.format(self.bit, self.x, self.gen, self.fitness, self.value, parent)


def formula(x: int):
    return (x ** 3) - (60 * (x ** 2)) + (900 * x) + 150


def fitness_cal(population: list(), min: bool = True):
    total = 0
    for i in population:
        i.set_x(string_decode(i.bit))
        y = formula(i.x)
        i.set_value(y)
        total = total + y
    for i in population:
        fit = 1 - (i.value / total) if min else (i.value / total)
        i.set_fitness(fit)


def crossover(p1: Chromosome, p2: Chromosome, point: list(), gen: int):
    child_chromosome = ""
    for i in range(len(p1.bit)):
        if i not in point:
            child_chromosome += p1.bit[i]
            continue
        child_chromosome += p2.bit[i]
    child = Chromosome(bit=child_chromosome, gen=gen)
    child.set_parent(p1, p2)
    return child


def output(data: list(), title=""):
    print(title)
    for i in data:
        print(i)


# parent selection : SUS
def parent_selection(candidate: list(), n: int):
    for i in range(1, len(candidate)):
        candidate[i].set_fitness(candidate[i].fitness + candidate[i-1].fitness)
    F = candidate[-1].fitness
    P = F / n  # split small size
    start = random.uniform(0, P)
    parent = []
    count_p = 0  # count parent
    count_c = 0  # count candidate
    p_location = [start + (i * P) for i in range(n)]

    while count_p != n:
        if candidate[count_c].fitness > p_location[count_p]:
            parent.append(candidate[count_c])
            count_p += 1
            continue
        count_c += 1
    # position = random.sample(range(0, len(candidate) - 1), n)
    # parent = list()
    # for i in range(len(candidate)):
    #     if i in position:
    #         parent.append(candidate[i])

    return parent


def mutation(chromosome: list(), n: int):
    position = random.sample(range(0, len(chromosome) - 1), n)
    for i in position:
        p_g = random.randint(0, 5)
        gene = chromosome[i].bit[p_g]
        new_gene = '0' if gene == '1' else '1'
        new_chromosome = chromosome[i].bit[:p_g] + \
            new_gene + chromosome[i].bit[p_g+1:]
        chromosome[i].bit = new_chromosome


def genetic_algorithm(population_size: int, crossover_rate: float = 0.8, crossover_point: list = [1, 3], mutation_rate: float = 0.2, min: bool = True):
    gen = 0
    repeat = 0
    gen_list = []
    crossover_size = math.floor(crossover_rate * population_size)
    mutation_size = math.floor(mutation_rate * population_size)

    first_gen = create_first_gen(population_size)
    fitness_cal(first_gen, min=min)
    first_gen.sort(key=lambda x: x.fitness, reverse=True)
    gen_list.append(first_gen)
    output(first_gen, "Gen 0 : Population")

    old = first_gen[0].x
    while repeat < 5:
        gen += 1
        prev_gen = gen_list[-1]
        present_gen = prev_gen[0:population_size - crossover_size]
        parent = parent_selection(prev_gen, crossover_size)

        for i in range(crossover_size):
            parentX, parentY = parent[i], parent[-i]
            child = crossover(parentX, parentY, crossover_point, gen)
            present_gen.append(child)

        mutation(present_gen, mutation_size)

        fitness_cal(present_gen, min=min)
        present_gen.sort(key=lambda x: x.fitness, reverse=True)
        gen_list.append(present_gen)

        output(present_gen, "Gen {} : Population".format(gen))
        new = present_gen[0].x

        if old == new:
            repeat += 1
        else:
            repeat = 0
            old = new
        # print(repeat)

    return gen_list[-1][0], len(gen_list)


def create_first_gen(population_size: int):
    pop_list = [[random.randint(0, 1) for i in range(BIT_SIZE)]
                for j in range(population_size)]

    pop = []
    for i in pop_list:
        pop.append(Chromosome(''.join(str(j) for j in i), 0))
    return pop


def string_decode(s: str) -> int:
    return int(r'0b{}'.format(s), 2) if s != '000000' else 64


def string_encode(i: int) -> str:
    if i == 64:
        return '000000'
    result = bin(i)[2:]
    return result.rjust(BIT_SIZE).replace(' ', '0')


if __name__ == "__main__":
    random.seed(0)

    best_list = []
    gen_list = []
    time_list = []
    for i in range(SIMULATION_RUNS):
        start = time.time()
        best_chromosome, gen_stop = genetic_algorithm(
            POPULATION_SIZE)
        stop = time.time() - start
        best_list.append(best_chromosome)
        gen_list.append(gen_stop)
        time_list.append(stop)
        print()

    print('\nSummary\n')
    for i in range(SIMULATION_RUNS):
        print('time: {} - chromosome: {} - x: {:>3} - stop: {:>3}'.format(i+1,
              best_list[i].bit, best_list[i].x, gen_list[i] - 1))
        print('take time: {}'.format(time_list[i]))
    print('\naverage time: {}'.format(sum(time_list) / SIMULATION_RUNS))
