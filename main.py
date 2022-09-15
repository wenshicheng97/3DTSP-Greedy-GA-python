import numpy as np
import matplotlib.pyplot as plt
import time


def read_file(file_name):
    f = open(file_name)
    cities_num = int(f.readline())
    str_lst = list()
    location_lst = list()
    for i in range(cities_num):
        line = f.readline()
        location_lst.append(list(map(int, line.split())))
    return cities_num, np.array(location_lst)


def initialize_population(population_size, cities_num):
    initial_population = []
    cities_seq = np.array(range(cities_num))
    for i in range(population_size):
        np.random.shuffle(cities_seq)
        initial_population.append(cities_seq.copy())
    return np.array(initial_population)


def initial_path_by_greedy(cities_num, location_lst):
    dist = np.zeros(shape=(cities_num, cities_num))
    for i in range(cities_num):
        for j in range(cities_num):
            if i == j:
                dist[i, j] = 0
            else:
                dist[i, j] = np.linalg.norm(location_lst[i] - location_lst[j])
    best_path = []
    best_dist = np.inf
    for start_point in range(cities_num):
        flag = True
        cur_path = [start_point]
        cur_dist = 0
        for i in range(cities_num - 1):
            nearest = np.argsort(dist[cur_path[-1]])
            for j in nearest:
                if j not in cur_path:
                    cur_dist += dist[cur_path[-1]][j]
                    cur_path.append(j)
                    break
            if cur_dist > best_dist:
                flag = False
                break
        if flag:
            cur_dist += dist[cur_path[-1]][start_point]
            if cur_dist < best_dist:
                best_dist = cur_dist
                best_path = cur_path.copy()
    return np.array(best_path), best_dist


def compute_distance(cities_num, path, location_lst):
    distance = 0
    for i in range(cities_num - 1):
        distance += np.linalg.norm(location_lst[path[i]] - location_lst[path[i + 1]])
    distance += np.linalg.norm(location_lst[path[cities_num - 1]] - location_lst[path[0]])
    return distance


def get_fitness(cities_num, population, location_lst):
    distances = []
    for path in population:
        distances.append(compute_distance(cities_num, path, location_lst))
    distances = np.array(distances)
    fitness = 1. / distances
    return fitness


def create_mating_pool(population_size, population, fitness):
    probability = fitness / sum(fitness)
    roulette_wheel = probability.cumsum()
    mating_pool_index = []
    for i in range(population_size):
        p = np.random.rand()
        for j in range(population_size):
            if p < roulette_wheel[0]:
                mating_pool_index.append(0)
                break
            elif roulette_wheel[j] < p <= roulette_wheel[j + 1]:
                mating_pool_index.append(j + 1)
                break
    mating_pool = population[mating_pool_index].copy()
    return mating_pool


def cross_over(parent_1, parent_2, start_index, end_index, cities_num):
    child = parent_2.copy()
    for i in range(start_index, end_index + 1):
        cur = parent_1[i]
        cur_child = child.copy()
        child[i] = cur
        duplicated = np.argwhere(child == cur)
        if len(duplicated) > 1:
            child[duplicated[duplicated != i]] = cur_child[i]
    return child


def cross_population(mating_pool, cur_best, cities_num, cross_prob):
    new_population = []
    for mate in mating_pool:
        if cross_prob >= np.random.rand():
            start_index = np.random.randint(cities_num)
            end_index = np.random.randint(cities_num)
            while start_index == end_index:
                end_index = np.random.randint(cities_num)
                start_index, end_index = min(start_index, end_index), max(start_index, end_index)
            child = cross_over(cur_best, mate, start_index, end_index, cities_num)
            new_population.append(child)
        else:
            new_population.append(mate)
    return np.array(new_population)


def mutation(individual, start_index, end_index):
    mutated_ind = individual.copy()
    mutated_ind[start_index:end_index] = mutated_ind[start_index:end_index][::-1]
    return mutated_ind


def mutate_population(population, cities_num, mutate_prob):
    new_population = []
    for ind in population:
        if mutate_prob >= np.random.rand():
            start_index = np.random.randint(cities_num)
            end_index = np.random.randint(cities_num)
            while start_index == end_index:
                end_index = np.random.randint(cities_num)
                start_index, end_index = min(start_index, end_index), max(start_index, end_index)
                new_population.append(mutation(ind, start_index, end_index))
        else:
            new_population.append(ind)
    return np.array(new_population)


def display_location(loc):
    return str(loc[0]) + ' ' + str(loc[1]) + ' ' + str(loc[2]) + '\n'


def main():
    cities_num, location_lst = read_file("input.txt")
    population_size = 200
    iteration = 200
    cross_prob = 0.95
    mutate_prob = 0.5

    population = initialize_population(population_size, cities_num)
    fitness = get_fitness(cities_num, population, location_lst)
    cur_best_path, cur_best_dist = initial_path_by_greedy(cities_num, location_lst)
    cur_best_fit = 1 / cur_best_dist

    best_dis_list = [cur_best_dist]

    for i in range(iteration):
        mating_pool = create_mating_pool(population_size, population, fitness)

        population = cross_population(mating_pool, cur_best_path, cities_num, cross_prob)

        population = mutate_population(population, cities_num, mutate_prob)

        fitness = get_fitness(cities_num, population, location_lst)

        best_index = np.argmax(fitness)
        if fitness[best_index] < cur_best_fit:
            population[best_index] = cur_best_path
            fitness[best_index] = cur_best_fit
        else:
            cur_best_path = population[best_index].copy()
            cur_best_fit = fitness[best_index].copy()

        best_dis_list.append(1 / cur_best_fit)

    with open('output.txt', 'w') as fout:
        for c in cur_best_path:
            fout.write(display_location(location_lst[c]))
        fout.write(display_location(location_lst[cur_best_path[0]]))

    print(compute_distance(cities_num, cur_best_path, location_lst))

    plt.plot(best_dis_list)
    plt.show()


if __name__ == '__main__':
    s = time.time()
    main()
    t = time.time()
    print(str(t-s), 'seconds')

