import numpy as np
import random

# add a random number


def add_random_binary(row, random_value):
    binary_str = ''.join(str(x) for x in row)
    decimal = int(binary_str, 2)
    new_decimal = decimal + random_value
    new_binary_str = bin(new_decimal)[2:]  # remove '0b' prefix
    new_binary = [int(x) for x in new_binary_str]
    return new_binary[:len(binary_str)]


# add a noise provided by an argument

def create_random_trigger_specified_size(row_length, trigger_size):
    modification_order = np.argsort(tuple(random.random() for _ in range(row_length)))
    trigger = row_length * [None]
    for position in modification_order[:trigger_size]:
        trigger[position] = 1
    return trigger


def apply_trigger(row, trigger):
    for i in range(len(row)):
        if trigger[i] is not None:
            row[i] = trigger[i]
    return row


def add_noise_trigger(row, trigger):
    row_with_noise = np.clip(row + trigger, 0, 1)
    return np.round(row_with_noise).astype(int)


def gaussian_noise(length, std_dev=0.1):
    return np.random.normal(0, std_dev, length)


def uniform_noise(length, range=0.1):
    return np.random.uniform(-range, range, length)


# Genetic algorithm methods
def initialize_population(number_of_features, trigger_size, population_size=None):
    """Initializes a population of size population_size with triggers of size trigger_size"""
    population_size = population_size or trigger_size
    population = [[0] * number_of_features for _ in range(population_size)]

    for i in range(population_size):
        modification_order = np.argsort(uniform_noise(number_of_features))
        for position in modification_order[:trigger_size]:
            population[i][position] = 1

    return population


def integrate_trigger(training_set, trigger):
    """Integrates the trigger into the training set"""
    X, y = training_set
    poisoned_training_set = np.array([np.array([max(row[i], trigger[i]) for i in range(len(row))]) for row in X.values])

    return poisoned_training_set, y


def mutation(population, mutation_probability):
    """Mutation operation with probability mutation_probability"""
    for row in population:
        if random.random() < mutation_probability:
            row_size = len(row)
            for i in range(row_size):
                if random.random() < mutation_probability:
                    row[i] = 1 - row[i]
    return population


def crossover(population, crossover_probability):
    """Crossover operation with probability crossover_probability"""
    population_size = len(population)
    for i in range(population_size):
        if random.random() < crossover_probability:
            j = int(random.random() * population_size)
            row_size = len(population[i])
            crossover_point = int(random.random() * row_size)
            population[i][crossover_point:] = population[j][crossover_point:]
    return population


def create_genetic_trigger(trigger_size, training_set, retrain_model, epsilon=0.001, crossover_probability=0.8,
                           mutation_probability=0.03):
    """
    Creates a trigger using a genetic algorithm. Based on the paper
    "Backdoor Attack on Machine Learning Based Android Malware Detectors. / Li, Chaoran; Chen, Xiao; Wang, Derui et al."

    Attributes:
    feature_weights: the weights of the features in the model, used to control evolution direction
    trigger_size: the size of the trigger
    training_set: the training set used to poison with candidate triggers and evaluate their effectiveness
    epsilon: the threshold for the difference between the last two generations
    crossover_probability: the probability of crossover operation
    mutation_probability: the probability of mutation operation
    retrain_model: a function that takes a training set and returns the weights of the features in the retrained model
    """
    feature_weights = retrain_model(training_set)
    number_of_weights = len(feature_weights)
    number_of_features = training_set[0].shape[1]
    population = initialize_population(number_of_features, trigger_size)
    population_size = len(population)

    cur_feature_weights_delta = [0] * number_of_weights

    trigger = None

    while True:
        prev_feature_weights_delta = cur_feature_weights_delta
        for i in range(population_size):
            trigger = population[i]
            training_set = integrate_trigger(training_set, trigger)
            cur_feature_weights = retrain_model(training_set)
            cur_feature_weights_delta = [abs(cur_feature_weights[i] - feature_weights[i]) for i in
                                         range(number_of_weights)]

        population = crossover(population, crossover_probability)
        population = mutation(population, mutation_probability)

        if abs(max(prev_feature_weights_delta) - max(cur_feature_weights_delta)) <= epsilon:
            break

    return trigger
