import numpy as np
import scipy.stats
import pickle
import itertools
from scipy.spatial.distance import hamming

def construct_transition_matrix(all_genotypes, landscape, gamma, mu, tau):

    genotypes = [tuple(x) for x in landscape.keys()]
    transition_matrix = np.zeros((len(all_genotypes), len(all_genotypes)))
    nneighbours = [len([x for x in genotypes if int(hamming(x, all_genotypes[0][1])*len(genotypes[0])) == d]) for d in range(len(all_genotypes[0][1])+1)]
    nneighbours_dict = dict(zip(range(len(all_genotypes[0][1])+1), nneighbours))

    for i in range(len(all_genotypes)):
    #for i in range(1):
        for j in range(len(all_genotypes)):
            p = 1
            # switching in mutation rate level
            if all_genotypes[i][0] == all_genotypes[j][0]:
                p *= (1-gamma)
            if all_genotypes[i][0] != all_genotypes[j][0]:
                p *= gamma

            # mutation 
            distance = int(hamming(all_genotypes[i][1], all_genotypes[j][1])*len(all_genotypes[i][1]))
            #nneighbours = len([x for x in genotypes if int(hamming(x, all_genotypes[j][1])*len(all_genotypes[i][1])) == distance])

            if all_genotypes[i][0] == 0:
                p *= scipy.stats.binom.pmf(distance, len(all_genotypes[i][1]), mu) / nneighbours_dict[distance]
            if all_genotypes[i][0] == 1:
                p *= scipy.stats.binom.pmf(distance, len(all_genotypes[i][1]), tau*mu) / nneighbours_dict[distance]

            transition_matrix[i][j] = p
            
    return(transition_matrix)

def construct_selection_matrix(all_genotypes, fitness):
    fitness_values = []
    for g in all_genotypes:
        fitness_values.append(fitness[g[1]])
    return(fitness_values)

def construct_selection_matrix_msb(all_genotypes, fitness, initial_genotype):
    fitness_values = []
    for g in all_genotypes:
        if hamming(list(initial_genotype), list(g[1])) > 1/len(initial_genotype):
            all_values = np.array(list(fitness.values()))
            minval = np.min(all_values[np.nonzero(all_values)])
            fitness_values.append(minval)
        else:
            fitness_values.append(fitness[g[1]])
    return(fitness_values)

def define_initial_population(all_genotypes, initial_genotype, initial_mrate = 0):
    population = np.zeros(len(all_genotypes))
    initial_index = all_genotypes.index((initial_mrate, tuple(initial_genotype)))
    population[initial_index] = 1.
    return(population)

def simulation_msb(all_genotypes, landscape, gamma, mu, tau, initial_genotype):
    transition_matrix = construct_transition_matrix(all_genotypes, landscape, gamma, mu, tau)
    population = define_initial_population(all_genotypes, initial_genotype)
    fitness_vector = construct_selection_matrix_msb(all_genotypes, landscape, initial_genotype)
    mpf = np.dot(population, fitness_vector)
    
    mpf_evol = []

    time = 0
    previous_mpf = 20.

    while previous_mpf - mpf != 0:
        time += 1
        previous_mpf = mpf.copy()
        population = np.dot(population, transition_matrix)
        population *= fitness_vector
        population /= np.sum(population)
        mpf = np.dot(population, fitness_vector)
        mpf_evol.append(np.dot(population, fitness_vector))
        
    return({'time': time, 'pop': population, 'mpf': mpf})


def mutator_single_mut(all_genotypes, population, initial_genotype):
    p = 0
    for g in range(len(all_genotypes)):
        if all_genotypes[g][0] == 1 and all_genotypes[g][1] != tuple(initial_genotype):
            p += population[g]
    return(p)