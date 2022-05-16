import numpy as np
from aspergillus_msb import *
import pickle
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
import time
from collections import Counter
import sys

L = int(sys.argv[1])
P = int(sys.argv[2])

path = '/home/labs/pilpel/gabril/ShortNKLandscapes/'

with open(path+'NK_landscapes_6.txt', 'rb') as f:
    landscapes = pickle.load(f)
    
landscape = landscapes[L]

# double mutant adaptation
def DM_adaptation(all_genotypes, population, initial_genotype, landscape):
    p = 0
    for g in all_genotypes:
        if (hamming(g[1], list(initial_genotype)) == 2/len(initial_genotype)) and landscape["".join(g[1])]>landscape["".join(initial_genotype)]:
            p += population[all_genotypes.index(g)]
    return(p)
    
genotypes = [tuple(x) for x in landscape.keys()]

all_genotypes = []
for i in range(2):
    for g in genotypes:
        all_genotypes.append((i, g))
        
params = {'mu': [1e-7, 1e-6, 1e-5], 'tau': [10, 100]}
params = list(ParameterGrid(params))
indexes = {1:0, 3:1, 5:2}
initial_genotypes = [tuple('111100'), tuple('100111'), tuple('100101')] 
initial_genotype = initial_genotypes[indexes[L]]
noise_levels = np.logspace(-6,np.log10(0.5), 30)

with open(path+'MSB_NK_landscape_'+str(L)+'param_'+str(P)+'.txt', 'rb') as f:
    results = pickle.load(f)

def complex_adaptation_simulation(param,noise,ngen):
    
    # initialise population, transition matrix, fitness matrix
    msb_pop = results[param]['result'][noise]['pop']
    transition_matrix = construct_transition_matrix(all_genotypes, landscape, noise_levels[noise], results[param]['mu'], results[param]['tau'])
    fitness_vector = construct_selection_matrix(all_genotypes, landscape)
    
    # initialise nreps, pop_size, ncat
    nreps = 500
    pop_size = 1000000
    ncat = len(all_genotypes)
    adaptation_results = np.zeros((nreps, ncat))
    
    for rep in range(nreps):
        adaptation_results[rep] = np.random.multinomial(pop_size, msb_pop)
    adaptation_results /= np.sum(adaptation_results, axis = 1, keepdims = 1)
    
    t = 0
    time_start = time.process_time()

    ngen = 50000

    evol_mpf = np.zeros((ngen, nreps))
    evol_argmax = np.zeros((ngen, nreps))
    evol_mrate = np.zeros((ngen, nreps))


    while t < ngen:

        adaptation_results = adaptation_results @ transition_matrix
        adaptation_results *= fitness_vector
        adaptation_results /= np.sum(adaptation_results, axis = 1, keepdims = True)

        for rep in np.arange(nreps):
            adaptation_results[rep] = np.random.multinomial(pop_size, adaptation_results[rep])
            evol_mpf[t, rep] = np.dot(adaptation_results[rep], fitness_vector) / pop_size
            evol_argmax[t, rep] = np.argmax(adaptation_results[rep][:64]+adaptation_results[rep][64:])
            evol_mrate[t, rep] = np.sum(adaptation_results[rep][64:])
        t += 1

        if t % 1000 == 0:
            print(t)

    elapsed_time = time.process_time() - time_start
    
    return({'evol_mpf':evol_mpf, 'evol_max':evol_argmax, 'evol_mrate':evol_mrate})
    
all_complex_results = []
for noise in [0,18,29]:
    print('Noise: ', noise)
    complex_adaptation_result = complex_adaptation_simulation(P,noise,50000)
    all_complex_results.append(complex_adaptation_result)
    
    output = '/home/labs/pilpel/gabril/ShortNKLandscapes/'
    
    with open(output+'complex_adaptation_NK3_'+str(noise)+'.txt', 'wb') as f:
        pickle.dump(complex_adaptation_result, f)
    
