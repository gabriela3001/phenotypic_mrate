import numpy as np
from aspergillus_msb import *
import pickle
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
import time
from collections import Counter
import sys

param = int(sys.argv[1])
noise = int(sys.argv[2])
L = int(sys.argv[3])

output = '/home/labs/pilpel/gabril/ShortNKLandscapes/'

with open(output+'NK_landscapes_6.txt', 'rb') as f:
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

with open(output+'MSB_NK_landscape_'+str(L)+'.txt', 'rb') as f:
    results = pickle.load(f)
with open(output+'complex_adaptation_NK'+str(L)+'_param_'+str(P)+str(noise)+'.txt', 'rb') as f:
    complex = pickle.load(f)  

def impose_pop_mrate(mut_mrate, pop):
    nonmut_pop = (pop[:64]*(1-mut_mrate)) /np.sum(pop[:64])
    mut_pop = (pop[64:]*mut_mrate) /np.sum(pop[64:])                                     
    return(np.hstack((nonmut_pop, mut_pop)))

def sim_annealing_simulation(mrate_matrix,param,noise, ngen):
    
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
        #adaptation_results[rep] = impose_pop_mrate(mrate_matrix[0][rep]/1000000, adaptation_results[rep])
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
        for rep in np.arange(nreps):
            adaptation_results[rep] = impose_pop_mrate(mrate_matrix[t][rep]/1000000, adaptation_results[rep])
        adaptation_results /= np.sum(adaptation_results, axis = 1, keepdims = True)

        for rep in np.arange(nreps):
            adaptation_results[rep] = np.random.multinomial(pop_size, adaptation_results[rep])
            adaptation_results[rep] = impose_pop_mrate(mrate_matrix[t][rep]/1000000, adaptation_results[rep])
            evol_mpf[t, rep] = np.dot(adaptation_results[rep], fitness_vector) 
            evol_argmax[t, rep] = np.argmax(adaptation_results[rep][:64]+adaptation_results[rep][64:])
            evol_mrate[t, rep] = np.sum(adaptation_results[rep][64:])
        t += 1

        if t % 1000 == 0:
            print(t)
            #print(evol_argmax[t])

    elapsed_time = time.process_time() - time_start
    
    return({'evol_mpf':evol_mpf, 'evol_max':evol_argmax, 'evol_mrate':evol_mrate})

sim_annealing_result = sim_annealing_simulation(complex['evol_mrate'],param,noise, 50000)    

output = '/home/labs/pilpel/gabril/ShortNKLandscapes/'
    
with open(output+'simulated_annealing_NK'+str(L)+'_'+str(noise)+'_'+str(param)+'.txt', 'wb') as f:
    pickle.dump(sim_annealing_result, f)
