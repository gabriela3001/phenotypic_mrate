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
C = int(sys.argv[4])
pM = int(sys.argv[5])

output = '/home/labs/pilpel/gabril/ShortNKLandscapes/'

L = 1

output = ''

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
with open(output+'complex_adaptation_NK'+str(L)+'_param_'+str(param)+str(noise)+'.txt', 'rb') as f:
    complex = pickle.load(f)  
    
def proportion_mutated(pop, genotype_names):
    p = 0
    for g in range(len(genotype_names)):
        if genotype_names[g][0] == 1:
            p += pop[g]
    return(p/np.sum(pop))

def proportion_pS(pop, genotype_names):
    major_genotype = genotype_names[np.argmax(pop)][1]
    p = 0
    for g in range(len(genotype_names)):
        if genotype_names[g][1] != major_genotype:
            p += pop[g]
    return(p/np.sum(pop))

def proportion_M1(pop, genotype_names):
    major_genotype = genotype_names[np.argmax(pop)][1]
    p = 0
    for g in range(len(genotype_names)):
        if genotype_names[g][1] != major_genotype and genotype_names[g][0] == 1:
            p += pop[g]
    return(p/np.sum(pop))
    
    
def impose_composition(pop, genotype_names_5, C, pM):
    
    start = time.time()

    M1 = C*proportion_pS(pop, genotype_names_5)
    m1 = (1-C)*proportion_pS(pop, genotype_names_5)
    M0 = pM-M1
    m0 = 1-M0-m1-M1
    
    end = time.time()
    
    start = time.time()

    ## assign each strain a category
    major_genotype = genotype_names_5[np.argmax(pop)][1]
    
    m0_strains = [True if ((genotype_names_5[g][0] == major_genotype) and (genotype_names_5[g][0] == 0)) else False for g in range(len(genotype_names_5))]
    M0_strains = [True if ((genotype_names_5[g][0] == major_genotype) and (genotype_names_5[g][0] == 1)) else False for g in range(len(genotype_names_5))]
    m1_strains = [True if ((genotype_names_5[g][0] != major_genotype) and (genotype_names_5[g][0] == 0)) else False for g in range(len(genotype_names_5))]
    M1_strains = [True if ((genotype_names_5[g][0] != major_genotype) and (genotype_names_5[g][0] == 1)) else False for g in range(len(genotype_names_5))]
    end = time.time()
    
    start = time.time()
    
    new_pop = pop.copy()
    
    if np.sum(pop[m0_strains]) != 0:
        new_pop[m0_strains] = (pop[m0_strains]*m0)/np.sum(pop[m0_strains])
    if np.sum(pop[m1_strains]) != 0:
        new_pop[m1_strains] = (pop[m1_strains]*m1)/np.sum(pop[m1_strains])
    if np.sum(pop[M0_strains]) != 0:
        new_pop[M0_strains] = (pop[M0_strains]*M0)/np.sum(pop[M0_strains])
    if np.sum(pop[M1_strains]) != 0:
        new_pop[M1_strains] = (pop[M1_strains]*M1)/np.sum(pop[M1_strains])
    
    return(new_pop)
    
    
def impose_pop_mrate(mut_mrate, pop):
    nonmut_pop = (pop[:64]*(1-mut_mrate)) /np.sum(pop[:64])
    mut_pop = (pop[64:]*mut_mrate) /np.sum(pop[64:])                                     
    return(np.hstack((nonmut_pop, mut_pop)))

def sim_annealing_simulation(param,noise, ngen, C, pM):
    
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
        adaptation_results[rep] = impose_composition(adaptation_results[rep], all_genotypes, C, pM)
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
            adaptation_results[rep] = impose_composition(adaptation_results[rep], all_genotypes, C, pM)
            evol_mpf[t, rep] = np.dot(adaptation_results[rep], fitness_vector) 
            evol_argmax[t, rep] = np.argmax(adaptation_results[rep][:64]+adaptation_results[rep][64:])
            evol_mrate[t, rep] = np.sum(adaptation_results[rep][64:])
        t += 1

        if t % 1000 == 0:
            print(t)
            #print(evol_argmax[t])

    elapsed_time = time.process_time() - time_start
    
    return({'evol_mpf':evol_mpf, 'evol_max':evol_argmax, 'evol_mrate':evol_mrate})

sim_annealing_result = sim_annealing_simulation(param,noise, 50000, C, pM) 

peaks = {1:39,3:58,5:61}
peak = peaks[L]
along_time = [list(sim_annealing_result['evol_max'][t]).count(peak)/500 for t in range(50000)]
output = '/home/labs/pilpel/gabril/ShortNKLandscapes/'
    
with open(output+'imposed_mrate_NK'+str(L)+'_'+str(noise)+'_'+str(param)+'_'+str(C)+'_'+str(pM)+'.txt', 'wb') as f:
    pickle.dump(sim_annealing_result, f)