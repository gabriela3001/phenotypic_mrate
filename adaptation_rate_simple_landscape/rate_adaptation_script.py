from model import *
from msb import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import pickle
import time
import sys

def proportion_adapted(all_genotypes, pop):
    pa = np.sum([pop[i] for i in range(len(pop)) if (all_genotypes[i][0] == 2 and all_genotypes[i][2] == 0)])
    return(pa)

noise_levels = np.logspace(-6, np.log10(0.5), 30)

genotype_names_3 = []
for i in range(3):
    for j in range(2):
        for k in range(5):
            genotype_names_3.append((i,j,k))

with open('results_MSB_paramgrid.pkl', 'rb') as f:
    param_grid, results_symmetric = pickle.load(f)
	
param = 6

nreps = 500
pop_size = float(10000000)
ncat = len(results_symmetric[param][0]['pop'])

result_per_noise = []
noise = int(sys.argv[1])


transition_matrix = mutation_rate_transition_V(param_grid[param]['mu'], param_grid[param]['tau'], 5, noise_levels[noise], noise_levels[noise], beta = 5000)
fitness_vector = fitness(genotype_names_3, param_grid[param]['s'], 5, 'V')
msb_pop = results_symmetric[param][noise]['pop']

results = np.zeros((nreps, ncat))

for rep in range(nreps):
    results[rep] = np.random.multinomial(pop_size, msb_pop)

results /= np.sum(results, axis = 1, keepdims = 1)

# we want: array of times of first appearance; fate of each mutation
t = 0
update_fates = np.array([False]*nreps)
adaptation_times = []

while sum(update_fates) / nreps < 0.95:

    t += 1

    results = results @ transition_matrix
    results *= fitness_vector

    results /= np.sum(results, axis = 1, keepdims = True)


    for rep in np.arange(nreps)[~update_fates]:

        results[rep] = np.random.multinomial(pop_size, results[rep])

        if ((results[rep][20]+results[rep][25])/pop_size > 0.99):
            adaptation_times.append(t)
            update_fates[rep] = True
            print(t)

result_per_noise.append(adaptation_times)
print('Finished Noise ' + str(noise) + 'with mean ' + str(1/np.mean(adaptation_times)))


output_loc = '/home/labs/pilpel/gabril/Rate of Adaptation/'

with open(output_loc + 'results_adaptation_'+str(noise)+'.txt', 'wb') as f:
    pickle.dump(adaptation_times, f)