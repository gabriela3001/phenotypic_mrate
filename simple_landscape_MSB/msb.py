import numpy as np
from model import *

def population_generator(all_genotypes, initial):
    
    pop = [0 for _ in range(len(all_genotypes))]
    pop[all_genotypes.index(initial)] = 1.
    
    return(np.array(pop))
    
def fitness(all_genotypes, s, H, mode):
    
    f_vec = []
    
    if mode == 'MSB':
        
        for genotype in all_genotypes:
            
            f_vec.append((1-s)**(genotype[0]+genotype[2]))
            
    elif mode == 'V':
        
        strain_fitness = [1,1-s,1+s*H]
        
        for genotype in all_genotypes:
            
            f_vec.append(((1-s)**genotype[2])*strain_fitness[genotype[0]])
            
    return(f_vec)
    
    
def msb_simulation(genotype_names, mode, s, H, mu, tau, noise_baseline, noise_elevated):

    population = population_generator(genotype_names, (0,0,0))
    fitness_vector = fitness(genotype_names, s, H, mode)
    mpf = np.sum(population*fitness_vector)

    transition_mutations = mutation_rate_transition_V(mu, tau, 5, noise_baseline, noise_elevated, beta = 5000)

    time = 0
    previous_mpf = 20.

    # reaching mutation selection balance
    while previous_mpf - mpf != 0:
        time += 1
        previous_mpf = mpf.copy()
        population = np.dot(population, transition_mutations)
        population *= fitness_vector
        population /= np.sum(population)
        mpf = np.sum(population*fitness_vector)

    msb_dict = {'pm': proportion_mutator(genotype_names, population), 'mpf': mpf, 'time': time, 'pop': population}
    
    return(msb_dict)
    
    
def proportion_mutator(all_genotypes,pop):
    pm = np.sum([pop[i] for i in range(len(pop)) if all_genotypes[i][1] == 1])
    return(pm)