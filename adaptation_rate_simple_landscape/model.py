from itertools import product
import numpy as np
from scipy.stats import poisson
from scipy.stats import binom

def noise_matrix(noise_baseline, noise_elevated):
    noise_m = np.zeros((2, 2))
    
    noise_m[0,0] = 1-noise_baseline
    noise_m[0,1] = noise_baseline
    noise_m[1,0] = noise_elevated
    noise_m[1,1] = 1-noise_elevated
    
    return(noise_m)

def background_vector(mu, nbackground):
    distribution_deleterious = poisson(mu)
    return([distribution_deleterious.pmf(x) for x in range(nbackground)])

def strain_matrix_3(mu):
    strain_matrix = [[(1-mu)**2, 2*mu*(1-mu), mu**2], [0, 1-mu, mu], [0, 0, 1]]
    strain_matrix = np.array(strain_matrix)
    
    return(strain_matrix)

def mutation_rate_transition_V(mu, tau, nbackground, noise_baseline, noise_elevated, beta = 5000):
    
    nmrates = 2
    
    all_genotypes = []
    for i in range(3):
        for j in range(nmrates):
            for k in range(nbackground):
                all_genotypes.append((i,j,k))
    
    mutator_background = background_vector(mu*tau, nbackground)
    baseline_background = background_vector(mu, nbackground)
    background = [baseline_background, mutator_background]
    
    noise_transition = noise_matrix(noise_baseline, noise_elevated)
    
    mutator_strain = strain_matrix_3((mu*tau)/beta)
    baseline_strain = strain_matrix_3(mu/beta)
    strain = [baseline_strain, mutator_strain]
    
    transition = np.zeros((len(all_genotypes), len(all_genotypes)))
    
    # each element (i,j) of the transition matrix gives the transition probability from i to j
    
    for i in range(len(all_genotypes)):
        
        for j in range(len(all_genotypes)):
            
            from_g = all_genotypes[i]
            to_g = all_genotypes[j]
            
            f = 1
            
            # deleterious background mutations
            if from_g[2] > to_g[2]:
                f = 0  
            else:
                del_mut = to_g[2] - from_g[2]
                f *= background[from_g[1]][del_mut]
                            
            # strain
            f *= strain[from_g[1]][from_g[0]][to_g[0]]
            
            # mutation rate type
            f *= noise_transition[from_g[1]][to_g[1]]

            transition[i][j] = f  

    return(transition)
    
    