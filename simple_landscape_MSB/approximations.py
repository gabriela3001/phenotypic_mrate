import numpy as np

# helper functions

def inverse_function(x, p, a):
    return(p/x + a)

def desai_fisher_fitness(mu, tau, s):
    M = mu * (tau-1)
    return(M-M*(M/s))

def desai_fisher(noise, mu, tau):
    return(noise/(mu*(tau-1)))
    
    
# Frequency of mutators, defined as M0 + M1

def P_function(noise, mu, tau, s):
    
    switch = 0.5 * mu * (tau-1)
    
    if noise > switch:
        return(1-inverse_function(noise, desai_fisher_fitness(mu, tau, s)*s, 0.5))
    else: 
        return(noise/(mu*(tau-1)))  
    
    
# Frequency of mutants, defined as m1 + M1

def S_function(noise, mu, tau, s):
    #K, p0 = 1-(0.5*mu + 0.5*mu*tau) / s, 1-mu/s 
    K, p0 = np.exp(-(0.5*mu + 0.5*mu*tau) / s), np.exp(-mu/s)
    r = 2/((tau)*mu + mu)
    num = K * p0
    den = p0+(K-p0)*np.exp(-r*noise)
    
    return((num/den))

def mean_pop_fitness(noise, mu, tau, sel):
    
    mut = S_function(noise, mu, tau, sel)
    mpf = 1 - sel*mut
    
    return(mpf)
    
# Ratio of Non-Mutator Mutants (m1) to Non-Mutator WT (m0)

def R_function(noise, mu, tau, s):
    
    K, p0 = 1-np.exp(-(0.5*mu + 0.5*mu*tau) / s), 1-np.exp(-mu/s)
    
    #r = mean_pop_fitness(noise, mu, tau, s) / s
    r = np.log(tau)/s
    #r = desai_fisher(mu, tau, s)
    
    num = K * p0
    den = p0+(K-p0)*np.exp(-r*noise)
    
    return((num/den))

    
# Functions to calculate frequencies of m0, m1, M0, M1 at MSB

def calculate_M1(noise, mu, tau, s):
    
    d = 1-S_function(noise, mu, tau, s) - calculate_m1(noise, mu, tau, s)
    
    return(d)
    
def calculate_M0(noise, mu, tau, s):
    
    c = P_function(noise, mu, tau, s) - calculate_M1(noise, mu, tau, s)
    
    return(c)
    
def calculate_m0(noise, mu, tau, s):

    a = 1-P_function(noise, mu, tau, s)
    b = calculate_m1(noise, mu, tau, s)
    
    return(a-b)
    
def calculate_m1(noise, mu, tau, s):
    
    den = 1+R_function(noise, mu, tau, s)
    num = R_function(noise, mu, tau, s) * (1-P_function(noise, mu, tau, s))
    
    return(num/den)