#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
    File name: rate_max_vs_noise_regular_networks.py
    Author: Giacomo Baggio
    Email: baggio.giacomo@gmail.com
    Description: Compute and Plot Max Shannon Transmission Rate vs. Noise Level for Regular Network Topologies
    Date created: 15/02/2018
    Date last modified: 21/02/2019
    Python Version: 2.7
"""

import autograd.numpy as np
import scipy
import matplotlib.pyplot as plt
import networkx as nx

from pymanopt.manifolds import Euclidean
from pymanopt import Problem
from pymanopt.solvers import TrustRegions

plt.close("all")

# network dimension
n = 8
# n-dim identity matrix
I = np.identity(n)
# input matrix
B = I
# output matrix
C = I

# network parameters
gamma = -3.
alpha = 5.

# sampling interval noise
sigma2_s = 1.
# min exponent noise level (log10 basis)
sigma2_min = -6.
# max exponent noise level (log10 basis)
sigma2_max = 6.
# vector of noise levels
sigma2_vec = 10**np.arange(sigma2_min,sigma2_max,sigma2_s)
# vector of max transmission rates
rate_max_vec = np.zeros(len(sigma2_vec))

# connectivity matrix A, and output matrix C
# generate a desired network architecture using NetworkX (e.g. n-dim chain network)
G = nx.path_graph(n)

# get the network adjacency matrix
A = nx.to_numpy_matrix(G)

# generate non-normal structure using parameter \alpha
for i in range(n):
    for j in range(n):
        if i > j:
            A[i,j] = A[i,j]*alpha
            A[j,i] = A[j,i]/alpha

# enforce stability via self-loop weights
A = gamma*I+A
# print resulting adjacency matrix
print(A)


k = 0

# plot max Shannon transmission rate R_max over noise level
for sigma2 in sigma2_vec: 

    #### compute Shannon transmission rate ####
        
    # function returning finite-horizon observability Gramian
    def gramian_ode(y, t0, A, C):
        temp = np.dot(scipy.linalg.expm(A.transpose()*t0),C.transpose())
        dQ = np.dot(temp,temp.transpose())      
        return dQ.reshape((A.shape[0]**2,1))[:,0]
    
    # function returning Shannon transmission rate at given time T
    def shannon_rate(L, T): 
        Ad = scipy.linalg.expm(A*T)
        y0 = np.zeros([A.shape[0]**2,1])[:,0]
        out = scipy.integrate.odeint(gramian_ode, y0, [0,T], args=(A,C))
        Wo = out[1,:].reshape([A.shape[0], A.shape[0]])
        lyap_tmp = - np.kron(Ad,Ad) + np.identity(n*n)
        Sigma = np.dot(L,L.transpose())
        Sigma_norm = Sigma/(np.trace(Sigma))
        Sigma_norm_B = np.dot(np.dot(B,Sigma_norm),B.transpose())
        Sigma_norm_vec = np.reshape(Sigma_norm_B,(n*n,1))
        Wcd_vec = np.dot(np.linalg.inv(lyap_tmp),Sigma_norm_vec)
        Wcd = np.reshape(Wcd_vec,(n,n))
        Wnum = np.dot(Wo,Wcd)
        Wden = np.dot(Wo,Wcd-Sigma_norm_B)
        num = np.linalg.det(Wnum+sigma2*I)
        den = np.linalg.det(Wden+sigma2*I)
        return -np.log2(num/den)/(2.*T)
        
    def make_shannon_capacity_fact_fixed_time(T): 
        def shannon_capacity_fact_fixed_time(L):
            return shannon_rate(L, T)  
        return shannon_capacity_fact_fixed_time
       
    # instantiate manifold for Pymanopt               
    manifold_fact = Euclidean(n,n)
    # instantiate solver for Pymanopt
    solver = TrustRegions()
    
        
    #### find R_max via bisection method ####
    
    # min value T
    T1=1e-8
    shannon_capacity_fact_fixed_time = make_shannon_capacity_fact_fixed_time(T1)
    problem = Problem(manifold=manifold_fact, cost=shannon_capacity_fact_fixed_time, verbosity=0)
    L_opt1 = solver.solve(problem)
    p_1 = -shannon_capacity_fact_fixed_time(L_opt1)
    #print p_1
    
    Sigma_opt1 = np.dot(L_opt1,L_opt1.transpose())
    Sigma_opt_norm1 = Sigma_opt1/(np.trace(Sigma_opt1)) 
    
    eigs, eigvs = np.linalg.eig(Sigma_opt_norm1)
    eff_rank1 = np.square(np.sum(np.real(eigs)))/np.sum(np.square(np.real(eigs)))
    
    # max value T
    T2=10.
    shannon_capacity_fact_fixed_time = make_shannon_capacity_fact_fixed_time(T2)
    problem = Problem(manifold=manifold_fact, cost=shannon_capacity_fact_fixed_time, verbosity=0)
    L_opt2 = solver.solve(problem)
    p_2 = -shannon_capacity_fact_fixed_time(L_opt2)
    #print p_2
    
    # mean value T
    Tm = (T1+T2)/2
    shannon_capacity_fact_fixed_time = make_shannon_capacity_fact_fixed_time(Tm)
    problem = Problem(manifold=manifold_fact, cost=shannon_capacity_fact_fixed_time, verbosity=0)
    L_optm = solver.solve(problem)
    p_m = -shannon_capacity_fact_fixed_time(L_optm)
    #print p_m
    
    # bisection method
    u = 0
    while np.abs(T2-T1)>0.01:
        
        Tl = (T1+Tm)/2
        shannon_capacity_fact_fixed_time = make_shannon_capacity_fact_fixed_time(Tl)
        problem = Problem(manifold=manifold_fact, cost=shannon_capacity_fact_fixed_time, verbosity=0)
        L_optl = solver.solve(problem)
        p_l = -shannon_capacity_fact_fixed_time(L_optl)
        
        #print p_l
        
        Tr = (T2+Tm)/2
        shannon_capacity_fact_fixed_time = make_shannon_capacity_fact_fixed_time(Tr)
        problem = Problem(manifold=manifold_fact, cost=shannon_capacity_fact_fixed_time, verbosity=0)
        L_optr = solver.solve(problem)
        p_r = -shannon_capacity_fact_fixed_time(L_optr)
        
        #print p_r
        
        p_min = max(p_1,p_2,p_m,p_l,p_r)
        
        if p_min==p_1 or p_min==p_l:
            T2=Tm; Tm=Tl; p_b=p_m; p_m=p_l
        elif p_min==p_m:
            T1=Tl; T2=Tr; p_a=p_l; p_b=p_r
        else:
            T1=Tm; Tm=Tr; p_a=p_m; p_b=p_r
        
        u=u+1
        #print u
    
    
    shannon_capacity_fact_fixed_time = make_shannon_capacity_fact_fixed_time(Tm)
    # instantiate problem for Pymanopt
    problem = Problem(manifold=manifold_fact, cost=shannon_capacity_fact_fixed_time, verbosity=0)
    # let Pymanopt do the rest
    L_opt = solver.solve(problem)
    
    # optimal covariance Sigma^*
    Sigma_opt = np.dot(L_opt,L_opt.transpose())
    Sigma_opt_norm = Sigma_opt/(np.trace(Sigma_opt))    
    
    # value of max transmission rate
    rate = -shannon_capacity_fact_fixed_time(L_opt)    
    
    rate_max_vec[k] = rate
    k = k+1
    
    print"noise:", sigma2
    print"Sigma opt:", Sigma_opt_norm
    print"Max rate:", rate
    print"Max time:", Tm
    

# plot max transmission rate vs. noise level (log-log scale)
plt.loglog(sigma2_vec,rate_max_vec, 'ro')
plt.xlabel('Noise variance')
plt.ylabel('Max transmission rate')
plt.show()