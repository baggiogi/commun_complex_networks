#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
    File name: rate_random_networks.py
    Author: Giacomo Baggio
    Email: baggio.giacomo@gmail.com
    Description: Compute and Plot Shannon Transmission Rate vs. Transmission Window for Random Network Topologies
    Date created: 18/01/2018
    Date last modified: 21/02/2019
    Python Version: 2.7
"""

import autograd.numpy as np
import scipy
import math
import matplotlib.pyplot as plt

from pymanopt.manifolds import Euclidean
from pymanopt import Problem
from pymanopt.solvers import TrustRegions

from scipy.stats import invwishart

plt.close("all")

# output noise variance (sigma^2)
sigma2 = 1. 
# network dimension
n = 10
# n-dim identity matrix
I = np.identity(n)
# input matrix
B = I
# output matrix
C = I

# sampling period
Ts = 0.1
# max transmission time window
T_bar = 10.
# vector of time windows
T_vec = np.arange(Ts,T_bar,Ts)
# vector of transmission rates
rate_vec = np.zeros(len(T_vec))

# generate a stable random network topology using the matrix parametrization described in
# Hennequin, Aitchison, and Lengyel, "Fast sampling-based inference in balanced neuronal networks". NIPS, p. 2240–2248, 2014.
nu = n-1+math.floor(1./0.2)
omega = nu-n-1
scale_mat = I/omega
InvWish = invwishart(df=math.floor(nu), scale=scale_mat) 
S_tmp = np.random.normal(0, 1, (n,n))
S = S_tmp - S_tmp.transpose()
LLT = InvWish.rvs()
A = np.dot(-I+S,LLT)

# print resulting adjacency matrix
print(A)

k = 0

# plot Shannon transmission rate R_T over time
for T in T_vec:
        
        ### compute Shannon transmission rate ###
        
        Ad = scipy.linalg.expm(A*T)
        
        # function returning finite-horizon observability Gramian
        def gramian_ode(y, t0, A, C):
            temp = np.dot(scipy.linalg.expm(A.transpose()*t0),C.transpose())
            dQ = np.dot(temp,temp.transpose())      
            return dQ.reshape((A.shape[0]**2,1))[:,0]
     
        y0 = np.zeros([A.shape[0]**2,1])[:,0]
        out = scipy.integrate.odeint(gramian_ode, y0, [0,T], args=(A,C))
        # finite-horizon observability Gramian
        Wo = out[1,:].reshape([A.shape[0], A.shape[0]])
        
        lyap_tmp = - np.kron(Ad,Ad) + np.identity(n*n)

        # function returning Shannon transmission rate
        def shannon_rate(L): 
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
                  
        # instantiate manifold for Pymanopt
        manifold_fact = Euclidean(n,n)
        # instantiate problem for Pymanopt
        problem = Problem(manifold=manifold_fact, cost=shannon_rate, verbosity=0)
        # instantiate solver for Pymanopt
        solver = TrustRegions()
        # let Pymanopt do the rest
        L_opt = solver.solve(problem)
        
        # optimal covariance Sigma^*
        Sigma_opt = np.dot(L_opt,L_opt.transpose())
        Sigma_opt_norm = Sigma_opt/(np.trace(Sigma_opt))
        print"Sigma opt:", Sigma_opt_norm
        
        # value of transmission rate
        rate = -shannon_rate(L_opt)
        print "Shannon transmission rate:", rate   
        rate_vec[k] = rate
        
        k = k+1
        
# plot transmission rate vs. transmission window
plt.plot(T_vec,rate_vec, 'ro')
plt.xlabel('Transmission window')
plt.ylabel('Transmission rate')
plt.show()