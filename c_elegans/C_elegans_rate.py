#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
    File name: C_elegans_rate.py
    Author: Giacomo Baggio
    Email: baggio.giacomo@gmail.com
    Description: Computation of the information rate for the chemical synapse network of the nematode C. elegans. The network data is taken from: 
    Varshney, Lav R., et al. "Structural properties of the Caenorhabditis elegans neuronal network." PLoS computational biology 7.2 (2011).
    Date created: 29/12/2019
    Date last modified: 26/01/2020
    Python Version: 2.7
"""

import autograd.numpy as np
import scipy
import warnings
import matplotlib.pyplot as plt

from pymanopt.manifolds import Euclidean
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent

from random import random

plt.close("all")

ITER_LIMIT = 10000
LYAPUNOV_EPSILON = 1.0E-6

# output noise variance (sigma^2)
sigma2 = 1.
# network dimension
n = 279

# n-dim identity matrix
I = np.identity(n)

# sampling period
Ts = 0.1
# max transmission time window
T_bar = 2.
# vector of time windows
T_vec = np.arange(Ts,T_bar,Ts)
# number of trials
M = 100
# vector of data
rate_data = np.zeros((len(T_vec),M))

# input matrix
B = I 
# output matrix
C = I

for pp in range(M):
    
    # upload C. elegans connectivity matrix (insert full file path)
    A = np.loadtxt("insert_full_path/A_C_elegans.txt",usecols=range(n))

    for tt in range(279):
        for qq in range(279):
            if tt>qq:
                if random() < 0.5:
                    A_tmp = A[tt,qq]
                    A[tt,qq] = A[qq,tt]
                    A[qq,tt] = A_tmp
             
    w,v = np.linalg.eig(A)
    A = A - (max(np.real(w))+0.1)*I

    # vector of transmission rates
    rate_vec = np.zeros(len(T_vec))

    k = 0

    # plot Shannon transmission rate R_T over time
    for T in T_vec:
        
            ### compute Shannon transmission rate ###
        
            def dlyap_iterative(a,q,eps=LYAPUNOV_EPSILON,iter_limit=ITER_LIMIT):
                error = 1E+6
        
                x = q
                ap = a
                apt = a.transpose()
                last_change = 1.0E+10
                count = 1

                (m,n) = a.shape
                if m != n:
                    raise ValueError("input 'a' must be square") 
    
                det_a = np.linalg.det(a)
                if det_a > 1.0:
                    raise ValueError("input 'a' must have eigenvalues within the unit circle") 
    
                while error > LYAPUNOV_EPSILON and count < iter_limit:
                    change = ap*q*apt
            
                    x = x + change
        
                    ap = ap*a
                    apt = apt*(a.transpose())
                    error = abs(change.max())
                    count = count + 1

                if count >= iter_limit:
                    warnings.warn('lyap_solve: iteration limit reached - no convergence',RuntimeWarning)
            
                return x
            
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
    
            # function returning Shannon transmission rate
            def shannon_rate(L): 
                Sigma = np.dot(L,L.transpose())
                Sigma_norm = Sigma/(np.trace(Sigma))
                Sigma_norm_B = np.dot(np.dot(B,Sigma_norm),B.transpose())
                Wcd = dlyap_iterative(Ad.transpose(),Sigma_norm_B)
                Wnum = np.dot(Wo,Wcd)
                Wden = np.dot(Wo,Wcd-Sigma_norm_B)
                num = np.linalg.det(Wnum/sigma2+I)
                den = np.linalg.det(Wden/sigma2+I)
                return -np.log2(num/den)/(2.*T)
                    
            # instantiate manifold for Pymanopt
            manifold_fact = Euclidean(n,n)
            # instantiate problem for Pymanopt
            problem = Problem(manifold=manifold_fact, cost=shannon_rate, verbosity=0)
            # instantiate solver for Pymanopt
            solver = SteepestDescent()
            # let Pymanopt do the rest
            L_opt = solver.solve(problem)
            
            # optimal covariance Sigma^*
            Sigma_opt = np.dot(L_opt,L_opt.transpose())
            Sigma_opt_norm = Sigma_opt/(np.trace(Sigma_opt))
            
            # value of transmission rate
            rate = -shannon_rate(L_opt)
            print "Shannon transmission rate:", rate   
            rate_vec[k] = rate
            rate_data[k,pp] = rate
            
            k = k+1
            
    # plot transmission rate vs. transmission window
    plt.plot(T_vec,rate_vec, 'bo')
    plt.xlabel('Transmission window')
    plt.ylabel('Transmission rate')
    plt.show()
    pause(0.001)