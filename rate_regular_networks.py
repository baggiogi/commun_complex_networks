### Plot Shannon Transmission Rate vs. Transmission Time for Regular Network Topologies

import autograd.numpy as np
import scipy
import matplotlib.pyplot as plt
import networkx as nx

from pymanopt.manifolds import Euclidean
from pymanopt import Problem
from pymanopt.solvers import TrustRegions


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

# network parameters
gamma = -3.
alpha = 5.

# sampling period
Ts = 0.1
# max transmission time window
T_bar = 10.
# vector of time windows
T_vec = np.arange(Ts,T_bar,Ts)
# vector of transmission rates
rate_vec = np.zeros(len(T_vec))

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

# plot Shannon transmission rate R_T over time
for T in T_vec:
        
        #### compute Shannon transmission rate ###
        
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
        def shannon_rate(B): 
            Sigma = np.dot(B,B.transpose())
            Sigma_norm = Sigma/(np.trace(Sigma))
            Sigma_norm_vec = np.reshape(Sigma_norm,(n*n,1))
            Wcd_vec = np.dot(np.linalg.inv(lyap_tmp),Sigma_norm_vec)
            Wcd = np.reshape(Wcd_vec,(n,n))
            Wnum = np.dot(Wo,Wcd)
            Wden = np.dot(Wo,Wcd-Sigma_norm)
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
        B_opt = solver.solve(problem)
        
        # optimal covariance Sigma^*
        Sigma_opt = np.dot(B_opt,B_opt.transpose())
        Sigma_opt_norm = Sigma_opt/(np.trace(Sigma_opt))
        print("Optimal input covariance:")
        print(Sigma_opt_norm)
        
        # value of transmission rate
        rate = -shannon_rate(B_opt)
        print("Shannon transmission rate:")
        print(rate)   
        rate_vec[k] = rate
        
        k = k+1
        

# plot transmission rate vs. time
plt.plot(T_vec,rate_vec, 'ro')
plt.xlabel('Time window (T)')
plt.ylabel('Transmission rate')
plt.show()
        