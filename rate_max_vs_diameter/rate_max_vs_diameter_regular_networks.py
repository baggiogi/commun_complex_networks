#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
    File name: rate_max_vs_diameter_regular_networks.py
    Author: Giacomo Baggio
    Email: baggio.giacomo@gmail.com
    Description: Compute and Plot Max Shannon Transmission Rate vs. Network Diameter for Non-normal Chain/2D Lattice "Layered" Architectures
    Date created: 26/01/2018
    Date last modified: 23/02/2019
    Python Version: 2.7
"""

import autograd.numpy as np
import numpy as np
import scipy
import networkx as nx
import matplotlib.pyplot as plt

from pymanopt.manifolds import Euclidean
from pymanopt import Problem
from pymanopt.solvers import TrustRegions

plt.close("all")

# max network dimension (number of nodes for chain network and number of nodes per side in 2D lattice network)
n_max = 10
# rewiring probability
p_rew = 0.1
# directionality parameter
alpha = 5.
# distance from instability
beta = 2.

# noise variance
sigma2 = 10.

# vector of network dimensions
n_vec = range(1,n_max)
# vector of max transmission rates
rate_max_vec = np.zeros(len(n_vec))
# vector of network diameters
diam_vec = np.zeros(len(n_vec))

for n in n_vec:

    print "dimension:", n
    
    ###### Building a non-normal "layered" chain or 2D lattice network ######
    
    #G = nx.path_graph(n) # uncomment for chain network
    G = nx.grid_2d_graph(n,n,periodic=False) # uncomment for 2D lattice network
    
    
    pos = dict( (n, n) for n in G.nodes() )
    labels = dict( ((i, j), i * n + j) for i, j in G.nodes() )
    G = nx.relabel_nodes(G,labels) 
    
    #nodes = list(range(n)) # uncomment for chain network
    nodes = list(range(n*n)) # uncomment for 2D lattice network
    
    #I = np.identity(n) # uncomment for chain network
    I = np.identity(n*n) # uncomment for 2D lattice network

    # periphery of the graph    
    per = nx.periphery(G)
    # diameter of the graph
    diam = nx.diameter(G)
    
    # pick to nodes in the periphery of the graph
    input_nodes = []
    output_nodes = []    
    for i in range(0,n*n):
        for j in range(0,n*n):
            if nx.shortest_path_length(G,i,j) == diam:
                input_nodes.append(i)
                output_nodes.append(j)
                        
    input_nodes = [input_nodes[0]]
    output_nodes = [output_nodes[0]] 
    
    # Define ouput matrix C
    C = I
    # uncomment if you want to select a subset of readout nodes
    #C=np.zeros((n,n)) # uncomment for chain network
    #C=np.zeros((n*n,n*n)) # uncomment for 2D lattice network
    #for kk in range(0,len(output_nodes)):
    #    C[output_nodes[kk],output_nodes[kk]]=1.
    
    # Define input matrix B
    #B=I # uncomment if you want to select all nodes
    # B = np.zeros((n,n)) # uncomment for chainnetwork
    B = np.zeros((n*n,n*n)) # uncomment for 2D lattice network
    for kk in range(0,len(input_nodes)):
        B[input_nodes[kk],input_nodes[kk]]=1./len(input_nodes)
    
    # "directed stratification" procedure with directionality parameter alpha
    G = G.to_directed()
    
    for j in input_nodes:
        for i in range(0,n*n):
            if i not in input_nodes:
                all_paths = [pth for pth in nx.all_shortest_paths(G,source=j,target=i)]
                for kk in range(0,len(all_paths)):
                    input_nodes_tmp = input_nodes[:]
                    input_nodes_tmp.remove(j)
                    if not input_nodes_tmp:
                        G.add_path(all_paths[kk],weight=alpha)
                    else:
                        if not bool(set(input_nodes_tmp).intersection(all_paths[kk])):
                            G.add_path(all_paths[kk],weight=alpha)
    
    A = nx.adjacency_matrix(G)
    A = A.todense()
    A = A.transpose()
    
    for i in range(0,n*n):
        for j in range(0,n*n):
            if A[i,j]==alpha:
                A[j,i]=1/A[i,j]
                
    for j in output_nodes:
        for i in output_nodes:
            if i!=j and A[i,j]!=0:
                A[i,j]=1

    # stabilize network matrix (beta is the distance from instability)
    eigA, eigvA = np.linalg.eig(A) 
    gamma = max(np.real(eigA)) 
    A = A - (gamma + beta)*I
    A = np.array(A)
    
    #### compute max Shannon transmission rate ####
        
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
    
    rate_max_vec[n] = rate
    diam_vec[n] = diam
    
    
# plot max transmission rate vs. network diameter
plt.plot(diam_vec,rate_max_vec, 'ro')
plt.xlabel('Diameter')
plt.ylabel('Max Transmission rate')
plt.show()
