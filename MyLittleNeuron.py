#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 10:41:42 2023

@author: sergio
"""

import numpy as np
#import scipy as sp
from itertools import combinations
import matplotlib.pyplot as plt
#import seaborn as sb
#import pandas as pd
#import sklearn as skl

def S(x):
    return 1/(1+np.exp(-x))

def fibo(a,b,n):
    if n == 0:
        return a
    elif n == 1:
        return b
    else:
        return fibo(a, b, n-1) + fibo(a, b, n-2)
    
def genDataFibo(q,S):
    C = np.array(list(combinations(range(q), 2)))
    N = len(C)
    
    v = np.concatenate( (np.ones(N), np.zeros(N)) )
    
    F = np.zeros((N,S))
    F[:,(0,1)] = np.array(C)
    
    for s in range(2,S):
        F[:,s] = F[:,s-1] + F[:,s-2] 
    F = F % q
    F = np.round(F).astype(int)
    
    R = np.random.randnint(0,q,(N,S))
    M = np.concatenate((F,R))
    
    ind = np.random.permutation(np.arange(2*N))
    v = v[ind]
    M = M[ind,:]
    M = np.transpose(M)
    
    name = f'Data-p{q:02d}-S{S:04d}.txt'
    np.savetxt(name, M, delimiter='\t');
    
    name = f'val-p{q:02d}-S{S:04d}.txt'
    np.savetxt(name, v, delimiter='\t');
    
    return M, v

def acc(pred,targ,N):
    a = 100 - sum(sum(np.abs(np.round(pred) - val)))/len(np.transpose(val)) * 100
    return a

def err(pred,targ,N):
    ran = np.random.uniform(0,1,(np.shape(val)))
    ERR = sum(sum(0.5*(pred-val)*(pred-val)))/N
    REF = sum(sum(0.5*(ran -val)*(ran -val)))/N
    return ERR/REF

def lilneuron(p,L,M,eta0,nIT):
    """
    Data --- matrix cointaining the training input set for the netwrok, each column is a different input
    value -- correct result that we want to predict
    p ------ peairc number
    M ------ number of neurons per hidden layer (not including bias)
    N ------ number of examples in the training set
    L,S----- length of each example of the training set
    W ------ weight matrices
    lam ---- coupling of the penalty for overfitting
    eta ---- learning rate
    nIT ---- number of iterations for backprop
    """
    
    K = 1   #steepness of the sigmoid @ x = 0
    
    name = f'Data-p{p:02d}-S{L:04d}.txt'
    Data = np.loadtxt(name, dtype='float', delimiter='\t')
    
    name = f'val-p{p:02d}-S{L:04d}.txt'
    val = np.loadtxt(name, dtype='float', delimiter='\t')
    
    S, N = np.shape(Data)
    
    val = np.reshape(val,(1,N))
    
    
    #weight matrices, +1 to account for the bias
    W1 = np.random.randn(M,S+1)/10
    #W2 = np.random.randn(M,M+1)/10
    #W3 = np.random.randn(M,M+1)
    #W4 = np.random.randn(M,M+1)
    #W5 = np.random.randn(M,M+1)
    #W6 = np.random.randn(M,M+1)
    #W7 = np.random.randn(M,M+1)
    #W8 = np.random.randn(M,M+1)
    #W9 = np.random.randn(M,M+1)
    W2 = np.random.randn(1,M+1)/10
    
    ind = [n for n in range(1,M+1)]
        
    file = open('EV.txt','w')
    
    for it in range(nIT):
        
        #forward
        o0 = np.concatenate((np.reshape(np.ones(N),(1,N)), Data))
        o1 = np.concatenate((np.reshape(np.ones(N),(1,N)), 1/(1+np.exp(-K*np.dot(W1,o0))) ))
        #o2 = np.concatenate((np.reshape(np.ones(N),(1,N)), 1/(1+np.exp(-K*np.dot(W2,o1))) ))
        #o3 = np.concatenate((np.reshape(np.ones(N),(1,N)), 1/(1+np.exp(-K*np.dot(W3,o2))) ))
        #o4 = np.concatenate((np.reshape(np.ones(N),(1,N)), 1/(1+np.exp(-K*np.dot(W4,o3))) ))
        #o5 = np.concatenate((np.reshape(np.ones(N),(1,N)), 1/(1+np.exp(-K*np.dot(W5,o4))) ))
        #o6 = np.concatenate((np.reshape(np.ones(N),(1,N)), 1/(1+np.exp(-K*np.dot(W6,o5))) ))
        #o7 = np.concatenate((np.reshape(np.ones(N),(1,N)), 1/(1+np.exp(-K*np.dot(W7,o6))) ))
        #o8 = np.concatenate((np.reshape(np.ones(N),(1,N)), 1/(1+np.exp(-K*np.dot(W8,o7))) ))
        #o9 = np.concatenate((np.reshape(np.ones(N),(1,N)), 1/(1+np.exp(-K*np.dot(W9,o8))) ))
        o2 = 1/(1+np.exp(-K*np.dot(W2,o1)))
        hat = o2
        
        hat = np.reshape(hat,(1,N))
        
        ac = acc(hat,val,N)
        file.write('%03d\t%0.5f\n' % (it, ac))

        eta = 2 * eta0 * (1-ac/100)
        #eta = 2**(3/2) * eta0 * (1-ac/100) ** (3/2)
        
        #backprop
        d2 = hat - val #+ lam * (sum(sum(W1)) + sum(sum(W2)) + sum(sum(W3)) + sum(sum(W4)))
        d1 = K*o1*(1-o1)*np.dot(np.transpose(W2),d2)
        #d8 = K*o8*(1-o8)*np.dot(np.transpose(W9),d9[ind,:])
        #d7 = K*o7*(1-o7)*np.dot(np.transpose(W8),d8[ind,:])
        #d6 = K*o6*(1-o6)*np.dot(np.transpose(W7),d7[ind,:])
        #d5 = K*o5*(1-o5)*np.dot(np.transpose(W6),d6[ind,:])
        #d4 = K*o4*(1-o4)*np.dot(np.transpose(W5),d5[ind,:])
        #d3 = K*o3*(1-o3)*np.dot(np.transpose(W4),d4[ind,:])
        #d2 = K*o2*(1-o2)*np.dot(np.transpose(W3),d3[ind,:])
        #d1 = K*o1*(1-o1)*np.dot(np.transpose(W2),d2[ind,:])
        
        D1 = -eta/N * np.dot(d1[ind,:],np.transpose(o0))
        #D2 = -eta/N * np.dot(d2[ind,:],np.transpose(o1))
        #D3 = -eta/N * np.dot(d3[ind,:],np.transpose(o2))
        #D4 = -eta/N * np.dot(d4[ind,:],np.transpose(o3))
        #D5 = -eta/N * np.dot(d5[ind,:],np.transpose(o4))
        #D6 = -eta/N * np.dot(d6[ind,:],np.transpose(o5))
        #D7 = -eta/N * np.dot(d7[ind,:],np.transpose(o6))
        #D8 = -eta/N * np.dot(d8[ind,:],np.transpose(o7))
        #D9 = -eta/N * np.dot(d9[ind,:],np.transpose(o8))
        D2 = -eta/N * np.dot(d2,np.transpose(o1))
        
        #update the weights
        W1 += D1
        W2 += D2
        #W3 += D3
        #W4 += D4
        #W5 += D5
        #W6 += D6
        #W7 += D7
        #W8 += D8
        #W9 += D9
        #W10+= D10
        
    file.close()   
    
    EV = np.loadtxt('EV.txt', dtype='float', delimiter='\t')
    plt.plot(EV[:,0],EV[:,1],'.')    
    
    return hat, val
