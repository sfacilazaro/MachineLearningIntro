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
#import sklearn as skl
from sklearn.model_selection import train_test_split
import tensorflow as tf
#import pandas as pd
import os
import sys
import subprocess


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
    a = 100 - sum(sum(np.abs(np.round(pred) - targ)))/N * 100
    return a

def err(pred,targ,N):
    ran = np.random.uniform(0,1,(np.shape(targ)))
    ERR = sum(sum(0.5*(pred-targ)*(pred-targ)))/N
    REF = sum(sum(0.5*(ran -targ)*(ran -targ)))/N
    return ERR/REF

def SillyLittleNeuron(p,L,M,eta0,nIT):
    """
    Data --- matrix cointaining the training input set for the netwrok, each column is a different input
    val ---- correct result that we want to predict
    p ------ module number
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
        
    #file = open('EV.txt','w')
    
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
        #file.write('%03d\t%0.5f\n' % (it, ac))

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
        
    #file.close()   
    
    #EV = np.loadtxt('EV.txt', dtype='float', delimiter='\t')
    #plt.plot(EV[:,0],EV[:,1],'.')    
    
    return acc(hat,val,N)

def ObscureLittleNeuron(p,L,M,nIT):
    """
    Data --- matrix cointaining the training input set for the netwrok, each column is a different input
    val ---- correct result that we want to predict
    p ------ module number
    M ------ number of neurons per hidden layer (not including bias)
    N ------ number of examples in the training set
    L,S----- length of each example of the training set
    nIT ---- number of iterations for backprop
    """
    
    #DOR = 0.2 #drop out rate
    seed = np.random.randint(0,nIT*nIT)
    
    name = f'Data-p{p:02d}-S{L:04d}.txt'
    Data = np.loadtxt(name, dtype='float', delimiter='\t')
    
    name = f'val-p{p:02d}-S{L:04d}.txt'
    val = np.loadtxt(name, dtype='float', delimiter='\t')
    
    S, N = np.shape(Data)
    
    DataT = np.transpose(Data)
    
    # load and preprocess your data (x and y)
    #xTrain, xTest, yTrain, yTest = skl.model_selection.train_test_split(np.transpose(Data), val, test_size = 0.2, random_state = seed)
    xTrain, xTest, yTrain, yTest = train_test_split(DataT, val, test_size = 0.2, random_state = seed)
    
    lilneu = tf.keras.Sequential([
        tf.keras.layers.Dense(M, activation='relu', input_shape=(S,)),
        #tf.keras.layers.Dropout(DOR),
        #tf.keras.layers.Dense(M, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid') #'softmax', 'relu', 'sigmoid'
    ])

    lilneu.compile(
        loss='binary_crossentropy', #'categorical_crossentropy' - energy function
        optimizer='adam',            #'SGD' stochastic gradient descent, 'RMSprop', 'adam' - optimizer algorithm
        metrics=['mean_squared_error']        #evaluation measure to track the training
    )
    
    # Save the original stdout
    original_stdout = sys.stdout
    
    # Redirect stdout to /dev/null (or 'nul' on Windows)
    with open(os.devnull, 'w') as fnull:
        sys.stdout = fnull
        history = lilneu.fit(x=DataT, y=val, epochs=nIT, batch_size=len(val))
        hat = lilneu.predict(DataT)
        
    # Restore the original stdout
    sys.stdout = original_stdout

    val = np.reshape(val,(N,1))
    
    return acc(hat,val,N)

def MyLittleComp(p,L):
    
    #p = 37
    #L = 1000
    M = 50
    eta0 = 0.1
    nIT = 1000
    nSTT = 250 #1000
    
    #sillyList = []
    obscrList = []
    
    for n in range(nSTT):
        prog = f'progress at {100*n/nSTT}%'
        print(prog)
        #sillyAcc = SillyLittleNeuron(p,L,M,eta0,nIT)
        obscrAcc = ObscureLittleNeuron(p,L,M,nIT)
        #sillyList.append(sillyAcc)
        obscrList.append(obscrAcc)
    
    #sillyArr = np.array(sillyList)
    obscrArr = np.array(obscrList)

    #name = f'SillyStat-p{p:02d}-S{L:04d}-STT{nSTT:05d}.txt'
    #np.savetxt(name, sillyArr, delimiter='\t');
    
    name = f'ObscrStat-p{p:02d}-S{L:04d}-STT{nSTT:05d}.txt'
    np.savetxt(name, obscrArr, delimiter='\t');
    
def complot(p,L,nSTT):
    
    name = f'SillyStat-p{p:02d}-S{L:04d}-STT{nSTT:05d}.txt'
    sillyData = np.loadtxt(name, dtype='float', delimiter='\t')
    
    name = f'ObscrStat-p{p:02d}-S{L:04d}-STT{nSTT:05d}.txt'
    obscrData = np.loadtxt(name, dtype='float', delimiter='\t')
    
    # Create a new figure
    plt.figure(figsize=(8, 4))
    
    # Plot the histograms
    plt.hist(sillyData, bins=15, alpha=0.5, label='Silly' ,  edgecolor='black')
    plt.hist(obscrData, bins=45, alpha=0.5, label='Obscure', edgecolor='black')
    
    # Add labels and title
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
    plt.title('Accuracy at 1000 steps')
    
    # Add legend
    plt.legend()
    
    # Show the histograms
    plt.show()
    
    
