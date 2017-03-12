# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 18:38:59 2017

Simulations

@author: marc
"""

from __future__ import division
import os
import numpy as np
import importlib as imp
import random

# Set working directory to location of this file
try:
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
except:
    pass


from mnielsen_network import Network
from own_wrapper import training_data
from own_wrapper import test_data



# Simulations


lr = np.array([2**x for x in range(10)], dtype = 'float') / 20


### 1 layer

## Take mini-batch size of 10, 30 hidden neurons in 1 layer, as in example in book

random.seed(111)
out = np.zeros((20, 10))
for i in range(10):
    net = Network([256, 30, 10])
    out[:, i] = net.SGD(training_data, 20, 10, lr[i], test_data=test_data)
    print(i)
out[19]  

## Take mini-batch size of 10, 20 hidden neurons in 1 layer

random.seed(111)
out1 = np.zeros((20, 10))
for i in range(10):
    net = Network([256, 20, 10])
    out1[:, i] = net.SGD(training_data, 20, 10, lr[i], test_data=test_data)
    print(i)
out1[19]  

## Take mini-batch size of 10, 10 hidden neurons in 1 layer

random.seed(111)
out1_1 = np.zeros((20, 10))
for i in range(10):
    net = Network([256, 10, 10])
    out1_1[:, i] = net.SGD(training_data, 20, 10, lr[i], test_data=test_data)
    print(i)
out1_1[19]  

## Take mini-batch size of 10, 15 hidden neurons in 1 layer

random.seed(111)
out1_2 = np.zeros((50, 10))
for i in range(10):
    net = Network([256, 15, 10])
    out1_2[:, i] = net.SGD(training_data, 50, 10, lr[i], test_data=test_data)
    print(i)
out1_2[13]  
# winner: 89.0% accuracy after 13 epochs

## Take mini-batch size of 10, 18 hidden neurons in 1 layer

random.seed(111)
out1_3 = np.zeros((50, 10))
for i in range(10):
    net = Network([256, 18, 10])
    out1_3[:, i] = net.SGD(training_data, 50, 10, lr[i], test_data=test_data)
    print(i)
out1_3[43]  
# rank 2: 88.6% accuracy after 43 epochs


## Take mini-batch size of 10, 100 hidden neurons in 1 layer as in example in book

random.seed(111)
out2 = np.zeros((20, 10))
for i in range(10):
    net = Network([256, 100, 10])
    out2[:, i] = net.SGD(training_data, 20, 10, lr[i], test_data=test_data)
    print(i)
out2[19]

### 3 layers

## Take mini-batch size of 10, 5/2/5 hidden neurons

random.seed(111)
out3 = np.zeros((20, 10))
for i in range(10):
    net = Network([256, 5, 2, 5, 10])
    out3[:, i] = net.SGD(training_data, 20, 10, lr[i], test_data=test_data)
    print(i)
out3[19]


### 2 layers

## Take mini-batch size of 10, 30/30 hidden neurons

random.seed(111)
out4 = np.zeros((20, 10))
for i in range(10):
    net = Network([256, 30, 30, 10])
    out4[:, i] = net.SGD(training_data, 20, 10, lr[i], test_data=test_data)
    print(i)
out4[19]

## Take mini-batch size of 10, 10/10 hidden neurons

random.seed(111)
out5 = np.zeros((20, 10))
for i in range(10):
    net = Network([256, 10, 10, 10])
    out5[:, i] = net.SGD(training_data, 20, 10, lr[i], test_data=test_data)
    print(i)
out5[19]



