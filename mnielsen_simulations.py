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

## Take mini-batch size of 10, 30 hidden neurons as in example in book

### vary learning rate
net = Network([256, 30, 10])

lr = np.array([2**x for x in range(10)], dtype = 'float') / 20
out = np.zeros((20, 10))


for i in range(10):
    net = Network([256, 30, 10])
    out[:, i] = net.SGD(training_data, 20, 10, lr[i], test_data=test_data)
    print(i)
    












