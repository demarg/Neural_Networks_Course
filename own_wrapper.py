# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 19:21:32 2017

@author: marc
"""

from __future__ import division
import os
import numpy as np

# Set working directory to location of this file
try:
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
except:
    pass

test_in = np.loadtxt("data/test_in.csv", delimiter = ',')
test_out = np.loadtxt("data/tets_out.csv", delimiter = ',')
train_in = np.loadtxt("data/train_in.csv", delimiter = ',')
train_out = np.loadtxt("data/train_out.csv", delimiter = ',')

def TupleUp(set_in, set_out):
    out = []
    for i in range(set_in.shape[0]):
        set_in_mat = np.reshape(set_in[i], (-1, 16))
        set_tuple = (set_in_mat, int(set_out[i]))
        out.append(set_tuple)   
    return out
    
test = TupleUp(test_in, test_out)
train = TupleUp(train_in, train_out)

