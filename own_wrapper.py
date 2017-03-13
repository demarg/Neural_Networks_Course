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

train_out = train_out.astype('int')
test_out = test_out.astype('int')


def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

tr_d = (train_in, train_out) 
te_d = (test_in, test_out)

#tr_d = TupleUp(train_in, train_out)
#te_d = TupleUp(test_in, test_out)
training_inputs = [np.reshape(x, (256, 1)) for x in tr_d[0]]
training_results = [vectorized_result(y) for y in tr_d[1]]
training_data = zip(training_inputs, training_results)
#validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
#validation_data = zip(validation_inputs, va_d[1])
test_inputs = [np.reshape(x, (256, 1)) for x in te_d[0]]
test_data = zip(test_inputs, te_d[1])



