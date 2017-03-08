# -*- coding: utf-8 -*-
"""
Created on Wed Mar 08 09:42:48 2017

@author: Anonymous & Anonymous
"""

import os
import numpy as np

# Set working directory to location of this file
try:
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
except:
    pass

# Our data
test_in = np.loadtxt("data/test_in.csv", delimiter = ',')
test_out = np.loadtxt("data/tets_out.csv", delimiter = ',')
train_in = np.loadtxt("data/train_in.csv", delimiter = ',')
train_out = np.loadtxt("data/train_out.csv", delimiter = ',')

# Proceed with train_in and train_out, use test_in and test_out to check prediction later

centers = np.array([None] * 10)
radiuses = np.array([None] * 10)

for d in range(0, 10):
    train_d = train_in[train_out == d, ]
    c_d = np.mean(train_d, 0)

    diff = [
        [px - ]
    ]
    r_d = [img - c_d for img in train_d]

    centers[d] = c_d
    radiuses[d] = r_d
