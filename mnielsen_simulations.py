# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 18:38:59 2017

Simulations

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

import mnielsen_network 
import own_wrapper

test = TupleUp(test_in, test_out)
train = TupleUp(train_in, train_out)

net = Network([256, 30, 10])

net.SGD(train, 30, 10, 3.0, test_data=test)
