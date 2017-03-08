# -*- coding: utf-8 -*-

##### Exercise 1

import os
import numpy as np
import matplotlib.pyplot as plt

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
    r_c = np.amax(abs(train_d - c_d), axis = 0)
    centers[d] = c_d
    radiuses[d] = r_c

dists = np.array([
  [abs(ci - cj) for ci in centers] for cj in centers
])

mean_dists = np.mean(dists, axis = 2)


fig = plt.figure()
ax = fig.add_subplot(111)
dist_mat = ax.matshow(mean_dists)
fig.colorbar(dist_mat)
ax.xaxis.set_ticks(range(0, 10))
ax.yaxis.set_ticks(range(0, 10))
ax.set_xticklabels(range(0, 10))
ax.set_yticklabels(range(0, 10))
#plt.show()


##### Exercise 2
def classify(point):
    return np.argmin(
        [np.mean(abs(point - c)) for c in centers],
        0
    )

train_pred = [classify(point) for point in train_in]
