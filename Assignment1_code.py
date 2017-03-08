# -*- coding: utf-8 -*-

##### Exercise 1

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn import metrics
from plot_confusion import plot_confusion_matrix

rcParams.update({'figure.autolayout': True})


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

def dist(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


for d in range(0, 10):
    train_d = train_in[train_out == d, ]
    c_d = np.mean(train_d, 0)
    r_c = np.amax(dist(train_d, c_d), axis = 0)
    centers[d] = c_d
    radiuses[d] = r_c
    print(str(d) + "s in training set: " + str(len(train_d)))


dists = np.array([
    [dist(ci, cj) for ci in centers] for cj in centers
])

fig = plt.figure()
ax = fig.add_subplot(111)
dist_mat = ax.matshow(dists)
fig.colorbar(dist_mat)
ax.xaxis.set_ticks(range(0, 10))
ax.yaxis.set_ticks(range(0, 10))
ax.set_xticklabels(range(0, 10))
ax.set_yticklabels(range(0, 10))
plt.savefig("out/digit-dists.png")


##### Exercise 2
def classify(point):
    return np.argmin(
        [np.mean(dist(point, c)) for c in centers],
        0
    )


for set_name, set_in, set_out in [("training", train_in, train_out), ("test", test_in, test_out)]:
    set_pred = [classify(point) for point in set_in]
    correct = set_pred == set_out
    print("Correctly classified in " + set_name + " set: " + str(np.sum(correct) / len(correct)))

    cnf_matrix = metrics.confusion_matrix(set_out, set_pred, labels = range(0, 10))

    fig = plt.figure()
    plot_confusion_matrix(cnf_matrix,
                          classes = range(0, 10),
                          title='Confusion matrix of ' + set_name + ' set',
                          normalize = True)

    plt.savefig("out/" + set_name + "_confusion_matrix.png")
