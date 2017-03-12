# -*- coding: utf-8 -*-

from __future__ import division
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

for d in range(0, 10):
    print("{digit}s in training set: {count}".format(digit = d, count = np.sum(train_out == d)))


def ex1and2(distance_metric):
    ##### Exercise 1

    print("Using {} distance".format(distance_metric))

    def dist(a, b):
        adim = a.ndim
        if adim == 1:
            a = np.array([a])

        dist_mat = metrics.pairwise.pairwise_distances(
            a, np.array([b]),
            metric = distance_metric)

        if adim == 2:
            return dist_mat[:, 0]
        else:
            return dist_mat[0, 0]


    centers = np.array([None] * 10)
    radiuses = np.array([None] * 10)

    for d in range(0, 10):
        train_d = train_in[train_out == d, ]
        c_d = np.mean(train_d, 0)
        r_c = np.amax(dist(train_d, c_d), axis = 0)
        centers[d] = c_d
        radiuses[d] = r_c


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
    plt.savefig("out/{}_digit-dists.png".format(distance_metric))


    ##### Exercise 2
    def classify(point):
        return np.argmin(
            [np.mean(dist(point, c)) for c in centers],
            0
        )


    for set_name, set_in, set_out in [("training", train_in, train_out), ("test", test_in, test_out)]:
        set_pred = [classify(point) for point in set_in]
        correct = set_pred == set_out
        print("Correctly classified in {} set: {}".format(set_name, np.sum(correct) / len(correct)))

        cnf_matrix = metrics.confusion_matrix(set_out, set_pred, labels = range(0, 10))

        fig = plt.figure()
        plot_confusion_matrix(cnf_matrix,
                              classes = range(0, 10),
                              title='Confusion matrix of {} set'.format(set_name),
                              normalize = True)

        plt.savefig("out/{}_{}_confusion_matrix.png".format(distance_metric, set_name))



# TODO uncomment
#for dist_metric in ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']:
#    ex1and2(dist_metric)



#####  Exercise 3

train_5 = train_in[train_out == 5]
train_7 = train_in[train_out == 7]

# feature: take lower 1/x ratio of matrix and sum

prior_5 = len(train_5) / (len(train_5) + len(train_7))
prior_7 = len(train_7) / (len(train_5) + len(train_7))

def lowsum(X, n):
    """
    Sum up the last n rows of each digit image
    """
    index = range(256 - 16 * n, 256)
    return np.array([sum(point[index]) for point in X])

lowsum_5 = lowsum(train_5, 3) # lowest 3 rows in image
lowsum_7 = lowsum(train_7, 3)

min_val = min(np.concatenate([lowsum_5, lowsum_7]))
max_val = max(np.concatenate([lowsum_5, lowsum_7]))

bins = np.linspace(min_val, max_val, num = 10, endpoint = False)[1:] #10 boxes
def bin_features(features):
    np.digitize(features, bins)

binned_5 = bin_features(lowsum_5)
binned_7 = bin_features(lowsum_7)

binprob_5 = np.array([sum(binned_5 == x) for x in range(0,10)]) / len(train_5)
binprob_7 = np.array([sum(binned_7 == x) for x in range(0,10)]) / len(train_7)

post_5 = prior_5 * binprob_5
post_7 = prior_7 * binprob_7

def classify(data):
    ls = lowsum(data, 3)
    binned = bin_features(ls)
    return np.where(post_5[binned] > post_7[binned], 5, 7)



##### Exercise 4
