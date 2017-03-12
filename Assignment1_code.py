# -*- coding: utf-8 -*-

from __future__ import division

# Set working directory to location of this file
try:
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
except:
    pass


import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn import metrics
from plot_confusion import plot_confusion_matrix

rcParams.update({'figure.autolayout': True})


# Our data
test_in = np.loadtxt("data/test_in.csv", delimiter = ',')
test_out = np.loadtxt("data/tets_out.csv", delimiter = ',')
train_in = np.loadtxt("data/train_in.csv", delimiter = ',')
train_out = np.loadtxt("data/train_out.csv", delimiter = ',')

print("\nExercise 1 & 2\n")

for d in range(10):
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

    for d in range(10):
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
    ax.xaxis.set_ticks(range(10))
    ax.yaxis.set_ticks(range(10))
    ax.set_xticklabels(range(10))
    ax.set_yticklabels(range(10))
    plt.savefig("out/1_{}_digit-dists.png".format(distance_metric))
    plt.close()


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

        cnf_matrix = metrics.confusion_matrix(set_out, set_pred, labels = range(10))

        fig = plt.figure()
        plot_confusion_matrix(cnf_matrix,
                              classes = range(10),
                              title='Confusion matrix of {} set'.format(set_name),
                              normalize = True)

        plt.savefig("out/2_{}_{}_confusion_matrix.png".format(distance_metric, set_name))
        plt.close()



# TODO uncomment
#for dist_metric in ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']:
#    ex1and2(dist_metric)



#####  Exercise 3

print("\nExercise 3\n")

train_5 = train_in[train_out == 5]
train_7 = train_in[train_out == 7]
test_5 = test_in[test_out == 5]
test_7 = test_in[test_out == 7]

prior_5 = len(train_5) / (len(train_5) + len(train_7))
prior_7 = len(train_7) / (len(train_5) + len(train_7))


class BayesClassifier:
    def __init__(self, rows):
        self.rows = rows

        lowsum_5 = self.lowsum(train_5) # lowest 3 rows in image
        lowsum_7 = self.lowsum(train_7)

        min_val = min(np.concatenate([lowsum_5, lowsum_7]))
        max_val = max(np.concatenate([lowsum_5, lowsum_7]))

        bins = np.linspace(min_val, max_val, num = 10, endpoint = False) # 10 bins, array of the left boundaries
        self.bin_limits = bins[1:] # get the limits between bins for use with np.digitize

        binned_5 = self.bin_features(lowsum_5)
        binned_7 = self.bin_features(lowsum_7)

        binprob_5 = np.array([sum(binned_5 == x) for x in range(10)]) / len(train_5)
        binprob_7 = np.array([sum(binned_7 == x) for x in range(10)]) / len(train_7)

        self.post_5 = prior_5 * binprob_5
        self.post_7 = prior_7 * binprob_7

        # histogram of likelihoods
        fig, ax = plt.subplots()
        width = bins[1] - bins[0]
        rects1 = ax.bar(bins, binprob_5, width = width, color = (0, 0, 1, 0.5))
        rects2 = ax.bar(bins, binprob_7, width = width, color = (0, 1, 0, 0.5))
        ax.legend((rects1[0], rects2[0]), ('5', '7'))
        plt.savefig("out/3_likelihoods_{}.png".format(rows))
        plt.close()

        # histogram of posteriors
        fig, ax = plt.subplots()
        width = bins[1] - bins[0]
        rects1 = ax.bar(bins, self.post_5, width = width, color = (0, 0, 1, 0.5))
        rects2 = ax.bar(bins, self.post_7, width = width, color = (0, 1, 0, 0.5))
        ax.legend((rects1[0], rects2[0]), ('5', '7'))
        plt.savefig("out/3_posteriors_{}.png".format(rows))
        plt.close()


    # Feature: take sum of lower n rows of matrix
    def lowsum(self, X):
        """
        Sum up the last n rows of each digit image
        """
        index = range(256 - 16 * self.rows, 256)
        return np.array([sum(point[index]) for point in X])

    def bin_features(self, features):
        return np.digitize(features, self.bin_limits)

    def classify(self, X):
        ls = self.lowsum(X)
        binned = self.bin_features(ls)
        return np.where(self.post_5[binned] > self.post_7[binned], 5, 7)


    def report_accuracy(self, set_name, in_5, in_7):
        correct_5 = np.sum(self.classify(in_5) == 5)
        correct_7 = np.sum(self.classify(in_7) == 7)
        accuracy = (correct_5 + correct_7) / (len(in_5) + len(in_7))
        print("Accuracy in {} set with {} rows: {:.2f}".format(set_name, self.rows, accuracy))
        return accuracy


# TODO uncomment
# classifiers = [BayesClassifier(rows) for rows in range(1, 17)]

# train_acc = [cls.report_accuracy("training", train_5, train_7) for cls in classifiers]
# test_acc = [cls.report_accuracy("test", test_5, test_7) for cls in classifiers]

# print("Rows:")
# print([cls.rows for cls in classifiers])
# print("Training set accuracy:")
# print(train_acc)
# print("Test set accuracy:")
# print(test_acc)



##### Exercise 4

print("\nExercise 4\n")

# input    x: n   x 257
# weights  w: 257 x 10
# outputs  y: n   x 10

# y = x*w
# nx257 * 257x10 = nx10
class PerceptronClassifier:
    def __init__(self, X, y, learning_rate = 0.03):
        # initialize weights
        self.w = np.zeros((257, 10))

        X = self.prepend_ones(X)

        # prepare Y
        Y = np.zeros((len(y), 10))
        for i in range(len(y)):
            Y[i, int(y[i])] = 1

        # train the network
        net = self.net(X)
        delta_w = (Y - self.activation(net)) * self.activation_drv(net)
        print(delta_w.shape)
        delta_w = np.matmul(delta_w, X)
        delta_w = learning_rate * delta_w
        print(delta_w)

    def prepend_ones(self, X):
        n = X.shape[0]
        ones = np.array([1] * n).reshape(-1, 1)
        return np.hstack([ones, X])

    def net(self, X):
        return np.matmul(X, self.w)

    def activation(self, net):
        """ sigmoid function """
        return 1 / (1 + np.exp(-net))

    def activation_drv(self, net):
        """ derivative of the sigmoid function """
        act = self.activation(net)
        return act * (1 - act)

    def classify(self, X):
        X = self.prepend_ones(X)

        net = self.net(X)
        act = self.activation(net)
        y = np.argmax(act, 1)
        print(y)

cls = PerceptronClassifier(train_in, train_out)
print(cls.classify(test_in))
