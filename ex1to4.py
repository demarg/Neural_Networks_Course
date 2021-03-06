# -*- coding: utf-8 -*-

# Runs using Python 3.5 and 2.7


from __future__ import division
import os

# Set working directory to location of this file
try:
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
except:
    pass


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



for dist_metric in ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']:
    ex1and2(dist_metric)



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


classifiers = [BayesClassifier(rows) for rows in range(1, 17)]

train_acc = [cls.report_accuracy("training", train_5, train_7) for cls in classifiers]
test_acc = [cls.report_accuracy("test", test_5, test_7) for cls in classifiers]

print("Rows:")
print([cls.rows for cls in classifiers])
print("Training set accuracy:")
print(train_acc)
print("Test set accuracy:")
print(test_acc)



##### Exercise 4

print("\nExercise 4\n")

# input    x: n   x 257
# weights  w: 257 x 10
# outputs  y: n   x 10

# y = x*w
# (n x 257) * (257 x 10) = n x 10
class PerceptronClassifier:
    def __init__(self):
        # initialize weights randomly
        self.w = np.random.rand(257, 10)


    def train(self, X, y, learning_rate = 0.03, max_iter = -1):
        """
        Train the network by Generalized Perceptron Algorithm (Duda et al.)
        Return True if 100% accuracy on training set reached before max_iter iterations.
        """
        y = y.astype('int')

        X = self.prepend_ones(X)
        Y = self.code_y(y)

        iteration = 0
        while iteration != max_iter:
            net = self.net(X)
            act = self.activation(net)

            digits = np.array(range(10))
            try:
                # first sample where the correct node is not the most activated
                i, a = next((i, a) for (i, a) in enumerate(act) if (a[digits != y[i]] >= a[y[i]]).any())
            except StopIteration:
                # if there is none, we're done
                return True

            correct = y[i]

            # update weights
            factors = np.zeros(10)
            factors = np.where(a >= a[correct], -1, 0) # reduce weights of nodes that were too active
            factors[correct] = 1                       # increase weights for the node that should have won
            self.w += learning_rate * np.transpose(np.array([X[i] * f for f in factors]))

            iteration += 1

        return False


    def prepend_ones(self, X):
        n = X.shape[0]
        ones = np.array([1] * n).reshape(-1, 1)
        return np.hstack([ones, X])

    def code_y(self, y):
        Y = np.zeros((len(y), 10))
        for i in range(len(y)):
            Y[i, int(y[i])] = 1
        return Y

    def net(self, X):
        return np.matmul(X, self.w)

    def activation(self, net):
        """ sigmoid function """
        return 1 / (1 + np.exp(-net))

    def classify(self, X):
        X = self.prepend_ones(X)

        net = self.net(X)
        act = self.activation(net)
        #print(act.shape)
        y = np.argmax(act, 1)
        return y


print("df = data.frame()")
for rate in [1., 0.5, 0.1, 0.05, 0.01]:

    cls = PerceptronClassifier()
    acc = []
    train_pred = cls.classify(train_in)
    acc.append(np.sum(train_pred == train_out) / len(train_out))
    for i in range(30):
        converged = cls.train(train_in, train_out, learning_rate = rate, max_iter = 100)
        train_pred = cls.classify(train_in)
        acc.append(np.sum(train_pred == train_out) / len(train_out))
        if converged:
            break

    print("df.add = data.frame(acc = c({}))".format(", ".join(str(a) for a in acc)))

    test_pred = cls.classify(test_in)
    print("df.add$rate = {}".format(rate))
    print("df.add$test_acc = {}".format(np.sum(test_pred == test_out) / len(test_out)))
    print("df.add$iter = seq(0, by = 100, length.out = nrow(df.add))")
    print("df = rbind(df, df.add)")
