import numpy as np
from scipy import stats
from sklearn.cluster import KMeans

from typing import List


class Theta:
    """
    Parameters used in EM algorithm.
    """

    def __init__(self):
        self.prior = 0  # type: float
        self.mean = None  # type: np.array
        self.variance = 0  # type: float


# FIXME: Sometimes certain clusters get no data assigned to them.
# This causes the program to crash, when computing the variance.
# If a cluster gets only 1 data point, then the variance is 0, which leads to a crash when computing normal.pdf(...)

# FIXME: There is no measure of "convergence". Thus, the algorithm always runs the exact number of iterations specified.

class EM:
    """
    EM algorithm for Rendering model, with Gaussian distributions
    """

    def __init__(self, data, num_classes, num_nuisances):
        self.data = np.copy(data)
        self.num_classes = num_classes
        self.num_nuisances = num_nuisances
        self.num_clusters = self.num_classes * self.num_nuisances
        self._data_count = None
        self._data_mean = None
        self._data_var = None
        self._data_dim = None

    def fit(self, max_iter):
        split_data = self.initialise_clusters_with_kmeans()
        thetas = self.maximization(split_data)
        for i in range(max_iter):
            split_data = self.expectation(thetas)
            thetas = self.maximization(split_data)
        return split_data, thetas

    def fit_and_plot(self, max_iter):
        """
        Plot clusters at each step. Works only for 2D data.
        """
        from matplotlib import pyplot as plt
        from matplotlib import cm

        colours = cm.rainbow(np.linspace(0, 1, self.num_classes))  # FIXME: rainbow list -> array

        def plot_data(d):
            for c in range(self.num_classes):
                for n in range(self.num_nuisances):
                    plt.scatter(*d[c][n].T, c=colours[c])
            plt.waitforbuttonpress()

        def plot_mean(th):
            for c in range(self.num_classes):
                for n in range(self.num_nuisances):
                    plt.scatter(*th[c][n].mean.T, c=colours[c], marker="x")
            plt.waitforbuttonpress()

        plt.ion()
        plt.scatter(*self.data.T)
        plt.waitforbuttonpress()

        split_data = self.initialise_clusters_with_kmeans()
        plot_data(split_data)
        thetas = self.maximization(split_data)
        plot_mean(thetas)

        for i in range(max_iter):
            plt.clf()
            split_data = self.expectation(thetas)
            plot_data(split_data)
            thetas = self.maximization(split_data)
            plot_mean(thetas)
        return split_data, thetas

    def initialise_clusters_with_kmeans(self):
        kmeans = KMeans(self.num_clusters).fit(self.data)
        clusters = [[None for _ in range(self.num_nuisances)] for _ in range(self.num_classes)]
        i = 0
        for c in range(self.num_classes):
            for n in range(self.num_nuisances):
                data_in_cluster = self.data[kmeans.labels_ == i, :]
                i += 1
                clusters[c][n] = data_in_cluster
        return clusters

    def expectation(self, thetas: List[List[Theta]]):
        """
        E-step: Assign each datum to most likely cluster
        """
        split_data = [[np.empty((self.data_count, self.data_dim)) for _ in range(self.num_nuisances)]
                      for _ in range(self.num_classes)]
        per_split_count = [[0 for _ in range(self.num_nuisances)] for _ in range(self.num_classes)]

        for datum in self.data:
            # Use log probabilities to avoid numerical problems
            log_posteriors = np.empty((self.num_classes, self.num_nuisances))
            for c in range(self.num_classes):
                for n in range(self.num_nuisances):
                    log_likelihood = stats.multivariate_normal.logpdf(datum, thetas[c][n].mean, thetas[c][n].variance)
                    log_posteriors[c, n] = np.log(thetas[c][n].prior) + log_likelihood
            map_class, map_nuisance = np.unravel_index(np.argmax(log_posteriors), log_posteriors.shape)
            split_data[map_class][map_nuisance][per_split_count[map_class][map_nuisance]] = datum
            per_split_count[map_class][map_nuisance] += 1

        reduced_split_data = [[split_data[c][n][:per_split_count[c][n]] for n in range(self.num_nuisances)]
                              for c in range(self.num_classes)]
        return reduced_split_data

    def maximization(self, split_data):
        """
        M-step: Compute most probable parameters of each cluster
        """
        thetas = [[Theta() for _ in range(self.num_nuisances)] for _ in range(self.num_classes)]
        cluster_sizes = [[split_data[c][n].shape[0] for n in range(self.num_nuisances)]
                         for c in range(self.num_classes)]
        for c, class_thetas in enumerate(thetas):
            for n, theta in enumerate(class_thetas):
                theta.prior = cluster_sizes[c][n] / self.data_count
                theta.mean = self.calc_mean(split_data[c][n])
                theta.variance = self.calc_var(split_data[c][n], theta.mean, cluster_sizes[c][n])
        return thetas

    @property
    def data_count(self):
        if self._data_count is None:
            self._data_count = self.data.shape[0]
        return self._data_count

    @property
    def data_dim(self):
        if self._data_dim is None:
            self._data_dim = self.data.shape[1]
        return self._data_dim

    @property
    def data_mean(self):
        if self._data_mean is None:
            self._data_mean = self.calc_mean(self.data)
        return self._data_mean

    @property
    def data_var(self):
        if self._data_var is None:
            self._data_var = self.calc_var(self.data, self.data_mean, self.data_count)
        return self._data_var

    @classmethod
    def calc_mean(cls, data):
        return np.mean(data, axis=0)

    @classmethod
    def calc_var(cls, data, mean, num):
        return np.sum(np.linalg.norm(data - mean, axis=1) ** 2) / num
