import numpy as np

from em import EM
import mnist


def generate_gaussian_data(data_per_cluster, num_clusters):
    data = np.empty((data_per_cluster * num_clusters, 2))
    for i in range(num_clusters):
        mean = np.random.normal(0, 5, 2)
        var = abs(np.random.normal(2, 0.5))
        data[i * data_per_cluster:(i + 1) * data_per_cluster] = np.random.normal(mean, var, [data_per_cluster, 2])
    np.random.shuffle(data)
    return data


def run_gaussian():
    data_per_cluster = 15
    num_clusters = 2
    data = generate_gaussian_data(data_per_cluster, num_clusters)

    num_classes = num_clusters
    num_nuisances = 1

    solver = EM(data, num_classes, num_nuisances)
    split_data, thetas = solver.fit_and_plot(5)
    for class_thetas in thetas:
        for theta in class_thetas:
            print(f"Mean: {theta.mean} - Var: {theta.variance}")


def run_mnist():
    #  FIXME: running EM on MNIST has the problem that all data collapses to one class
    # This is because the likelihood for that class is slightly higher than all other.
    # Probably has to do with the variance being lower for one, form k-means,
    # and that being more important than closeness to mean for such high dimensional data?
    # Running it with 0 iterations (i.e. on k-means) work fine, then it finds different orientations of the digits.
    data_per_class = 20

    training_data = list(mnist.read("training"))
    dim_x, dim_y = np.shape(training_data[0][1])
    ones = [d[1] for d in training_data if d[0] == 1]
    fours = [d[1] for d in training_data if d[0] == 4]
    fives = [d[1] for d in training_data if d[0] == 5]

    ones = ones[:data_per_class]
    fours = fours[:data_per_class]
    fives = fives[:data_per_class]

    data = np.array(ones + fours + fives).reshape((-1, dim_x * dim_y))
    solver = EM(data=data, num_classes=3, num_nuisances=3)
    split_data, thetas = solver.fit(max_iter=1)

    for c, class_thetas in enumerate(thetas):
        for n, theta in enumerate(class_thetas):
            print(f"Prior: {theta.prior}, Var: {theta.variance}")
            mnist.show(thetas[c][n].mean.reshape(28, 28))


if __name__ == "__main__":
    run_gaussian()
