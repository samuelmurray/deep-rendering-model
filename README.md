# Deep Rendering Model

An implementation of a (shallow) rendering model from the paper [A Probabilistic Theory of Deep Learning](https://arxiv.org/abs/1504.00641). It's probably not working very well, use at own risk.

## Description

The code was made to get a better understanding of the rendering model, and is not complete in terms of functionality or lack of bugs. The algorithm works for Gaussian data (2D for purpose of visualisation), either with or without (or rather 1) nuisance variables. However, when nuisances, it is more likely to crash, due to the EM algorithm not handling empty clusters.

When running on MNIST data (download from [here](http://yann.lecun.com/exdb/mnist/), unpack and put in same directory as code), the EM algorithm does not work, since all instances get assigned to one class. However, it does work when only running k-means (_i.e._ setting number of iterations to 0), with or without nuisance variables.

## Run

To run the code, simply call

```bash
> python3 main.py
```

Modify the `main.py` to change the number of clusters, use nuisances or run it on the MNIST data.