"""Generate synthetic data from a Gaussian Mixture Model (GMM)

References:
  http://www.nehalemlabs.net/prototype/blog/2014/04/03/quick-introduction-to-gaussian-mixture-models-with-python/

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from mpl_toolkits.mplot3d import Axes3D
from sklearn import mixture


def gaussians(x, y):
    g1 = mlab.bivariate_normal(x, y, 1.0, 1.0, -1, -1, -0.8)
    g2 = mlab.bivariate_normal(x, y, 1.5, 0.8, 1, 2, 0.6)
    return 0.6 * g1 + 28.4 * g2 / (0.6 + 28.4)


def plot_gaussians():
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = np.arange(-5, 5, 0.1)
    Y = np.arange(-5, 5, 0.1)
    X, Y = np.meshgrid(X, Y)
    Z = gaussians(X, Y)
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('coolwarm'),
                           linewidth=0, antialiased=True)
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.savefig('3dgauss.png')
    plt.clf()


def sample_prgs(N=10000, s=10, prgs=[np.random.normal] * 2, show=True, save='prg_samples.png', verbosity=1):
    '''Metropolis Hastings

    Inputs:
      N (int): Number of random samples to generate
      s (int):
      prgs (list of func): list of Pseudo Random Number Generators
        default: np.random.normal
    '''
    r = np.zeros(2)
    p = q(r[0], r[1])
    if verbosity > 0:
        print("True mean: {}".format(p))
    samples = []
    for i in xrange(N):
        rn = r + np.array([prg(size=1) for prg in prgs]).T[0]
        pn = q(rn[0], rn[1])
        if pn >= p:
            p = pn
            r = rn
        else:
            u = np.random.rand()
            if u < pn / p:
                p = pn
                r = rn
        if i % s == 0:
            samples.append(r)

    samples = np.array(samples)
    if show or save:
        plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=1)

    '''Plot target'''
    dx = 0.01
    x = np.arange(np.min(samples), np.max(samples), dx)
    y = np.arange(np.min(samples), np.max(samples), dx)
    X, Y = np.meshgrid(x, y)
    Z = q(X, Y)
    CS = plt.contour(X, Y, Z, 10, alpha=0.5)
    plt.clabel(CS, inline=1, fontsize=10)
    if save:
        plt.savefig(save)
    return samples


def fit_gmm(samples, show=True, save='class_predictions.png', verbosity=1):
    gmm = mixture.GMM(n_components=2, covariance_type='full')
    gmm.fit(samples)
    if verbosity > 0:
        print gmm.means_
    colors = list('rgbyomck')
    if show:
        c = [colors[i % len(colors)] for i in gmm.predict(samples)]
        ax = plt.gca()
        ax.scatter(samples[:, 0], samples[:, 1], c=c, alpha=0.8)
        if save:
            plt.savefig(save)


if __name__ == '__main__':
    plot_q()
    samples = sample_prgs()
    fit_gmm(samples)
