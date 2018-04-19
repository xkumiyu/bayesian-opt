import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

from skopt import gp_minimize


def f(x):
    if isinstance(x, list):
        x = x[0]
    return x * np.sin(x)


def plot(X, y, i):
    gp = GaussianProcessRegressor()
    gp.fit(X, y)

    x = np.atleast_2d(np.linspace(0, 10, 1000)).T
    y_pred, sigma = gp.predict(x, return_std=True)

    fig, ax = plt.subplots()
    ax.plot(x, f(x), 'r:', label='$f(x) = x\,\sin(x)$')
    ax.plot(X, y, 'r.', markersize=10, label='Observations')
    ax.plot(x, y_pred, 'b-', label='Prediction')
    ax.fill(
        np.concatenate([x, x[::-1]]),
        np.concatenate([y_pred - 1.96 * sigma, (y_pred + 1.96 * sigma)[::-1]]),
        alpha=.5,
        fc='b',
        ec='None',
        label='95% confidence interval')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$f(x)$')
    ax.set_ylim(-10, 20)
    ax.legend(loc='upper left')
    fig.savefig(f'{i}.png')


def plot_true():
    x = np.atleast_2d(np.linspace(0, 10, 1000)).T

    fig, ax = plt.subplots()
    ax.plot(x, f(x), 'r:', label='$f(x) = x\,\sin(x)$')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$f(x)$')
    ax.set_ylim(-10, 20)
    ax.legend(loc='upper left')
    fig.savefig('true.png')


def main():
    plot_true()

    res = gp_minimize(
        f, [(0, 10)], acq_func="gp_hedge", n_calls=10, random_state=2)

    X = np.empty(0)
    for i, x in enumerate(res.x_iters):
        X = np.append(X, x[0])
        X = np.atleast_2d([X]).T
        y = f(X).ravel()
        plot(X, y, i)


if __name__ == '__main__':
    main()
