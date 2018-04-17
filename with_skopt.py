import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

from skopt import gp_minimize
from skopt.plots import plot_convergence
from utils import load_data, plot_layout


def main():
    X_train, X_test, y_train, y_test = load_data()

    def f(x):
        clf = svm.SVC(gamma=x[0], C=x[1])
        clf.fit(X_train, y_train)
        return -1 * clf.score(X_test, y_test)

    spaces = [
        (2**-15, 2**3, 'log-uniform'),
        (2**-5, 2**15, 'log-uniform'),
        # ['linear', 'poly', 'rbf']
    ]
    res = gp_minimize(
        f, spaces, acq_func="gp_hedge", n_calls=20, random_state=0)
    print(f'res.x: {res.x}')
    print(f'res.fun: {res.fun}')

    plot_layout(
        np.array(res.x_iters).T[0],
        np.array(res.x_iters).T[1], -1 * res.func_vals, 'skopt.png')

    fig, ax = plt.subplots()
    plot_convergence(res, ax=ax)
    fig.savefig('convergence.png')


if __name__ == '__main__':
    main()
