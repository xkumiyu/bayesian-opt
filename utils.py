import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split


def load_data():
    digits = datasets.load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test


def plot_layout(gamma, C, accuracy, outfile):
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('gamma')
    ax.set_ylabel('C')
    ax = ax.scatter(gamma, C, c=accuracy, cmap='Blues')
    fig.colorbar(ax)
    fig.savefig(outfile)
