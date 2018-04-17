import numpy as np
import pandas as pd
from sklearn import svm
from tqdm import tqdm

from utils import load_data, plot_layout


def main():
    scores = pd.DataFrame()
    X_train, X_test, y_train, y_test = load_data()

    with tqdm(total=19 * 21) as pbar:
        for g in np.logspace(-15, 3, 19, base=2):
            for c in np.logspace(-5, 15, 21, base=2):
                clf = svm.SVC(gamma=g, C=c)
                clf.fit(X_train, y_train)
                scores = scores.append(
                    {
                        'gamma': g,
                        'C': c,
                        'accuracy': clf.score(X_test, y_test)
                    },
                    ignore_index=True)
                pbar.update(1)

    plot_layout(scores.gamma, scores.C, scores.accuracy, 'gridsearch.png')
    print(scores.sort_values('accuracy', ascending=False).head())


if __name__ == '__main__':
    main()
