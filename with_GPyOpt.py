from sklearn import svm

from GPyOpt.methods import BayesianOptimization
from utils import load_data


def main():
    X_train, X_test, y_train, y_test = load_data()

    def f(x):
        res = []
        for i in x:
            clf = svm.SVC(gamma=i[0], C=i[1])
            clf.fit(X_train, y_train)
            res.append(-1 * clf.score(X_test, y_test))
        return res

    domain = [{
        'name': 'C',
        'type': 'continuous',
        'domain': (2**-15, 2**3)
    }, {
        'name': 'gamma',
        'type': 'continuous',
        'domain': (2**-5, 2**15)
    }]

    opt = BayesianOptimization(f=f, domain=domain, acquisition_type='LCB')
    opt.run_optimization(max_iter=20)

    print(opt.x_opt)
    print(opt.fx_opt)


if __name__ == '__main__':
    main()
