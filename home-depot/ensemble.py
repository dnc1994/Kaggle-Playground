import time
start_time = time.time()
import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor


class Ensemble(object):
    def __init__(self, n_folds, stacker, base_models):
        self.n_folds = n_folds
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        folds = list(KFold(y, self.n_folds))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))

        for i, clf in enumerate(self.base_models):

            print('Fitting For Base Model #%d / %d ---', i+1, len(self.base_models))

            S_test_i = np.zeros((T.shape[0], len(folds)))

            for j, (train_idx, test_idx) in enumerate(folds):

                print('--- Fitting For Fold %d / %d ---', j+1, self.n_folds)

                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                # y_holdout = y[test_idx]
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_holdout)[:]
                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict(T)[:]

                print('Elapsed: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

            S_test[:, i] = S_test_i.mean(1)

            print('Elapsed: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

        print('--- Base Models Trained: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

        clf = self.stacker
        clf.fit(S_train, y)
        clf.predict(S_train)

        print('--- Stacker Trained: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

        y_pred = clf.predict(S_test)[:, 1]

        return y_pred


def main(input='df_new_419_3.csv'):
    df_all = pd.read_csv(input, encoding='ISO-8859-1', index_col=0)
    num_train = 74067
    df_train = df_all.iloc[:num_train]
    df_test = df_all.iloc[num_train:]

    id_test = df_test['id']
    y_train = df_train['relevance'].values

    cols_to_drop = ['id', 'product_uid', 'relevance', 'search_term', 'product_title', 'product_description', 'brand', 'attr', 'product_info']
    for col in cols_to_drop:
        try:
            df_train.drop(col, axis=1, inplace=True)
            df_test.drop(col, axis=1, inplace=True)
        except:
            continue

    X_train = df_train[:]
    X_test = df_test[:]

    print('--- Features Set: %s minutes ---' % round(((time.time() - start_time) / 60), 2))
    print('Number of Features: ', len(X_train.columns.tolist()))

    base_models = [
        RandomForestRegressor(
            n_jobs=1, random_state=2016, verbose=1,
            n_estimators=500, max_features=10
        )
        # ,
        # ExtraTreesRegressor()
    ]
    ensemble = Ensemble(
        n_folds=5,
        stacker=LogisticRegression(C=0.1, penalty='l1'),
        base_models=base_models
    )

    y_pred = ensemble.fit_predict(X=X_train, y=y_train, T=X_test)
    for i in range(len(y_pred)):
        if y_pred[i] < 1.0:
            y_pred[i] = 1.0
        if y_pred[i] > 3.0:
            y_pred[i] = 3.0
    pd.DataFrame({'id': id_test, 'relevance': y_pred}).to_csv('submission_rfr.csv', index=False)

    print('--- Submission Generated: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

if __name__ == '__main__':
    main()
