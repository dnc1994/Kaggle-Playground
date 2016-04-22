import time
start_time = time.time()
import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn import grid_search
from sklearn.metrics import mean_squared_error, make_scorer
from xgboost import XGBRegressor


def mean_squared_error_(ground_truth, predictions):
    return mean_squared_error(ground_truth, predictions) ** 0.5

RMSE = make_scorer(mean_squared_error_, greater_is_better=False)


class Ensemble(object):
    def __init__(self, n_folds, stacker, base_models):
        self.n_folds = n_folds
        self.stacker = stacker
        self.base_models = base_models

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        folds = list(KFold(len(y), n_folds=self.n_folds, shuffle=True, random_state=2016))
        S_train = np.zeros((X.shape[0], len(self.base_models)))

        for i, clf in enumerate(self.base_models):

            print('Fitting For Base Model #%d / %d ---', i+1, len(self.base_models))
            for j, (train_idx, test_idx) in enumerate(folds):

                print('--- Fitting For Fold %d / %d ---', j+1, self.n_folds)

                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                # y_holdout = y[test_idx]
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_holdout)[:]
                S_train[test_idx, i] = y_pred

                print('Elapsed: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

            print('Elapsed: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

        print('--- Base Models Trained: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

        clf = self.stacker
        clf.fit(S_train, y)

        print('--- Stacker Trained: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

    def preidct(self, X):
        X = np.array(X)
        folds = list(KFold(len(X), n_folds=self.n_folds, shuffle=True, random_state=2016))
        S_test = np.zeros((X.shape[0], len(self.base_models)))

        for i, clf in enumerate(self.base_models):
            S_test_i = np.zeros((X.shape[0], len(folds)))
            for j, (train_idx, test_idx) in enumerate(folds):
                S_test_i[:, j] = clf.predict(X)[:]
            S_test[:, i] = S_test_i.mean(1)

        clf = self.stacker
        y_pred = clf.predict(S_test)[:]
        return y_pred

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        folds = list(KFold(len(y), n_folds=self.n_folds, shuffle=True, random_state=2016))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))

        for i, clf in enumerate(self.base_models):

            print('Fitting For Base Model #{0} / {1} ---'.format(i+1, len(self.base_models)))

            S_test_i = np.zeros((T.shape[0], len(folds)))

            for j, (train_idx, test_idx) in enumerate(folds):

                print('--- Fitting For Fold #{0} / {1} ---'.format(j+1, self.n_folds))

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

        # param_grid = {
        #     'n_estimators': [100],
        #     'learning_rate': [0.45, 0.05, 0.055],
        #     'subsample': [0.72, 0.75, 0.78]
        # }
        param_grid = {
            'n_estimators': [100],
            'learning_rate': [0.05],
            'subsample': [0.75]
        }
        grid = grid_search.GridSearchCV(estimator=self.stacker, param_grid=param_grid, n_jobs=1, cv=5, verbose=20, scoring=RMSE)
        grid.fit(S_train, y)

        # a little memo
        message = 'to determine local CV score of #28'

        try:
            print('Param grid:')
            print(param_grid)
            print('Best Params:')
            print(grid.best_params_)
            print('Best CV Score:')
            print(-grid.best_score_)
            print('Best estimator:')
            print(grid.best_estimator_)
            print(message)
        except:
            pass

        print('--- Stacker Trained: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

        y_pred = grid.predict(S_test)[:]

        return y_pred


def main(input='df_new_422.csv'):
    df_all = pd.read_csv(input, encoding='ISO-8859-1', index_col=0)
    num_train = 74067
    df_train = df_all.iloc[:num_train]
    df_test = df_all.iloc[num_train:]

    id_test = df_test['id']
    y_train = df_train['relevance'].values

    cols_to_drop = ['id', 'relevance']
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
            n_estimators=500, max_features=12
        ),
        ExtraTreesRegressor(
            n_jobs=1, random_state=2016, verbose=1,
            n_estimators=500, max_features=12
        ),
        GradientBoostingRegressor(
            random_state=2016, verbose=1,
            n_estimators=500, max_features=12, max_depth=8,
            learning_rate=0.05, subsample=0.8
        ),
        XGBRegressor(
            seed=2016,
            n_estimators=200, max_depth=8,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.85
        )
    ]
    ensemble = Ensemble(
        n_folds=5,
        stacker=GradientBoostingRegressor(
            random_state=2016, verbose=1
        ),
        base_models=base_models
    )

    y_pred = ensemble.fit_predict(X=X_train, y=y_train, T=X_test)
    for i in range(len(y_pred)):
        if y_pred[i] < 1.0:
            y_pred[i] = 1.0
        if y_pred[i] > 3.0:
            y_pred[i] = 3.0
    pd.DataFrame({'id': id_test, 'relevance': y_pred}).to_csv('submission_ensemble.csv', index=False)

    print('--- Submission Generated: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

if __name__ == '__main__':
    main()
