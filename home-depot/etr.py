# -*- encoding:ISO-8859-1 -*-
import warnings
warnings.filterwarnings('ignore')
import time
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn import grid_search
import random
random.seed(2016)


def mean_squared_error_(ground_truth, predictions):
    return mean_squared_error(ground_truth, predictions) ** 0.5

RMSE = make_scorer(mean_squared_error_, greater_is_better=False)


def main(input='df_new_419_3.csv'):
    start_time = time.time()

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
    # print(X_train.columns.tolist())

    # exit(0)

    etr = ExtraTreesRegressor(n_jobs=1, random_state=2016, verbose=1)

    param_grid = {'n_estimators': [500], 'max_features': [10]}
    model = grid_search.GridSearchCV(estimator=etr, param_grid=param_grid, n_jobs=5, cv=10, verbose=20, scoring=RMSE)
    model.fit(X_train, y_train)

    print('--- Grid Search Completed: %s minutes ---' % round(((time.time() - start_time) / 60), 2))
    print('Best Params:')
    print(model.best_params_)
    print('Best CV Score:')
    print(-model.best_score_)

    y_pred = model.predict(X_test)
    for i in range(len(y_pred)):
        if y_pred[i] < 1.0:
            y_pred[i] = 1.0
        if y_pred[i] > 3.0:
            y_pred[i] = 3.0
    pd.DataFrame({'id': id_test, 'relevance': y_pred}).to_csv('submission_etr.csv', index=False)

    print('--- Submission Generated: %s minutes ---' % round(((time.time() - start_time) / 60), 2))


if __name__ == '__main__':
    main()
