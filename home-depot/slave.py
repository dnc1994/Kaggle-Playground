# -*- encoding:ISO-8859-1 -*-
import warnings
warnings.filterwarnings("ignore")
import time
start_time = time.time()
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn import pipeline, grid_search
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, make_scorer
from nltk.stem.porter import *
stemmer = PorterStemmer()
import re
import random
random.seed(2016)

# 'electr','paint','pipe','light','kitchen','wood','outdoor','door','bathroom'
stop_w = ['for', 'xbi', 'and', 'in', 'th', 'on', 'sku',
          'with', 'what', 'from', 'that', 'less', 'er', 'ing']
strNum = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
          'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9}


def seg_words(str1, str2):
    str2 = str2.lower()
    str2 = re.sub("[^a-z0-9./]", " ", str2)
    str2 = [z for z in set(str2.split()) if len(z) > 2]
    words = str1.lower().split(" ")
    s = []
    for word in words:
        if len(word) > 3:
            s1 = []
            s1 += segmentit(word, str2, True)
            if len(s) > 1:
                s += [z for z in s1 if z not in ['er', 'ing', 's', 'less'] and len(z) > 1]
            else:
                s.append(word)
        else:
            s.append(word)
    return (" ".join(s))


def segmentit(s, txt_arr, t):
    st = s
    r = []
    for j in range(len(s)):
        for word in txt_arr:
            if word == s[:-j]:
                r.append(s[:-j])
                # print(s[:-j],s[len(s)-j:])
                s = s[len(s) - j:]
                r += segmentit(s, txt_arr, False)
    if t:
        i = len(("").join(r))
        if not i == len(st):
            r.append(st[i:])
    return r


def str_common_word(str1, str2):
    words, cnt = str1.split(), 0
    for word in words:
        if str2.find(word) >= 0:
            cnt += 1
    return cnt


def str_whole_word(str1, str2, i_):
    cnt = 0
    while i_ < len(str2):
        i_ = str2.find(str1, i_)
        if i_ == -1:
            return cnt
        else:
            cnt += 1
            i_ += len(str1)
    return cnt


def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions)**0.5
    return fmean_squared_error_

RMSE = make_scorer(fmean_squared_error, greater_is_better=False)


class cust_regression_vals(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, hd_searches):
        d_col_drops = ['id', 'product_uid', 'relevance',
                       'search_term', 'search_term_stemmed',
                       'product_title', 'product_title_stemmed',
                       'product_description', 'product_description_stemmed',
                       'brand', 'brand_stemmed'
                       ]
        hd_searches = hd_searches.drop(d_col_drops, axis=1).values
        return hd_searches


class cust_txt_col(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key].apply(str)


if __name__ == '__main__':
    num_train = 74067
    df_all = pd.read_csv('df_start_over.csv', encoding='ISO-8859-1', index_col=0)
    df_train = df_all.iloc[:num_train]
    df_test = df_all.iloc[num_train:]
    id_test = df_test['id']
    y_train = df_train['relevance'].values
    X_train = df_train[:]
    X_test = df_test[:]
    print("--- Features Set: %s minutes ---" % round(((time.time() - start_time) / 60), 2))

    rfr = RandomForestRegressor(n_jobs=1, random_state=2016, verbose=1)
    tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
    tsvd = TruncatedSVD(n_components=10, random_state=2016)

    import xgboost
    xgb = xgboost.XGBClassifier(objective='reg:linear',
                                seed=2016)

    clf = pipeline.Pipeline([
        ('union', FeatureUnion(
            transformer_list=[
                            ('cst', cust_regression_vals()),
                            ('txt1', pipeline.Pipeline(
                                [('s1', cust_txt_col(key='search_term_stemmed')), ('tfidf1', tfidf), ('tsvd1', tsvd)])),
                            ('txt2', pipeline.Pipeline(
                                [('s2', cust_txt_col(key='product_title_stemmed')), ('tfidf2', tfidf), ('tsvd2', tsvd)])),
                            ('txt3', pipeline.Pipeline(
                                [('s3', cust_txt_col(key='product_description_stemmed')), ('tfidf3', tfidf), ('tsvd3', tsvd)]))
                            ],
            transformer_weights={
                'cst': 1.0,
                'txt1': 0.5,
                'txt2': 0.25,
                'txt3': 0.5,
            },
            # n_jobs = -1
        )),
        ('rfr', rfr)])
        # ('xgb', xgb)])

    param_grid = {'rfr__n_estimators': [500], 'rfr__max_features': [10], 'rfr__max_depth': [20]}
    # param_grid = {'xgb__n_estimators': [300, 500],
    #               'xgb__max_depth': [3, 5, 7],
    #               'xgb__learning_rate': [0.05],
    #               'xgb__colsample_bytree': [0.9, 0.95],
    #               'xgb__subsample': [0.8, 0.85, 0.9]
    #               }
    model = grid_search.GridSearchCV(estimator=clf, param_grid=param_grid, n_jobs=1, cv=5, verbose=20, scoring=RMSE)
    model.fit(X_train, y_train)

    print("Best parameters found by grid search:")
    print(model.best_params_)
    print("Best CV score:")
    print(-model.best_score_)

    # from sklearn import cross_validation
    # sub_clf = model.best_estimator_
    # x_sub_fit, x_sub_val, y_sub_fit, y_sub_val = cross_validation.train_test_split(X_train, y_train, random_state=2016, test_size=0.25)
    # sub_clf.fit(x_sub_fit, y_sub_fit, early_stopping_rounds=25, eval_metric="rmse", eval_set=[(x_sub_val, y_sub_val)], verbose=True)

    # print("--- Training & Testing: %s minutes ---" % round(((time.time() - start_time) / 60), 2))
    # exit(0)

    y_pred = model.predict(X_test)
    pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('submission.csv', index=False)
    # pd.DataFrame({"xgb": model.predict(X_train)}).to_csv('stacking_xgb.csv', index=False)

    print("--- Training & Testing: %s minutes ---" % round(((time.time() - start_time) / 60), 2))
