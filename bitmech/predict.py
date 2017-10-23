#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline, Pipeline, _name_estimators
from sklearn.linear_model import ElasticNet, ElasticNetCV, SGDRegressor
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import log_loss, mean_absolute_error, make_scorer
from sklearn.base import BaseEstimator, TransformerMixin
import xgboost as xgb
import lightgbm as lgb
import catboost as ctb
from sklearn.externals import joblib
import tqdm
import os
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

from logging import basicConfig, getLogger, INFO
basicConfig(level=INFO, format='%(asctime)-15s %(levelname)s %(filename)-s:%(lineno)-s: %(message)s')
logger = getLogger(__name__)
info = lambda x: logger.info(x)

"""
pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
"""

class EnsembleTimeSeries(object):
    def __init__(self, n_splits, stacker, base_models):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)
        xshape = X.shape[0]
        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))


        for i, (name, clf) in enumerate(self.base_models.items()):
            print("Model: {} ".format(name))

            S_test_i = np.zeros((T.shape[0], self.n_splits))
            model_name = "model_{}_{}.pkl".format(os.path.basename(__file__), name)

            if os.path.exists(model_name):
                clf = joblib.load(model_name)
            for j in range(0, self.n_splits):
                preindex = xshape*(j)/self.n_splits
                index = xshape*(j+1)/self.n_splits
                tindex = int((index-preindex) * testsplit)+preindex
                train_idx = range(preindex, tindex)
                test_idx = range(tindex, index)
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                y_holdout = y[test_idx]
                if name == "ctb":
                    clf.fit(X_train, y_train, catboostregressor__eval_set=(X_train, y_train))
                else:
                    clf.fit(X_train, y_train)
                y_pred = clf.predict(X_holdout)[:]

                print("Model %s fold %d score %f" % (name, j, mean_absolute_error(y_holdout, y_pred)))

                S_train[test_idx, i] = y_pred
                res = clf.predict(T)
                S_test_i[:, j] = res
            joblib.dump(clf, model_name)
            #results = clf.score(X, y)
            #print("%s score: %.4f (%.4f)" % (name, results.mean(), results.std()))
            S_test[:, i] = clf.predict(T)
            """
            res = []
            for i in tqdm.tqdm(range(T.shape[0])):
                res.append(clf.predict(T[:i+1, :])[-1])
            return np.array(res)
            """

        print(S_train, S_test)
        self.stacker.fit(S_train, y)
        res = self.stacker.predict(T)
        return res

class AddColumns(BaseEstimator, TransformerMixin):
    def __init__(self, transform_=None):
        self.transform_ = transform_

    def fit(self, X, y=None):
        self.transform_.fit(X, y)
        return self

    def transform(self, X, y=None):
        try:
            xform_data = self.transform_.transform(X, y)
        except:
            xform_data = self.transform_.transform(X)
        return np.append(X, xform_data, axis=1)


class LogExpPipeline(Pipeline):
    def fit(self, X, y):
        super(LogExpPipeline, self).fit(X, np.log1p(y))

    def predict(self, X):
        return np.expm1(super(LogExpPipeline, self).predict(X))


import backtest
bt = backtest.Backtest()
bt.init_candle()
bt.resetbacktest()
bt.index = bt.size
bt.updateIndicators()
truth = bt.df

memory = joblib.Memory(cachedir=".")
n = truth.shape[1]
#
# XGBoost model
#
xgb_params = {}
xgb_params['objective'] = 'reg:linear'
xgb_params['learning_rate'] = 0.001
xgb_params['max_depth'] = int(6.0002117448743721)
xgb_params['max_depth'] = 9
xgb_params['subsample'] = 0.72476106045336319
xgb_params['min_child_weight'] = int(4.998433055249718)
#xgb_params['colsample_bytree'] = 0.97058965304691203
#xgb_params['colsample_bylevel'] = 0.69302144647951536
xgb_params['reg_alpha'] = 0.59125639278096453
xgb_params['gamma'] = 0.11900602913417056
xgb_params['silent'] = 1
xgb_model = xgb.sklearn.XGBRegressor(**xgb_params)
xgb_pipe = Pipeline(_name_estimators([AddColumns(transform_=PCA(n_components=n)),
                                      AddColumns(transform_=FastICA(n_components=n, max_iter=500)),
                                      xgb_model]), memory=memory)
xgb_pipe = make_pipeline(xgb_model, memory=memory)

ctb_params = {}
ctb_params['learning_rate'] = 0.003
ctb_params['iterations'] = 1000
ctb_params['depth'] = int(9.4640351140450782)
#ctb_params['od_pval'] =  0.75478964184949227
#ctb_params['bagging_temperature'] = 0.88941313109465181
#ctb_params['l2_leaf_reg'] = int(2.9284212808458823)
#ctb_params['rsm'] = 0.80067757495084524
ctb_params['use_best_model'] = True
ctb_params['loss_function'] = 'RMSE'
ctb_params['eval_metric'] = 'RMSE'
ctb_model = ctb.CatBoostRegressor(**ctb_params)
ctb_pipe = Pipeline(_name_estimators([AddColumns(transform_=PCA(n_components=n)),
                                      AddColumns(transform_=FastICA(n_components=n, max_iter=500)),
                                      ctb_model]), memory=memory)
ctb_pipe = make_pipeline(ctb_model, memory=memory)

lgb_params = {}
lgb_params['boosting_type'] = 'gbdt'
lgb_params['objective'] = 'regression'
lgb_params['metric'] = 'rmse'
lgb_params['learning_rate'] = 0.001
lgb_params['num_leaves'] = int(641.25017321994062)
lgb_params['min_data_in_leaf'] = int(20.050723378367028)
lgb_params['max_depth'] = int(10.388852633004857)
lgb_params['subsample'] = 0.86987167700893142
lgb_params['feature_fraction'] = 0.50386338591314295
lgb_params['bagging_fraction'] = 0.99818032677189694
lgb_params['bagging_freq'] = int(3.0103968337769231)
lgb_params['early_stopping_round'] = 20
lgb_params['bagging_seed'] = 3
lgb_model = lgb.LGBMRegressor(**lgb_params)
lgb_pipe = Pipeline(_name_estimators([AddColumns(transform_=PCA(n_components=n)),
                                      AddColumns(transform_=FastICA(n_components=n, max_iter=500)),
                                      lgb_model]), memory=memory)
lgb_pipe = make_pipeline(lgb_model, memory=memory)


#results = cross_val_score(xgb_model, train, y_train1, cv=5, scoring='mae')
#print("XGB score: %.4f (%.4f)" % (results.mean(), results.std()))

xgb_model2 = xgb.sklearn.XGBRegressor(**xgb_params)
lgb_model2 = lgb.LGBMRegressor(**lgb_params)

stack1 = EnsembleTimeSeries(n_splits=10,
                 #stacker=ElasticNetCV(l1_ratio=[x/10.0 for x in range(1,10)]),
                 #stacker=ElasticNet(alpha=0.001, l1_ratio=0.1),
                 #stacker=SGDRegressor(penalty='elasticnet', alpha=0.001, l1_ratio=0.1),
                 stacker=lgb_model2,
                 base_models={
                         #'svm': svm_pipe,
                         #'en': en_pipe,
                         #'xgb': xgb_pipe,
                         'lgb': lgb_pipe,
                         'ctb': ctb_pipe,
                         #'rf': rf_model
                         })
ntruth = truth.shape[0] - 1
fold = 10
testsplit=0.8
col = {x: truth.columns.tolist().index(x) for x in ["open", "high", "low", "close", "vwp", "volume", "start"]}
print(col)
print(truth.shape)

test_i = 30
ids = np.array(range(bt.status.shape[0])).reshape(bt.status.shape[0], 1)
print(ids)
print(ids.shape)
print(bt.status)
bt.status = np.append(bt.status, ids, axis=1)
print(bt.status)

# 一つ前のindicatorの値を使って予測する
X = np.roll(bt.status, 1)
X = X[:-test_i, :]
y = truth["close"].values
psma = bt.moving_average(y, 9)
y = y[:-test_i]
psma = psma[:-test_i]
prevclose = np.roll(y, 1).reshape(y.shape[0], 1)
prevsma = np.roll(psma, 1).reshape(y.shape[0], 1)
X = np.append(X, prevclose, axis=1)
X = np.append(X, prevsma, axis=1)
X[0, :] = float("nan")

#最後の1行をpredictする
test = X[-1, :].reshape(1, X.shape[1])
X = X[:-1, :]
y = y[:-1]
print(y)
print(test.shape)
#pred1 = stack1.fit_predict(X, y, test)
model_name = "model_predict.py_lgb.pkl"
name = "lgb"
clf = joblib.load(model_name)
if name == "ctb":
    clf.fit(X, y, catboostregressor__eval_set=(X, y))
else:
    clf.fit(X, y)
pred1 = clf.predict(test)

print("="*50)
base = truth["close"].values[-30]
print(test)
print(pred1, base)
print("="*50)

pred = np.zeros(truth.shape[0])
for i in range(test_i):
    index = test_i - i
    print(index)
    # 一つ前のindicatorの値を使って予測する
    X = np.roll(bt.status, 1)
    X = X[:-index, :]
    y = truth["close"].values
    psma = bt.SMApd(y, 9)
    y = y[:-index]
    psma = psma[:-index]
    print(y.shape, psma.shape, X.shape)
    prevclose = np.roll(y, 1).reshape(y.shape[0], 1)
    prevsma = np.roll(psma, 1).reshape(psma.shape[0], 1)
    print(prevclose)
    X = np.append(X, prevclose, axis=1)
    X = np.append(X, prevsma, axis=1)
    X[0, :] = float("nan")
    print(X.shape)
    print(X)
    test = X[-1:, :]
    X = X[:-1, :]
    y = y[:-1]
    print(test, y)

    #pred1 = stack1.fit_predict(X, y, test)
    pred1 = clf.predict(test)
    print(pd.DataFrame(test), pred1)
    pred[i] =  pred1
print(test)
print("="*50)
print(pred[:30])
base = truth["close"].values[-30:]
print(base)
print("="*50)

filename = './input/submission_stack.{}.{}.tsv'.format(os.path.basename(__file__), datetime.now().strftime('%Y%m%d_%H%M%S'))
print("Writing predictions to {}".format(filename))

#joined.to_csv(filename, index=False, sep="\t", header=False)

