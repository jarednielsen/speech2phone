import numpy as np

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from hyperopt import hp
from hyperopt import fmin, tpe, space_eval
from hyperopt.pyll.base import scope

import mag
from mag.experiment import Experiment
from mag import summarize

from speech2phone.preprocessing import get_TIMIT, get_phones
from speech2phone.preprocessing.filters import mel

import argparse
import json

class Decoder(json.JSONDecoder):
    def decode(self, s):
        result = super().decode(s)
        return self._decode(result)

    def _decode(self, o):
        if isinstance(o, str) or isinstance(o, str):
            try:
                if float(o) == int(float(o)):
                    return int(float(o))
                else:
                    return float(o)
            except ValueError:
                return o
        elif isinstance(o, dict):
            return {k: self._decode(v) for k, v in o.items()}
        elif isinstance(o, list):
            return [self._decode(v) for v in o]
        else:
            return o

class XGBoost:
    def __init__(self, **params):
        self.num_round = params["num_round"]
        del params["num_round"]
        self.params = params
        self.bst = None
    
    def fit(self, X, y):
        dtrain = xgb.DMatrix(X, label=y)
        self.bst = xgb.train(self.params, dtrain, self.num_round)
    
    def score(self, X, y):
        dtest = xgb.DMatrix(X, label=y)
        y_pred = self.bst.predict(dtest)
        return sum(y_pred == y) / len(y)

def run_model(model, X_train, y_train, X_test, y_test, params):
    clf = model(**params)
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)

def main(model, data, _dir, params):
    try:
        model = eval(model)
    except NameError:
        print("Model must be one of the following:\n\
               RandomForestClassifier,\n\
               QuadraticDiscriminantAnalysis,\n\
               MultinomialNB,\n\
               LogisticRegression,\n\
               SVC,\n\
               KNeighborsClassifier,\n\
               KMeans,\n\
               GaussianMixture,\n\
               XGBoost")
        raise
    
    if data == "toy":
        X_train, y_train = get_TIMIT(dataset='toy', 
                                     preprocessor=mel, 
                                     TIMIT_root='../TIMIT/TIMIT', 
                                     use_cache=True)
        X_test, y_test = X_train, y_train
    elif data == "full":
        X_train, y_train = get_TIMIT(dataset='train', 
                                     preprocessor=mel, 
                                     TIMIT_root='../TIMIT/TIMIT', 
                                     use_cache=True)
        X_test, y_test = get_TIMIT(dataset='val', 
                                   preprocessor=mel, 
                                   TIMIT_root='../TIMIT/TIMIT', 
                                   use_cache=True)
    else:
        print("Data must be either 'toy' or 'full'.")
        raise
    
    with Experiment(config=params, experiments_dir=_dir) as experiment:
        config = experiment.config
        score = run_model(model, X_train, y_train, X_test, y_test, params)
        experiment.register_result('score', score)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', 
                        help="Classification model. \
                        Must be one of the following:\n\
                        RandomForestClassifier,\n\
                        QuadraticDiscriminantAnalysis,\n\
                        MultinomialNB,\n\
                        LogisticRegression,\n\
                        SVC,\n\
                        KNeighborsClassifier,\n\
                        KMeans,\n\
                        GaussianMixture,\n\
                        XGBoost", 
                        type=str, 
                        required=True)
    parser.add_argument('--data', 
                        help="Data to use. Either 'toy' or 'full'", 
                        type=str, 
                        required=True)
    parser.add_argument('--params', 
                        help="Hyperparameters to be used on the model. \
                              Must be of the form '{\"keyword\": \"value\", ...}'", 
                        type=str,
                        required=True)
    parser.add_argument('--dir', 
                        help="Directory to save the experiment", 
                        type=str, 
                        required=True)
    
    args = parser.parse_args()
    params = json.loads(args.params, cls=Decoder)
    print(params)
    main(args.model, args.data, args.dir, params)
