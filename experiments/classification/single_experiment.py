"""Module for running a single experiment with mag using a nifty command line interface.

Seong-Eun Cho. Documented by Kyle Roth, 2019-03-09.
"""


import argparse
import json
import time
import datetime
from math import ceil

from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from mag.experiment import Experiment

from speech2phone.preprocessing.TIMIT.phones import get_data
from speech2phone.preprocessing.filters import mel

from speech2phone.models import XGBoost, FCNN


class Decoder(json.JSONDecoder):
    """Class for decoding JSON by overriding the default json.JSONDecoder.

    Converts strings that look like ints or floats to those data types.
    """

    def decode(self, s):
        result = super().decode(s)
        return self._decode(result)

    def _decode(self, o):
        if isinstance(o, str):
            # try converting to float or int
            try:
                if float(o) == int(float(o)):
                    return int(float(o))
                return float(o)
            except ValueError:  # couldn't convert to number
                return o
        elif isinstance(o, dict):
            # decode each of the items in the dict
            return {k: self._decode(v) for k, v in o.items()}
        elif isinstance(o, list):
            # decode each of the items in the list
            return [self._decode(v) for v in o]
        return o


def get_model(name):  # pylint: disable=too-many-return-statements
    """Get the model constructor specified by the name given.

    Args:
        name (str): name of model constructor to be called.
    Returns:
        (callable): model constructor to be called.
    """
    name = name.lower()
    if name == 'randomforestclassifier':
        return RandomForestClassifier
    if name == 'quadraticdiscriminantanalysis':
        return QuadraticDiscriminantAnalysis
    if name == 'multinomialnb':
        return MultinomialNB
    if name == 'logisticregression':
        return LogisticRegression
    if name == 'svc':
        return SVC
    if name == 'kneighborsclassifier':
        return KNeighborsClassifier
    if name == 'kmeans':
        return KMeans
    if name == 'gaussianmixture':
        return GaussianMixture
    if name == 'xgboost':
        return XGBoost
    if name == 'fcnn':
        return FCNN
    raise ValueError(
        "model must be one of {RandomForestClassifier, QuadraticDiscriminantAnalysis, MultinomialNB, " \
        "LogisticRegression, SVC, KNeighborsClassifier, KMeans, GaussianMixture, XGBoost, FCNN}, " +
        "got '{}'".format(name)
    )


def run_model(model, X_train, y_train, X_test, y_test, params):  # pylint: disable=too-many-arguments
    """Train the given model on the training data, and return the score for the model's predictions on the test data.

    Specify parameters of the model with the provided params.

    Args:
        model (callable): model to be initialized with the specified parameters.
        X_train (list-like): features of train set.
        y_train (list-like): labels of train set.
        X_test (list-like): features of test set.
        y_test (list-like): labels of test set.
        params (dict): parameters to specify to the model.
    Returns:
        (float): the accuracy score of the model's predictions on the test set after being trained on the train set.
    """
    # convert anything that can be to integers
    for k in params:
        if isinstance(params[k], float) and ceil(params[k]) == params[k]:
            params[k] = int(params[k])
    clf = model(**params)
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)


def single_experiment(model, data, params):
    """Apply the model to the data and store the results using mag.

    Args:
        model (str): name of callable model constructor in the current namespace.
        data (str): specify the TIMIT data sets to use. If specified, must be one of {'full', 'toy'}.
        params (dict): dictionary with parameters for model.
    """
    # prepare the experiment directory
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
    _dir = "./results/" + model + "_" + st
    if data is not None and data.lower() == 'toy':
        _dir = "./results/TOY_" + model + "_" + st + '/'
    else:
        _dir = "./results/" + model + "_" + st + '/'

    # select the model to be used
    model = get_model(model)

    # get the specified dataset
    if data is None:
        data = "full"
    if data.lower() == "toy":
        X_train, y_train = get_data(dataset='toy', preprocessor=mel, TIMIT_root='../../TIMIT/TIMIT', use_cache=True)
        X_test, y_test = X_train, y_train
    elif data == "full":
        X_train, y_train = get_data(dataset='train', preprocessor=mel, TIMIT_root='../../TIMIT/TIMIT', use_cache=True)
        X_test, y_test = get_data(dataset='val', preprocessor=mel, TIMIT_root='../../TIMIT/TIMIT', use_cache=True)
    else:
        raise ValueError("data must be one of {'toy', 'full'}")

    with Experiment(config=params, experiments_dir=_dir) as experiment:
        score = run_model(model, X_train, y_train, X_test, y_test, params)
        experiment.register_result('score', score)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        help='Classification model. Must be one of {RandomForestClassifier, ' \
                            'QuadraticDiscriminantAnalysis, MultinomialNB, LogisticRegression, SVC, ' \
                            'KNeighborsClassifier, KMeans, GaussianMixture, XGBoost, FCNN',
                        type=str,
                        required=True)
    parser.add_argument('--data',
                        help="Data to use. One of {'toy', 'full'}",
                        type=str,
                        required=False)
    parser.add_argument('--params',
                        help="Hyperparameters to be used on the model. \
                              Must be of the form '{\"keyword\": \"value\", ...}'",
                        type=str,
                        required=True)

    args = parser.parse_args()
    model_params = json.loads(args.params, cls=Decoder)
    single_experiment(args.model, args.data, model_params)
