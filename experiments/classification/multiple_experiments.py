"""Module for running multiple experiments with mag and hyperopt using a nifty command line interface.

Seong-Eun Cho. Documented by Kyle Roth, 2019-03-09.
"""


import argparse
import json
import time
import datetime
import ast
import csv

import numpy as np

from hyperopt import hp  # pylint: disable=unused-import
from hyperopt import fmin, tpe, space_eval
from hyperopt.pyll.base import scope

from mag.experiment import Experiment

from speech2phone.preprocessing.TIMIT.phones import get_data
from speech2phone.preprocessing.filters import mel

from speech2phone.experiments.classification.single_experiment import run_model, get_model


def do_exp(params, _dir, X_train, y_train, X_test, y_test, result_dict):
    """Perform an experiment using the specified parameters.
    
    Args:
        params (dict): specific hyperparameter set to use.
    Returns:
        (float): score found using specified hyperparameters.
    """
    try:
        with Experiment(config=params, experiments_dir=_dir) as experiment:
            score = run_model(model, X_train, y_train, X_test, y_test, params)
            # save the params and score
            for k in params.keys():
                result_dict[k].append(params[k])
            experiment.register_result('score', score)
    except ValueError:
        # if something breaks, return the worst score possible
        return np.inf
    return -score  # pylint: disable=invalid-unary-operand-type


def multiple_experiments(model, data, space, max_evals):  # pylint: disable=too-many-locals
    """Use hyperopt to sample from the space of hyperparameters, applying the model to the data and storing the results
    of each such experiment using mag.

    Args:
        model (str): name of callable model constructor in the current namespace.
        data (str): specify the TIMIT data sets to use. If specified, must be one of {'full', 'toy'}.
        space (dict): dictionary describing the hyperopt search space to be used.
        max_evals (int): number of evaluations hyperopt can run before quitting.
    """
    # prepare the experiment directory
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
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
    
    def wrapper(params):
        """Wrapper for experiment function."""
        score = do_exp(params, _dir, X_train, y_train, X_test, y_test)
        result_dict['score'].append(-score)
        return score

    result_dict = {k:[] for k in space.keys()}
    result_dict['score'] = []
    best = fmin(wrapper, space, algo=tpe.suggest, max_evals=max_evals)
    print("Best Raw:", best)
    print("Best Readable:", space_eval(space, best))

    with open(_dir + "results.csv", "w") as outfile:
        writer = csv.writer(outfile, delimiter=',')
        writer.writerow(result_dict.keys())
        writer.writerows(zip(*result_dict.values()))


def create_hyperopt_space(string_space):
    """Convert the parameters passed from the command line into the corresponding hyperopt space definition.

    WARNING: Uses eval() to convert strings to names in the namespace.

    For example,

        {'C': 'float:hp.lognormal:[0, 1]', 'kernel': 'rbf', 'gamma': 'float:hp.lognormal:[0, 1]'}

    gets converted to

        {'C': float(hp.lognormal('C', 0, 1)), 'kernel': 'rbf', 'gamma': float(hp.lognormal('gamma', 0, 1))}

    Args:
        string_space (dict): dictionary of space definitions taken from the raw JSON.
    Returns:
        (dict): space dictionary ready to be used by hyperopt.
    """
    space = {}
    for k, v in string_space.items():
        if ':' in v:  # contains specifications that need to be converted to hyperopt ranges
            p_type, p_func, p_params = v.split(':')
            p_params = ast.literal_eval(p_params)
            if p_type == 'int':
                # convert to integer after evaluation
                space[k] = scope.int(eval(p_func)(k, *p_params))  # pylint: disable=eval-used
            elif p_type == 'float':
                # no need to convert because hyperopt natively returns floats
                space[k] = eval(p_func)(k, *p_params)  # pylint: disable=eval-used
            else:
                raise TypeError("parameter types must be int or float")
        else:  # is a constant string
            space[k] = v
    return space


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
    parser.add_argument('--space',
                        help="Hyperparameter space to search optimal values over",
                        type=str,
                        required=True)
    parser.add_argument('--max-evals',
                        help="Number of evaluations when searching",
                        type=int,
                        required=True)

    args = parser.parse_args()
    hyperopt_space = json.loads(args.space)
    hyperopt_space = create_hyperopt_space(hyperopt_space)

    multiple_experiments(args.model, args.data, hyperopt_space, args.max_evals)
