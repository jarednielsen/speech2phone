"""Find the best hyperparameters for each model class.

Kyle Roth. 2019-04-16.
"""


from collections import defaultdict
from concurrent import futures
import csv
from glob import iglob as glob
import os
from shutil import rmtree
import sys

# import psutil
import pandas as pd
from mag.experiment import Experiment

from speech2phone.preprocessing.TIMIT.phones import get_data
from speech2phone.preprocessing.filters import get_filter
from speech2phone.experiments.classification.single_experiment import run_model, get_model


def get_top_params(loc='results', top_num=5):
    """Get the hyperparameters of the top five performing runs of each model.

    Yields:
        (str): name of model run.
        (dict): model parameters used.
    """
    for result_file in glob(os.path.join(loc, '*/results.csv')):
        df = pd.read_csv(result_file)

        if 'score' not in df.columns:
            print('No score column in {}'.format(result_file))
            continue

        for _, row in df.nlargest(top_num, columns=['score']).iterrows():
            thing = {}
            for item, value in row.iteritems():
                if item != 'score':
                    thing[item] = value
            model_temp = result_file.split('/')[1]
            if model_temp.split('_')[1][0].isdigit():
                yield model_temp.split('_')[0], thing
            else:
                yield model_temp, thing


def do_exp(model_name, params, _dir, preproc_name, padding):
    """Perform an experiment using the specified parameters.

    Args:
        params (dict): specific hyperparameter set to use.
    Returns:
        (dict): score found using specified hyperparameters.
    """
    model = get_model(model_name)
    preproc = get_filter(preproc_name)
    X_train, y_train = get_data(dataset='train', preprocessor=preproc, TIMIT_root='../../TIMIT/TIMIT', padding=padding)
    X_test, y_test = get_data(dataset='val', preprocessor=preproc, TIMIT_root='../../TIMIT/TIMIT', padding=padding)
    try:
        with Experiment(config=params, experiments_dir=_dir) as experiment:
            score = run_model(model, X_train, y_train, X_test, y_test, params)
            experiment.register_result('score', score)
    except ValueError:
        # if something breaks, return the worst score possible
        return 0
    return score


def run_best_models_val(max_jobs=6, padding=(0,)):
    """Run the top-performing models using various preprocessors, reporting results on the validation set."""
    # try getting memory constraints
    # https://stackoverflow.com/questions/9850995/tracking-maximum-memory-usage-by-a-python-function
    # mem_cap = psutil.virtual_memory().total

    # create the queue of experiments to run
    preprocessors = ('mel', 'resample')
    model_runs = get_top_params()
    to_run = []
    for preprocessor in preprocessors:
        for model_name, params in model_runs:
            for pad in padding:
                to_run.append((preprocessor, model_name, params, pad))
    num_left = len(to_run)
    to_run = iter(to_run)

    with futures.ProcessPoolExecutor() as executor:
        future_to_params = {}  # dictionary to store references to results from other threads

        while num_left:
            for preproc_name, model_name, params, pad in to_run:
                print('Running {} with {} (padding {}):\n\t{}'.format(model_name, preproc_name, pad, params))
                # prepare the experiment directory
                _dir = "./final_val/{}_{}_{}/".format(model_name, preproc_name, pad)

                # submit the experiment to another thread
                callback = executor.submit(do_exp, model_name, params, _dir, preproc_name, pad)
                future_to_params[callback] = (model_name, params, _dir, preproc_name, pad)
                if len(future_to_params) >= max_jobs:
                    break  # don't start too many jobs at once

            # get the results as they finish, recording them in results.csv
            for callback in futures.as_completed(future_to_params):
                model_name, params, _dir, preproc_name, pad = future_to_params[callback]
                print('Got result from {} with {} (padding {}):\n\t{}'.format(model_name, preproc_name, pad, params))
                num_left -= 1
                try:
                    record = params.copy()
                    record['score'] = callback.result()
                    del future_to_params[callback]  # make room for more processes
                except futures.process.BrokenProcessPool:
                    print('    Out of memory! Try again later.')
                    del future_to_params[callback]
                    break

                os.makedirs(_dir, exist_ok=True)
                if not os.path.isfile(os.path.join(_dir, 'results.csv')):
                    with open(os.path.join(_dir, 'results.csv'), 'w+') as outfile:
                        writer = csv.writer(outfile, delimiter=',')
                        writer.writerow(record.keys())
                        writer.writerow(record.values())
                else:
                    with open(os.path.join(_dir, 'results.csv'), 'a') as outfile:
                        writer = csv.writer(outfile, delimiter=',')
                        writer.writerow(record.values())
                break  # add another job


def run_best_models_test(max_jobs=6):
    """Run the top-performing models from run_best_models_val, reporting results on the test set."""
    # create the queue of experiments to run
    val_runs = get_top_params('final_val', top_num=1)
    to_run = []
    for model_preproc_pad, params in val_runs:
        model, preproc, pad = model_preproc_pad.split('_')
        to_run.append((preproc, model, params, pad))
    num_left = len(to_run)
    to_run = iter(to_run)

    with futures.ProcessPoolExecutor() as executor:
        future_to_params = {}  # dictionary to store references to results from other threads
        while num_left:
            for preproc_name, model_name, params, pad in to_run:
                print('Running {} with {} (padding {}):\n\t{}'.format(model_name, preproc_name, pad, params))
                # prepare the experiment directory
                _dir = "./final_test/{}_{}_{}/".format(model_name, preproc_name, pad)

                # submit the experiment to another thread
                callback = executor.submit(do_exp, model_name, params, _dir, preproc_name, pad)
                future_to_params[callback] = (model_name, params, _dir, preproc_name, pad)
                if len(future_to_params) >= max_jobs:
                    break  # don't start too many jobs at once

            # get the results as they finish, recording them in results.csv
            for callback in futures.as_completed(future_to_params):
                model_name, params, _dir, preproc_name, pad = future_to_params[callback]
                print('Got result from {} with {} (padding {}):\n\t{}'.format(model_name, preproc_name, pad, params))
                num_left -= 1
                try:
                    record = params.copy()
                    record['score'] = callback.result()
                    del future_to_params[callback]  # make room for more processes
                except futures.process.BrokenProcessPool:
                    print('    Out of memory! Try again later.')
                    del future_to_params[callback]
                    break

                os.makedirs(_dir, exist_ok=True)
                if not os.path.isfile(os.path.join(_dir, 'results.csv')):
                    with open(os.path.join(_dir, 'results.csv'), 'w+') as outfile:
                        writer = csv.writer(outfile, delimiter=',')
                        writer.writerow(record.keys())
                        writer.writerow(record.values())
                else:
                    with open(os.path.join(_dir, 'results.csv'), 'a') as outfile:
                        writer = csv.writer(outfile, delimiter=',')
                        writer.writerow(record.values())
                break  # add another job


def clean_up(loc):
    """Remove unfinished result directories and entries in results.csv."""
    for model_folder in glob(os.path.join(loc, '*')):
        # delete folders with no results.csv
        if not os.path.isfile(os.path.join(model_folder, 'results.csv')):
            print('removing', model_folder)
            rmtree(model_folder)
            continue

        # delete empty results.csv
        if os.stat(os.path.join(model_folder, 'results.csv')).st_size == 0:
            print('removing', model_folder)
            rmtree(model_folder)
            continue

        # delete directories of unfinished experiments
        for exp_folder in glob(os.path.join(model_folder, '*')):
            if os.path.isdir(exp_folder):
                if not os.path.isfile(os.path.join(exp_folder, 'results.json')):
                    print('removing', exp_folder)
                    rmtree(exp_folder)

        # remove lines of results.csv where the score was zero
        lines_to_keep = []
        with open(os.path.join(model_folder, 'results.csv'), 'r') as results_file:
            try:
                header = next(results_file)
                lines_to_keep.append(header)
                score_idx = header.strip().split(',').index('score')
            except ValueError:
                rmtree(model_folder)
                continue
            for line in results_file:
                if line.strip().split(',')[score_idx] in ('0', '0.0'):
                    print('removing line {} from {}'.format(line.strip(), model_folder))
                else:
                    lines_to_keep.append(line)
        with open(os.path.join(model_folder, 'results.csv'), 'w') as results_file:
            results_file.writelines(lines_to_keep)


if __name__ == '__main__':
    clean_up('final_val')
    if len(sys.argv) <= 1 or sys.argv[1].lower() not in ('clean', 'test'):
        print()
        if len(sys.argv) > 1:
            run_best_models_val(int(sys.argv[1]), padding=[0, 100, 400])
        else:
            run_best_models_val(padding=[0, 100, 400])
    elif sys.argv[1].lower() == 'test':
        print()
        if len(sys.argv) > 2:
            run_best_models_test(int(sys.argv[2]))
        else:
            run_best_models_test()
