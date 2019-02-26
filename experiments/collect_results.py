"""Tools for handling the results of experiments.

Kyle Roth. 2019-02-26.
"""


from glob import iglob as glob
from os import path
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt


def get_results(directory=''):
    """Return a dictionary mapping model type to a list of results achieved by that model type."""
    results = defaultdict(set)
    for result_file in glob(path.join(directory, '**/results.csv'), recursive=True):
        model_type = result_file.split('/')[1].split('_')[0]
        with open(result_file) as f:
            for line in f:
                score = line.split(',')[-1][:-1]
                if 'score' != score:  # don't use the header
                    results[model_type].add(float(score))
    return results


def reduce_names(l):
    """Reduce the names in the list to acronyms, if possible."""
    for i, item in enumerate(l):
        if item == 'QuadraticDiscriminantAnalysis':
            l[i] = 'QDA'
        elif item == 'KNeighborsClassifier':
            l[i] = 'KNN'
        elif item == 'RandomForestClassifier':
            l[i] = 'RandomForest'
        elif item == 'LogisticRegression':
            l[i] = 'LogReg'
    return l


def plot_best_results(directory='', ax=None):
    """Plot the best result of each model type in a horizontal bar chart."""
    results = get_results(directory)

    # get the best result for each model
    models = []
    bests = []
    for model in results:
        models.append(model)
        bests.append(max(results[model]))

    # reduce the names to acronyms if possible
    models = reduce_names(models)

    # plot
    if ax is None:
        ax = plt.gca()
    ticks = np.arange(len(models))
    ax.barh(ticks, bests)
    ax.set_yticks(ticks)
    ax.set_yticklabels(models, rotation=45)

    ax.set_xlabel('Accuracy')
    ax.set_title('Accuracy of best performances by model type')

    return ax, zip(models, bests)


if __name__ == '__main__':
    axis, bests = plot_best_results()
    print(list(bests))
    plt.show()
