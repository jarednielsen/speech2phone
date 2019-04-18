"""Run every model with every preprocessor."""


from speech2phone.temp_kyle.multiple_experiments import multiple_experiments, create_hyperopt_space


preprocs = {
    'mel'
}

fcnn_space = {
    'num_layers': 'int:hp.uniform:[2,20]',
    'eta': 'float:hp.loguniform:[-9,-2]',
    'hidden_dim': 'int:hp.loguniform:[4,7]',
    'epochs': 20
}

rf_space = {
    'n_estimators': '[10,300]',
    'max_depth': '[2,50]'
}

knn_space = {
    'n_neighbors': 'int:hp.uniform:[2,20]'
}

xgboost_space = {
    "max_depth": "int:hp.uniform:[5,50]",
    "eta": "float:hp.loguniform:[-1,1]",
    "gamma": "float:hp.loguniform:[-4,4]",
    "lambda": "float:hp.loguniform:[-4,4]",
    "alpha": "float:hp.loguniform:[-4,4]",
    "num_class": 61,
    "eval_metric": "merror",
    "num_round": 10
}

spaces = {
    'fcnn': create_hyperopt_space(fcnn_space),
    'randomforestclassifier': create_hyperopt_space(rf_space),
    'kneighborsclassifier': create_hyperopt_space(knn_space),
    'xgboost': create_hyperopt_space(xgboost_space)
}


def main():
    """Run the models with each preprocessor using multiple_experiments."""
    for preproc in preprocs:
        for model in spaces:
            multiple_experiments(model, 'full', spaces[model], 5, preproc)
            


if __name__ == '__main__':
    main()