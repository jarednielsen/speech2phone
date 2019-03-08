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

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.autograd import Variable

from hyperopt import hp
from hyperopt import fmin, tpe, space_eval
from hyperopt.pyll.base import scope

import mag
from mag.experiment import Experiment
from mag import summarize

from speech2phone.preprocessing.TIMIT.phones import get_data, get_phones
from speech2phone.preprocessing.filters import mel

import argparse
import json
import time
import datetime

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

class TimitMelClassifier(nn.Module):
    def __init__(self):
        super(TimitMelClassifier, self).__init__()
        embedding_dim = 80
        output_dim = 61
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, inp):
        return self.net(inp)

class TimitMelDataset(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class FCNN:
    def __init__(self, batch_size=128, epochs=100, eta=1e-4):
        self.batch_size = batch_size
        self.epochs = epochs
        self.eta = eta
        self.model = TimitMelClassifier()
        self.objective = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.eta)
        self.dataset = None
        self.dataloader = None

    def fit(self, X, y):
        self.dataset = TimitMelDataset(X, y)
        self.dataloader = DataLoader(self.dataset,
                                     batch_size=self.batch_size,
                                     shuffle=True)
        self._train()

    def score(self, X, y):
        y_pred = self.model(Variable(torch.from_numpy(X)).float()).argmax(dim=1).detach().numpy()
        return sum(y_pred == y) / len(y)

    def _train(self):
        for e in range(self.epochs):
            for batch, (x, y_truth) in enumerate(self.dataloader):
                self.optimizer.zero_grad()
                y_hat = self.model(x.float())
                loss = self.objective(y_hat, y_truth.long())
                loss.backward()
                self.optimizer.step()

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

def single_experiment(model, data, params):
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
    _dir = "./results/" + model + "_" + st
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
               XGBoost,\n\
               FCNN")
        raise

    if data == None: data = "full"
    if data == "toy":
        X_train, y_train = get_data(dataset='toy',
                                    preprocessor=mel,
                                    TIMIT_root='../TIMIT/TIMIT',
                                    use_cache=True)
        X_test, y_test = X_train, y_train
    elif data == "full":
        X_train, y_train = get_data(dataset='train',
                                    preprocessor=mel,
                                    TIMIT_root='../TIMIT/TIMIT',
                                    use_cache=True)
        X_test, y_test = get_data(dataset='val',
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
                        XGBoost,\n\
                        FCNN",
                        type=str,
                        required=True)
    parser.add_argument('--data',
                        help="Data to use. Either 'toy' or 'full'",
                        type=str,
                        required=False)
    parser.add_argument('--params',
                        help="Hyperparameters to be used on the model. \
                              Must be of the form '{\"keyword\": \"value\", ...}'",
                        type=str,
                        required=True)

    args = parser.parse_args()
    params = json.loads(args.params, cls=Decoder)
    single_experiment(args.model, args.data, params)
