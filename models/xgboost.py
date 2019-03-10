"""XGBoost wrapper implementing .fit and .score methods.

Seong-Eun Cho. 2019-03-09.
"""


import xgboost as xgb


class XGBoost:
    """Wrapper for xgboost with sklearn-style API."""
    def __init__(self, **params):
        """Store the specified parameters to the xgboost model.

        Args:
            **params (kwargs): parameters to specify to xgboost.train.
        """
        self.num_round = params["num_round"]
        del params["num_round"]
        self.params = params
        self.bst = None

    def fit(self, X, y):
        """Train xgboost on the provided data.

        Args:
            X (list-like): data features.
            y (list-like): data labels.
        """
        train = xgb.DMatrix(X, label=y)
        self.bst = xgb.train(self.params, train, self.num_round)

    def score(self, X, y):
        """Apply the xgboost model to data and return the accuracy score.

        Args:
            X (list-like): data features.
            y (list-like): data labels.
        Returns:
            (float): accuracy score of model's predictions.
        """
        test = xgb.DMatrix(X, label=y)
        y_pred = self.bst.predict(test)
        return sum(y_pred == y) / len(y)
