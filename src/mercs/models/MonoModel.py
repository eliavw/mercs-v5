

# Classes
class MonoModel(object):
    """
    Class that hosts the most low-level models of MERCS

    I.e.: a wrapper class for underlying, external models,
    e.g.:
        sklearn.tree.DecisionTreeClassifier
        XGBoost
    """

    def __init__(self, model, **kwargs):
        self.model = model(**kwargs)

    def fit(self, X, **kwargs):
        self.model.fit(X, **kwargs)

    def predict_proba(self, X, **kwargs):
        return self.model.predict_proba(X, **kwargs)

    def predict(self, X, **kwargs):
        return self.model.predict(X, **kwargs)
