# third-party imports
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, features, mask=None):
        """

        FeatureExtractor([
            ('dist_target', MinDistanceToStructure(grid_name=grid_name, struct_name=target_name)),
            ('dist_target2', 'dist_target**2')
        ],
            mask='dist_target < 10'
        )
        """
        self.feature_funcs, self.feature_eqns = [], []
        for name, f in features:
            target = self.feature_eqns if isinstance(f, str) else self.feature_funcs
            target.append((name, f))

        self.mask = mask

    def transform(self, X):
        dfs = []
        for i, p in enumerate(X):
            df = self.extract(p)
            df.insert(len(df.columns), 'Patient ID', i)
            dfs.append(df)

        return pd.concat(dfs)

    def extract(self, p):
        # create DataFrame of features
        df = pd.DataFrame({name: func(p) for name, func in self.feature_funcs})
        for name, eqn in self.feature_eqns:
            df.eval('%s = %s' % (name, eqn), inplace=True)

        # apply mask criteria
        if self.mask:
            df = df.query(self.mask)

        return df


class MyEstimator(BaseEstimator):

    def __init__(self, extractor, features, target, estimator):
        self.extractor = extractor
        self.features = features
        self.target = target
        self.estimator = estimator

    @property
    def _estimator_type(self):
        return self.estimator._estimator_type

    def extract_features(self, X):
        df = self.extractor.transform(X)
        return df[self.features]

    def extract_target(self, X):
        df = self.extractor.transform(X)
        return df[self.target]

    def fit(self, X, y=None, **fit_params):
        df = self.extractor.transform(X)
        Xt = df[self.features]
        yt = df[self.target]

        self.estimator.fit(Xt, yt, **fit_params)
        return self

    def transform(self, X):
        df = self.extractor.transform(X)
        Xt = df[self.features]

        return self.estimator.transform(Xt)

    def fit_transform(self, X, y=None, **fit_params):
        df = self.extractor.transform(X)
        Xt = df[self.features]
        yt = df[self.target]

        if hasattr(self.estimator, 'fit_transform'):
            return self.estimator.fit_transform(Xt, yt, **fit_params)
        else:
            return self.estimator.fit(Xt, yt, **fit_params).transform(Xt)

    def predict(self, X):
        df = self.extractor.transform(X)
        Xt = df[self.features]

        y_pred = self.estimator.predict(Xt)
        y_pred = y_pred.clip(min=0)

        return y_pred

    def fit_predict(self, X, y=None, **fit_params):
        df = self.extractor.transform(X)
        Xt = df[self.features]
        yt = df[self.target]

        y_pred = self.estimator.fit_predict(Xt, yt, **fit_params)
        y_pred = y_pred.clip(min=0)

        return y_pred
