# third-party imports
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class VoxelFeatureExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, features, mask=None):
        """Transformer that extracts voxel-wise features from a cohort of
        Patient objects and returns a pandas.DataFrame object. Each row
        corresponds to a voxel (patients are tracked by 'Patient ID' feature).
        Voxel selection criteria are supported by the mask argument.

        Example:
            VoxelFeatureExtractor([
                ('dist_target', MinDistanceToStructure(grid_name, target_name)),
                ('dist_target2', 'dist_target**2')
            ],
                mask='0 < dist_target < 10'
            )

        Args:
            features: list of 2-tuples like (feature_name, feature_func) where
                feature_func is either:
                    - function called on each Patient object
                    - expression using other feature_name variables

            mask: Boolean expression using feature_name variables

        Note: expressions using feature_name variables use pandas.eval()
        http://pandas.pydata.org/pandas-docs/stable/enhancingperf.html#expression-evaluation-via-eval-experimental
        """
        self.features = features
        self.mask = mask

    def transform(self, X):
        """Extracts features from a cohort of Patient objects.

        Args:
            X: list of Patient objects

        Returns:
            df: DataFrame of features
        """
        dfs = []
        for i, p in enumerate(X):
            df = self.extract(p)
            df.insert(len(df.columns), 'Patient ID', i)
            dfs.append(df)

        return pd.concat(dfs)

    def extract(self, p):
        """Extracts features from a single Patient object.

        Args:
            p: Patient object

        Returns:
            df: DataFrame of features
        """
        feature_funcs, feature_eqns = [], []
        for name, f in self.features:
            target = feature_eqns if isinstance(f, str) else feature_funcs
            target.append((name, f))

        df = pd.DataFrame({name: func(p) for name, func in feature_funcs})
        for name, eqn in feature_eqns:
            df.eval('%s = %s' % (name, eqn), inplace=True)

        if self.mask:
            df = df.query(self.mask)

        return df


class VoxelEstimator(BaseEstimator):
    """Used to estimate a voxel-wise target (e.g. dose) based upon features of
    the voxel. Although the features and targets are voxel-wise, the input data
    remains patient-wise. It is important that this estimator maintains an
    interface where X is a list of Patient objects, for sub-sampling and
    sorting purposes. This is achieved using a VoxelFeatureExtractor class.
    """

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
