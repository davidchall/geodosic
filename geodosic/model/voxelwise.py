# third-party imports
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone

# project imports
from ..dvh import DVH
from .features import StructureMask


class VoxelFeatureExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, features, grid_name, mask=None):
        """Transformer that extracts voxel-wise features from a cohort of
        Patient objects and returns a pandas.DataFrame object. Each row
        corresponds to a voxel (patients are tracked by 'Patient ID' feature).
        Voxel selection criteria are supported by the mask argument.

        Example:
            VoxelFeatureExtractor([
                ('dist_target', MinDistanceToStructure(target_name)),
                ('dist_target2', 'dist_target**2')
            ],
                grid_name=dose_name,
                mask='0 < dist_target < 10'
            )

        Args:
            features: list of 2-tuples like (feature_name, feature_func) where
                feature_func is either:
                    - function called on each Patient object
                    - expression using other feature_name variables

            grid_name: grid upon which voxel features are extracted
            mask: Boolean expression using feature_name variables

        Note: expressions using feature_name variables use pandas.eval()
        http://pandas.pydata.org/pandas-docs/stable/enhancingperf.html#expression-evaluation-via-eval-experimental
        """
        self.features = features
        self.grid_name = grid_name
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

        df = pd.DataFrame({name: func(p, self.grid_name) for name, func in feature_funcs})
        for name, eqn in feature_eqns:
            df.eval('%s = %s' % (name, eqn), inplace=True)

        if self.mask:
            df = df.query(self.mask)

        return df

    def clone_with_structure_mask(self, struct_name, keep_train_mask=False):
        other = clone(self)

        mask_name = 'temporary_mask'
        other.features.append((mask_name, StructureMask(struct_name)))

        if keep_train_mask:
            other.mask = self.mask + ' and ' + mask_name
        else:
            other.mask = mask_name

        return other


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

    def predict(self, X, struct_name=None, keep_train_mask=False):
        if struct_name:
            extractor = self.extractor.clone_with_structure_mask(struct_name, keep_train_mask)
        else:
            extractor = self.extractor

        df = extractor.transform(X)
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

    def generate_validation_dvhs(self, X, oar_name, dose_name, n_dose_bins=100):
        """Generates predicted and planned DVHs for model validation.

        Args:
            X: validation cohort of Patient objects
            n_dose_bins: resolution of DVH
        """
        for p in X:
            if dose_name not in p.dose_names:
                continue
            if oar_name not in p.structure_names:
                continue

            # choose appropriate binning for DVHs
            dose_plan = p.dose_array(dose_name, dose_name)
            oar_mask = p.structure_mask(oar_name, dose_name)
            dose_plan = dose_plan[oar_mask]

            dose_pred = self.predict([p], oar_name)

            max_dose_dvh = max(dose_plan.max(), dose_pred.max())
            dose_edges = DVH.choose_dose_edges(max_dose_dvh, n_dose_bins)

            dvh_pred = DVH.from_raw(dose_pred, dose_edges=dose_edges)
            dvh_plan = DVH.from_raw(dose_plan, dose_edges=dose_edges)

            yield p, dvh_pred, dvh_plan

    def generate_validation_dose_arrays(self, X):
        for p in X:
            df = self.extractor.transform([p])
            Xt = df[self.features]
            yt = df[self.target]

            grid = p.grid_vectors(self.extractor.grid_name)
            grid_shape = p.grid_shape(self.extractor.grid_name)

            dose_pred = np.zeros(grid_shape)
            dose_pred.put(df.index.values, self.estimator.predict(Xt))
            dose_pred = dose_pred.clip(min=0)

            dose_plan = np.zeros(grid_shape)
            dose_plan.put(df.index.values, yt)

            yield p, grid, dose_pred, dose_plan
