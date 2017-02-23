# third-party imports
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# project imports
from ..utils import initialize_attributes


class PatientFeature(BaseEstimator, TransformerMixin):

    def transform(self, X):
        # flatten before concatenating since grids have inconsistent shape
        features = [np.full(p.grid_shape(self.grid_name), self.extract(p)).flatten() for p in X]
        features = np.concatenate(features, axis=0)

        # FeatureUnion requires 2D array
        return features[:, np.newaxis]


class VoxelFeature(BaseEstimator, TransformerMixin):

    def transform(self, X):
        # flatten before concatenating since grids have inconsistent shape
        features = [self.extract(p).flatten() for p in X]
        features = np.concatenate(features, axis=0)

        # FeatureUnion requires 2D array
        return features[:, np.newaxis]


class Dose(VoxelFeature):

    @initialize_attributes
    def __init__(self, grid_name=None, dose_name=None):
        pass

    def fit(self, X):
        assert self.grid_name is not None
        assert self.dose_name is not None

    def extract(self, p):
        return p.dose_array(self.dose_name, self.grid_name)


class MinDistanceToStructure(VoxelFeature):

    @initialize_attributes
    def __init__(self, grid_name=None, struct_name=None):
        pass

    def fit(self, X):
        assert self.grid_name is not None
        assert self.struct_name is not None

    def extract(self, p):
        if isinstance(self.struct_name, str):
            return p.distance_to_surface(self.struct_name, self.grid_name)
        else:
            return np.minimum.reduce([p.distance_to_surface(s, self.grid_name) for s in self.struct_name])


class StructureVolume(PatientFeature):

    @initialize_attributes
    def __init__(self, grid_name=None, struct_name=None):
        pass

    def fit(self, X):
        assert self.grid_name is not None
        assert self.struct_name is not None

    def extract(self, p):
        return p.structure_volume(self.struct_name)
