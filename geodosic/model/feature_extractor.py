# third-party imports
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# project imports
from ..utils import initialize_attributes


class PatientWiseFeatureExtractor(BaseEstimator, TransformerMixin):

    def transform(self, X):
        features = [self.extract(p) * np.ones(p.grid_shape(self.grid_name)).flatten() for p in X]
        return np.concatenate(features, axis=0)[:, np.newaxis]

    def extract(self, p):
        raise NotImplementedError


class VoxelWiseFeatureExtractor(BaseEstimator, TransformerMixin):

    def transform(self, X):
        features = [self.extract(p).flatten() for p in X]
        return np.concatenate(features, axis=0)[:, np.newaxis]

    def extract(self, p):
        raise NotImplementedError


class Dose(VoxelWiseFeatureExtractor):

    @initialize_attributes
    def __init__(self, grid_name=None, dose_name=None):
        pass

    def fit(self, X):
        assert self.grid_name is not None
        assert self.dose_name is not None

    def extract(self, p):
        return p.dose_array(self.dose_name, self.grid_name)


class MinDistanceToStructure(VoxelWiseFeatureExtractor):

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


class StructureVolume(PatientWiseFeatureExtractor):

    @initialize_attributes
    def __init__(self, grid_name=None, struct_name=None):
        pass

    def fit(self, X):
        assert self.grid_name is not None
        assert self.struct_name is not None

    def extract(self, p):
        return p.structure_volume(self.struct_name)
