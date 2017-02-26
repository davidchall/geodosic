# third-party imports
import numpy as np

# project imports
from ..geometry import centroid


class BasePatientFeature(object):
    def __call__(self, p):
        return np.full(p.grid_shape(self.grid_name), self.extract(p)).flatten()


class BaseVoxelFeature(object):
    def __call__(self, p):
        return self.extract(p).flatten()


class Dose(BaseVoxelFeature):

    def __init__(self, grid_name, dose_name):
        self.grid_name = grid_name
        self.dose_name = dose_name

    def extract(self, p):
        return p.dose_array(self.dose_name, self.grid_name)


class StructureMask(BaseVoxelFeature):

    def __init__(self, grid_name, struct_name):
        self.grid_name = grid_name
        self.struct_name = struct_name

    def extract(self, p):
        return p.structure_mask(self.struct_name, self.grid_name)


class MinDistanceToStructure(BaseVoxelFeature):

    def __init__(self, grid_name, struct_name):
        self.grid_name = grid_name
        self.struct_name = struct_name

    def extract(self, p):
        if isinstance(self.struct_name, str):
            return p.distance_to_surface(self.struct_name, self.grid_name)
        else:
            return np.minimum.reduce([p.distance_to_surface(s, self.grid_name) for s in self.struct_name])


class PolarAngle(BaseVoxelFeature):

    def __init__(self, grid_name, struct_name=None):
        self.grid_name = grid_name
        self.struct_name = struct_name

    def extract(self, p):
        grid = p.grid_vectors(self.grid_name)
        x, y, z = np.meshgrid(*grid, indexing='ij')

        if self.struct_name:
            struct_mask = p.structure_mask(self.struct_name, self.grid_name)
            x0, y0, z0 = centroid(struct_mask, grid)
            x -= x0
            y -= y0
            z -= z0

        r = np.sqrt(z**2 + y**2 + z**2)
        return np.arccos(z / r)


class AzimuthalAngle(BaseVoxelFeature):

    def __init__(self, grid_name, struct_name=None):
        self.grid_name = grid_name
        self.struct_name = struct_name

    def extract(self, p):
        grid = p.grid_vectors(self.grid_name)
        x, y, z = np.meshgrid(*grid, indexing='ij')

        if self.struct_name:
            struct_mask = p.structure_mask(self.struct_name, self.grid_name)
            x0, y0, z0 = centroid(struct_mask, grid)
            x -= x0
            y -= y0
            z -= z0

        return np.arctan2(y, x)


class StructureVolume(BasePatientFeature):

    def __init__(self, grid_name, struct_name):
        self.grid_name = grid_name
        self.struct_name = struct_name

    def extract(self, p):
        return p.structure_volume(self.struct_name)
