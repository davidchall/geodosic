# system imports
import os.path
from functools import wraps
import json

# third-party imports
import numpy as np
import h5py
import matplotlib.pyplot as plt

# project imports
from .dicom import DicomCollection
from .geometry import distance_to_surface, bwperim
from .utils import lazy_property
from .dvh import DVH


def persistent_result(func_key, i_keys):
    """Indicates that the object returned by the decorated function should be
    persistently stored in the patient's result_file.  If the result already
    exists in result_file then this stored version is returned instead, to
    save unnecessary computation.

    The decorator arguments determine how to build the key used to store the
    result. The func_key is used to identify the function used to produce the
    result (e.g. distance, structure_mask). The result_key_args identify which
    arguments are appended to the func_key (e.g. struct_name, grid_name). The
    result_key_args is either an int or a tuple of ints. The arguments it
    refers to must be strings.

        class Patient(object):
            @persistent_result('structure_mask', (1,2))
            def structure_mask(self, struct_name, grid_name):
                return self.dicom.structure_mask(struct_name, grid_name)
    """
    def persistent_result_decorator(func):
        @wraps(func)
        def func_wrapper(self, *args, **kwargs):

            if hasattr(i_keys, '__iter__'):
                i_arg_keys = list(i_keys)
            else:
                i_arg_keys = [i_keys]

            # if there is a missing arg, let the function complain
            if max(i_arg_keys) > len(args):
                return func(self, *args, **kwargs)

            result_key = ', '.join(args[i-1] for i in i_arg_keys)

            group = self.results.require_group(func_key)
            if result_key not in group:
                result = func(self, *args, **kwargs)
                group.create_dataset(result_key, data=result,
                    compression="gzip", compression_opts=9)

            return group[result_key][:]

        return func_wrapper
    return persistent_result_decorator


def translate_struct(i):
    def translate_decorator(func):
        @wraps(func)
        def func_wrapper(self, *args, **kwargs):
            if i > len(args):
                return func(self, *args, **kwargs)

            args = list(args)
            name = args[i-1]
            name = self.structure_aliases.get(name, name)
            args[i-1] = name
            return func(self, *args, **kwargs)
        return func_wrapper
    return translate_decorator


def translate_dose(i):
    def translate_decorator(func):
        @wraps(func)
        def func_wrapper(self, *args, **kwargs):
            if i > len(args):
                return func(self, *args, **kwargs)

            args = list(args)
            name = args[i-1]
            name = self.dose_aliases.get(name, name)
            args[i-1] = name
            return func(self, *args, **kwargs)
        return func_wrapper
    return translate_decorator


def translate_grid(i):
    def translate_decorator(func):
        @wraps(func)
        def func_wrapper(self, *args, **kwargs):
            if i > len(args):
                return func(self, *args, **kwargs)

            args = list(args)
            name = args[i-1]
            if name.lower() == 'default':
                name = self.dicom.default_grid_name
            elif name.lower() in ('ct', 'custom'):
                name = name.lower()
            else:
                name = self.dose_aliases.get(name, name)
            args[i-1] = name
            return func(self, *args, **kwargs)
        return func_wrapper
    return translate_decorator


class Patient(object):
    """This class provides a consistent interface to patient data. This is
    achieved via configuration files that link aliases for structures and
    dose arrays to the names actually used in the DICOM file (e.g. 'cochlea_L'
    and 'imrt' instead of 'Left Cochlea.DCH' and 'CTV 20Gy').

    It also persistently memoizes results that take some time to calculate
    from the DICOM files (e.g. structure masks and distance transforms), so
    that these can be loaded more quickly in future.

    There is a large overlap with the methods provided by the DicomCollection
    class, and this allows these two classes to be used interchangeably in
    plotting and estimator routines.
    """
    def __init__(self, dicom_dir):

        if not os.path.isdir(dicom_dir):
            raise ValueError('Input DICOM directory not found %s' % dicom_dir)
        self.dicom_dir = dicom_dir

        self.structure_aliases = {}
        self.dose_aliases = {}

        config_fname = os.path.join(dicom_dir, 'config.json')
        if os.path.isfile(config_fname):
            with open(config_fname, 'r') as f:
                config = json.load(f)
            if 'structure_aliases' in config:
                self.structure_aliases = config['structure_aliases']
            if 'dose_aliases' in config:
                self.dose_aliases = config['dose_aliases']

        result_fname = os.path.join(dicom_dir, 'intermediate-results.hdf5')
        self.results = h5py.File(result_fname, 'a')

    def close(self):
        self.results.close()

    @lazy_property
    def dicom(self):
        return DicomCollection(self.dicom_dir)

    @lazy_property
    def dose_names(self):
        return set(self.dose_aliases.keys()) | self.dicom.dose_names

    @lazy_property
    def structure_names(self):
        return set(self.structure_aliases.keys()) | self.dicom.structure_names

    @translate_struct(1)
    @translate_grid(2)
    @persistent_result('structure_mask', (1, 2))
    def structure_mask(self, struct_name, grid_name):
        return self.dicom.structure_mask(struct_name, grid_name)

    @translate_struct(1)
    @translate_grid(2)
    @persistent_result('distance', (1, 2))
    def distance_to_surface(self, struct_name, grid_name):
        x, y, z = self.dicom.grid_vectors(grid_name)
        ddz = np.diff(np.diff(z))
        if not np.allclose(ddz, np.zeros_like(ddz)):
            raise ValueError('Attempted to use non-regular grid "%s" to compute distance-to-surface' % grid_name)

        mask = self.structure_mask(struct_name, grid_name)
        grid_spacing = self.dicom.grid_spacing(grid_name)

        return distance_to_surface(mask, grid_spacing).astype(np.float16)

    @translate_dose(1)
    @translate_grid(2)
    @persistent_result('dose', (1, 2))
    def dose_array(self, dose_name, grid_name):
        return self.dicom.dose_array(dose_name, grid_name).astype(np.float32)

    @translate_grid(1)
    def grid_vectors(self, name):
        return self.dicom.grid_vectors(name)

    @translate_grid(1)
    def grid_spacing(self, name):
        return self.dicom.grid_spacing(name)

    def ct_array(self):
        return self.dicom.ct_array()

    def calculate_dvh(self, struct_name, dose_name, struct_normalize=None,
                      dose_edges=None):
        dose = self.dose_array(dose_name, dose_name)
        struct_mask = self.structure_mask(struct_name, dose_name)
        struct_dose = dose[struct_mask]

        if struct_normalize:
            norm_mask = self.structure_mask(struct_normalize, dose_name)
            struct_dose /= np.mean(dose[norm_mask])

        if dose_edges is None:
            eps = np.finfo(float).eps
            dose_edges, binwidth = np.linspace(0, (1+eps)*np.amax(struct_dose)+eps,
                                               200, retstep=True)
            dose_edges = np.append(dose_edges, (dose_edges[-1] + binwidth))

        return DVH.from_raw(struct_dose, dose_edges)

    def calculate_dsh(self, struct_name, dose_name, struct_normalize=None,
                      dose_edges=None):
        dose = self.dose_array(dose_name, dose_name)
        struct_mask = self.structure_mask(struct_name, dose_name)
        struct_surface = bwperim(struct_mask)

        if struct_normalize:
            norm_mask = self.structure_mask(struct_normalize, dose_name)
            dose /= np.mean(dose[norm_mask])

        surface_dose = dose[struct_surface]

        if dose_edges is None:
            eps = np.finfo(float).eps
            dose_edges, binwidth = np.linspace(0, (1+eps)*np.amax(surface_dose)+eps,
                                               200, retstep=True)
            dose_edges = np.append(dose_edges, (dose_edges[-1] + binwidth))

        return DVH.from_raw(surface_dose, dose_edges)

    def calculate_ovh(self, struct_name, target_name, grid_name,
                      dist_edges=None):
        dist = self.distance_to_surface(target_name, grid_name)
        struct_mask = self.structure_mask(struct_name, grid_name)
        struct_dist = dist[struct_mask]

        if dist_edges is None:
            eps = np.finfo(float).eps
            dist_edges, binwidth = np.linspace((1-eps)*np.amin(struct_dist)-eps,
                                               (1+eps)*np.amax(struct_dist)+eps,
                                               200, retstep=True)
            dist_edges = np.append(dist_edges, (dist_edges[-1] + binwidth))

        return DVH.from_raw(struct_dist, dist_edges, skip_checks=True)

    def plot_dose_vs_distance(self, dose_name, struct_name, target_name):
        dose = self.dose_array(dose_name, dose_name)
        dist = self.distance_to_surface(target_name, dose_name)
        struct_mask = self.structure_mask(struct_name, dose_name)

        struct_dose = dose[struct_mask]
        struct_dist = dist[struct_mask]

        dose_bins = np.arange(0, np.amax(struct_dose), 1)
        dist_bins = np.arange(np.amin(struct_dist), np.amax(struct_dist), 1)

        plt.hist2d(struct_dist, struct_dose, [dist_bins, dose_bins])
        plt.xlabel('Distance-to-target [mm]')
        plt.ylabel('Dose [Gy]')
