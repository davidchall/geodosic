# system imports
import os.path
from functools import wraps

# third-party imports
import numpy as np

# project imports
from .dicom import DicomCollection
from .geometry import distance_to_surface
from .utils import lazy_property


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

        This is used like so:
            p = Patient(...)
            mask = p.get_structure_mask('bladder', 'ct')

        which would produce a result file like:
        {
            "structure_mask, bladder, ct": mask_array
        }
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

            result_key = func_key + ', '
            result_key += ', '.join(args[i-1] for i in i_arg_keys)

            if result_key not in self._results:
                self._results[result_key] = func(self, *args, **kwargs)

            return self._results[result_key]

        return func_wrapper
    return persistent_result_decorator


def translate_struct(i):
    def translate_decorator(func):
        @wraps(func)
        def func_wrapper(self, *args, **kwargs):
            args = list(args)
            name = args[i-1]
            name = self.structure_aliases.get(name, name)
            args[i-1] = name
            return func(self, *args, **kwargs)
        return func_wrapper
    return translate_decorator


def translate_grid(i):
    def translate_decorator(func):
        @wraps(func)
        def func_wrapper(self, *args, **kwargs):
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


def translate_dose(i):
    def translate_decorator(func):
        @wraps(func)
        def func_wrapper(self, *args, **kwargs):
            args = list(args)
            name = args[i-1]
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
    def __init__(self, dicom_dir, structure_aliases={}, dose_aliases={},
                 result_file=None):

        self.dicom_dir = dicom_dir
        self.result_file = result_file or os.path.join(dicom_dir, 'results.npz')

        self.structure_aliases = structure_aliases
        self.dose_aliases = dose_aliases

        self._results = {}
        if os.path.isfile(self.result_file):
            with np.load(self.result_file) as f:
                for k,v in f.items():
                    if 'structure' in k:
                        self._results[k] = np.unpackbits(v, axis=0).astype(bool)
                    else:
                        self._results[k] = v

    def write(self):
        for k,v in self._results.items():
            if v.dtype == bool:
                self._results[k] = np.packbits(v, axis=0)
        np.savez_compressed(self.result_file, **self._results)

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
    @persistent_result('structure_mask', (1,2))
    def structure_mask(self, struct_name, grid_name):
        return self.dicom.structure_mask(struct_name, grid_name)

    @translate_struct(1)
    @translate_grid(2)
    @persistent_result('distance', (1,2))
    def distance_to_surface(self, struct_name, grid_name):
        mask = self.structure_mask(struct_name, grid_name)
        grid_spacing = self.dicom.grid_spacing(grid_name)

        return distance_to_surface(mask, grid_spacing).astype(np.float16)

    @translate_grid(1)
    def grid_vectors(self, name):
        return self.dicom.grid_vectors(name)

    @translate_grid(1)
    def grid_spacing(self, name):
        return self.dicom.grid_spacing(name)

    @translate_dose(1)
    def dose_array(self, name):
        return self.dicom.dose_array(name)

    def ct_array(self):
        return self.dicom.ct_array()
