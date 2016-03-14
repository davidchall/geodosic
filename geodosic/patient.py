# system imports
import os.path
from functools import wraps

# third-party imports
import msgpack
import msgpack_numpy as m

# project imports
from .dicom import DicomCollection
from .geometry import distance_to_surface
from .utils import lazy_property

m.patch()


def persistent_result(group_key, result_key_args):
    """Indicates that the object returned by the decorated function should be
    persistently stored in result_file.  If the result already exists in
    result_file then this version is returned instead, to save unnecessary
    computation.

        class Patient(object):

            @persistent_result('structure_masks', (0,1))
            def structure_mask(self, struct_name, grid_name):
                return self.dicom.structure_mask(struct_name, grid_name)

        This is used like so:
            p = Patient(...)
            mask = p.get_structure_mask('bladder', 'ct')

        which would produce a result file like:
            {
                "structure_masks": {
                    ("bladder", "ct"): mask_array
                }
            }

    Note that the sub-dictionary key is specified in the decorator, and the
    actual result key is obtained from the method arguments, as specified in
    the decorator. In this instance "bladder_DCH" is the structure name used
    in the DICOM file, not the key used in the result file. This is found
    from the config file.
    """
    def persistent_result_decorator(func):
        @wraps(func)
        def func_wrapper(self, *args, **kwargs):

            if hasattr(result_key_args, '__iter__'):
                result_key = tuple(args[i-1] for i in result_key_args)
            else:
                result_key = args[result_key_args-1]

            group = self._results.setdefault(group_key, {})
            if result_key not in group:
                group[result_key] = func(self, *args, **kwargs)
            return group[result_key]

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
        self.result_file = result_file or os.path.join(dicom_dir, 'results.msgpk')

        self.structure_aliases = structure_aliases
        self.dose_aliases = dose_aliases

        if os.path.isfile(self.result_file):
            with open(self.result_file, 'rb') as f:
                self._results = msgpack.load(f, use_list=False)
        else:
            self._results = {}

    def write(self):
        with open(self.result_file, 'wb') as f:
            msgpack.dump(self._results, f)

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
    @persistent_result('structure_masks', (1,2))
    def structure_mask(self, struct_name, grid_name):
        return self.dicom.structure_mask(struct_name, grid_name)

    @translate_struct(1)
    @translate_grid(2)
    @persistent_result('distances', (1,2))
    def distance_to_surface(self, struct_name, grid_name):
        mask = self.structure_mask(struct_name, grid_name)
        grid_spacing = self.dicom.grid_spacing(grid_name)

        return distance_to_surface(mask, grid_spacing)

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
