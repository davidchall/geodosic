# system imports
import json
import os.path
from functools import wraps

# third-party imports
from msgpack import load, dump
import msgpack_numpy as m

# project imports
from .dicom import DicomCollection
from .geometry import distance_to_surface

m.patch()
_missing = object()


class lazy_property(object):
    """A decorator that converts a function into a lazy property.  The
    function wrapped is called the first time to retrieve the result and then
    that calculated result is used the next time you access the value::

        class Foo(object):

            @lazy_property
            def foo(self):
                # calculate something important here
                return 42

    The class has to have a `__dict__` in order for this property to work.
    """
    # http://stackoverflow.com/a/17487613/2669425
    def __init__(self, func, name=None, doc=None):
        self.__name__ = name or func.__name__
        self.__module__ = func.__module__
        self.__doc__ = doc or func.__doc__
        self.func = func

    def __get__(self, obj, type=None):
        if obj is None:
            return self
        value = obj.__dict__.get(self.__name__, _missing)
        if value is _missing:
            value = self.func(obj)
            obj.__dict__[self.__name__] = value
        return value


def persistent_result(subdict_key):
    """Indicates that the object returned by the decorated function should be
    persistently stored in config['result_file'].  If the result already exists
    in config['result_file'] then this version is returned instead, to save
    unnecessary computation.

        class Patient(object):

            @persistent_result('structures')
            def get_structure_mask(self, name, grid):
                roi_name = self.config.find_roi_name(name)
                return self.dicom.calculate_structure_mask(roi_name, grid)

        This is used like so:
            p = Patient(cfg_file)
            mask = p.get_structure_mask('bladder', grid)

        which would produce a result file like:
            {
                "structures": {
                    "bladder": mask_array
                }
            }

        Note that the sub-dictionary key is specified in the decorator, and
        the actual result key is specified as the first argument to the method.
        In this instance "bladder_DCH" is the structure name used in the DICOM
        file, not the key used in the result file. This is found from the
        patient config file.
    """
    def persistent_result_decorator(func):
        @wraps(func)
        def func_wrapper(self, result_key, *args, **kwargs):

            subdict = self.results.setdefault(subdict_key, {})
            if result_key not in subdict:
                subdict[result_key] = func(self, result_key, *args, **kwargs)
            return subdict[result_key]

        return func_wrapper
    return persistent_result_decorator


class Patient(object):
    """docstring for Patient"""
    def __init__(self, config_path):
        super(Patient, self).__init__()

        with open(config_path) as f:
            self.config = json.load(f)

        if os.path.isfile(self.config['result_file']):
            with open(self.config['result_file'], 'rb') as f:
                self.results = load(f, encoding='utf-8')
        else:
            self.results = {}

    @lazy_property
    def dicom(self):
        return DicomCollection(self.config['dicom_dir'])

    @persistent_result('structure_masks')
    def structure_mask(self, name, grid):
        roi_name = self.config['structure'][name]
        return self.dicom.structure_mask(roi_name, grid)

    @persistent_result('distances')
    def distance(self, name, grid, grid_spacing):
        mask = self.structure_mask(name, grid)
        return distance_to_surface(mask, grid_spacing)

    def write(self):
        with open(self.config['result_file'], 'wb') as f:
            dump(self.results, f, use_bin_type=True)
