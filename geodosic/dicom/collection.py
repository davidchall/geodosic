# system imports
import os
import re
import fnmatch
import logging

# third-party imports
import numpy as np
try:
    from pydicom.dicomio import read_file as pydicom_read_file
except ImportError:
    from dicom import read_file as pydicom_read_file

# project imports
from geodosic.utils import lazy_property
from geodosic.geometry import distance_to_surface

from geodosic.dicom.rtdose import DicomRtDose
from geodosic.dicom.rtstruct import DicomRtStruct
from geodosic.dicom.rtplan import DicomRtPlan
from geodosic.dicom.ct import DicomCt


CT_UID = '1.2.840.10008.5.1.4.1.1.2'
RTDOSE_UID = '1.2.840.10008.5.1.4.1.1.481.2'
RTPLAN_UID = '1.2.840.10008.5.1.4.1.1.481.5'
RTSTRUCT_UID = '1.2.840.10008.5.1.4.1.1.481.3'


class DicomCollection(object):
    """A class that provides an interface to a collection of DICOM files,
    representing a single patient.
    """

    def __init__(self, paths, skip_check=False):
        """Constructor for DicomCollection.

        Parameters:
            paths: list of paths to DICOM files and/or directories.
                Directories will be recursively searched for DICOM files.
            skip_check: disable validation that files are from a single study
        """

        dicom_files = [paths] if isinstance(paths, str) else paths

        # build regex that case-insensitively matches file extensions
        dicom_ext = ['dcm', 'dicom']
        regex = '|'.join(fnmatch.translate('*.'+ext) for ext in dicom_ext)
        reobj = re.compile('('+regex+')', re.IGNORECASE)

        # replace directories their files
        dicom_dirs = [p for p in dicom_files if os.path.isdir(p)]
        for p in dicom_dirs:
            dicom_files.remove(p)
            for root, dirs, files in os.walk(p, topdown=True):
                dicom_files += [os.path.join(root, f) for f in files
                                if re.match(reobj, f)]

        # remove any duplicates
        dicom_files = list(set(dicom_files))

        # read DICOM files
        self.rtdose, self.rtss, self.rtplan, self.ct = [], [], [], []
        for fname in dicom_files:
            ds = pydicom_read_file(fname, defer_size=100, force=True)

            if 'SOPClassUID' not in ds:
                logging.warning('Not a valid DICOM file: %s' % fname)
                continue

            if ds.SOPClassUID == RTDOSE_UID:
                self.rtdose.append(DicomRtDose(ds))
            elif ds.SOPClassUID == RTSTRUCT_UID:
                self.rtss.append(DicomRtStruct(ds))
            elif ds.SOPClassUID == RTPLAN_UID:
                self.rtplan.append(DicomRtPlan(ds))
            elif ds.SOPClassUID == CT_UID:
                self.ct.append(DicomCt(ds))
            else:
                logging.warning('Unsupported DICOM SOP class: %s' % fname)

        # sort CT files by z value
        self.ct = sorted(self.ct, key=lambda x: x.grid_vectors()[2])

        # TODO: validate files are from the same study
        if not skip_check:
            pass

    @lazy_property
    def structure_names(self):
        """Set of available structures."""
        return set(x for ss in self.rtss for x in ss.structure_names())

    @lazy_property
    def dose_names(self):
        """Set of available doses."""
        return set(dose.dose_name() for dose in self.rtdose)

    @lazy_property
    def default_grid_name(self):
        if len(self.rtdose) > 0:
            return self.rtdose[0].dose_name()
        elif len(self.ct) > 0:
            return 'ct'
        else:
            return 'custom'

    def structure_mask(self, struct_name, grid_name):
        index = self._find_structure(struct_name)
        grid = self.grid_vectors(grid_name)
        if index is None:
            return None
        else:
            return self.rtss[index].structure_mask(struct_name, grid)

    def distance_to_surface(self, struct_name, grid_name):
        mask = self.structure_mask(struct_name, grid_name)
        grid_spacing = self.grid_spacing(grid_name)

        return distance_to_surface(mask, grid_spacing)

    def grid_vectors(self, grid_name):
        if grid_name.lower() == 'default':
            grid_name = self.default_grid_name
        if grid_name.lower() == 'ct':
            return self.ct_grid_vectors()
        if grid_name.lower() == 'custom':
            return self.custom_grid_vectors()
        else:
            return self.dose_grid_vectors(grid_name)

    def grid_spacing(self, grid_name):
        if grid_name.lower() == 'default':
            grid_name = self.default_grid_name
        if grid_name.lower() == 'ct':
            return self.ct_grid_spacing()
        if grid_name.lower() == 'custom':
            return self.custom_grid_spacing()
        else:
            return self.dose_grid_spacing(grid_name)

    def dose_grid_vectors(self, name):
        index = self._find_dose(name)
        if index is None:
            return None
        else:
            return self.rtdose[index].grid_vectors()

    def dose_grid_spacing(self, name):
        index = self._find_dose(name)
        if index is None:
            return None
        else:
            return self.rtdose[index].grid_spacing()

    def dose_array(self, name):
        index = self._find_dose(name)
        if index is None:
            return None
        else:
            return self.rtdose[index].dose_array()

    def ct_grid_vectors(self):
        z_tot = np.array([])
        x_ref, y_ref, z_ref = self.ct[0].grid_vectors()
        for ct in self.ct:
            x, y, z = ct.grid_vectors()
            if not np.allclose(x, x_ref) or not np.allclose(y, y_ref):
                logging.error('CT pixel locations vary between slices')
            z_tot = np.append(z_tot, z)

        return x_ref, y_ref, np.sort(z_tot)

    def ct_grid_spacing(self):
        dz_min = np.inf
        dx_ref, dy_ref, dz_ref = self.ct[0].grid_spacing()
        for ct in self.ct:
            dx, dy, dz = ct.grid_spacing()
            if not np.isclose(dx, dx_ref) or not np.isclose(dy, dy_ref):
                logging.error('CT pixel spacing varies between slices')
            dz_min = min(dz_min, dz)

        return dx_ref, dy_ref, dz_min

    def ct_array(self):
        return np.dstack(ct.HU_array() for ct in self.ct)

    def custom_grid_vectors(self):
        raise NotImplementedError

    def custom_grid_spacing(self):
        raise NotImplementedError

    def _find_structure(self, name):
        """Returns index for RTSS file containing the desired structure.
        """
        found = [name in ss.structure_names() for ss in self.rtss]

        if sum(found) == 0:
            logging.error('Unable to find structure "%s"' % name)
            return None
        if sum(found) >= 2:
            logging.warning('Found multiple "%s" structures' % name)

        first_found = next((i for i, x in enumerate(found) if x), None)
        return first_found

    def _find_dose(self, name):
        """Returns index for RTDOSE file containing the desired dose.
        """
        found = [name == dose.dose_name() for dose in self.rtdose]

        if sum(found) == 0:
            logging.error('Unable to find dose "%s"' % name)
            return None
        if sum(found) >= 2:
            logging.warning('Found multiple "%s" doses' % name)

        first_found = next((i for i, x in enumerate(found) if x), None)
        return first_found
