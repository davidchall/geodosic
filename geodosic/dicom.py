# system imports
import os
import re
import fnmatch
import logging

# third-party imports
from six import PY2
import numpy as np
import matplotlib.path

try:
    from pydicom.dicomio import read_file as pydicom_read_file
except ImportError:
    from dicom import read_file as pydicom_read_file

# project imports
from .geometry import cartesian_product, distance_to_surface
from .utils import lazy_property


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
                self.rtdose.append(RTDose(ds))
            elif ds.SOPClassUID == RTSTRUCT_UID:
                self.rtss.append(RTStruct(ds))
            elif ds.SOPClassUID == RTPLAN_UID:
                self.rtplan.append(RTPlan(ds))
            elif ds.SOPClassUID == CT_UID:
                self.ct.append(CT(ds))
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


def read_file(filename):
    ds = pydicom_read_file(filename, defer_size=100, force=True)

    if 'SOPClassUID' not in ds:
        raise IOError('Not a valid DICOM file: %s' % filename)

    if ds.SOPClassUID == RTDOSE_UID:
        return RTDose(ds)
    elif ds.SOPClassUID == RTPLAN_UID:
        return RTPlan(ds)
    elif ds.SOPClassUID == RTSTRUCT_UID:
        return RTStruct(ds)
    elif ds.SOPClassUID == CT_UID:
        return CT(ds)
    else:
        raise IOError('Unsupported DICOM SOP class: %s' % filename)


class DicomBase(object):
    """docstring for DicomBase"""

    def __init__(self, ds):
        self.ds = ds

    def GetPatientInfo(self):
        """Return the patient metadata."""

        # Set up some sensible defaults for demographics
        patient = {'name': 'N/A',
                   'id': 'N/A',
                   'birth_date': None,
                   'gender': 'N/A'}
        if 'PatientName' in self.ds:
            if PY2:
                self.ds.decode()
            name = self.ds.PatientName
            patient['name'] = name
            patient['given_name'] = name.given_name
            patient['middle_name'] = name.middle_name
            patient['family_name'] = name.family_name
        if 'PatientID' in self.ds:
            patient['id'] = self.ds.PatientID
        if 'PatientSex' in self.ds:
            if (self.ds.PatientSex == 'M'):
                patient['gender'] = 'M'
            elif (self.ds.PatientSex == 'F'):
                patient['gender'] = 'F'
            else:
                patient['gender'] = 'O'
        if 'PatientBirthDate' in self.ds:
            if len(self.ds.PatientBirthDate):
                patient['birth_date'] = str(self.ds.PatientBirthDate)

        return patient

    def GetStudyInfo(self):
        """Return the study information of the current file."""

        study = {}
        if 'StudyDescription' in self.ds:
            desc = self.ds.StudyDescription
        else:
            desc = 'No description'
        study['description'] = desc
        if 'StudyDate' in self.ds:
            date = self.ds.StudyDate
        else:
            date = None
        study['date'] = date
        # Don't assume that every dataset includes a study UID
        if 'StudyInstanceUID' in self.ds:
            study['id'] = self.ds.StudyInstanceUID
        else:
            study['id'] = str(random.randint(0, 65535))

        return study

    def GetSeriesInfo(self):
        """Return the series information of the current file."""

        series = {}
        if 'SeriesDescription' in self.ds:
            desc = self.ds.SeriesDescription
        else:
            desc = 'No description'
        series['description'] = desc
        series['id'] = self.ds.SeriesInstanceUID
        # Don't assume that every dataset includes a study UID
        series['study'] = self.ds.SeriesInstanceUID
        if 'StudyInstanceUID' in self.ds:
            series['study'] = self.ds.StudyInstanceUID
        series['referenceframe'] = self.ds.FrameOfReferenceUID \
            if 'FrameOfReferenceUID' in self.ds \
            else str(random.randint(0, 65535))
        if 'Modality' in self.ds:
            series['modality'] = self.ds.Modality
        else:
            series['modality'] = 'OT'

        return series


class RTDose(DicomBase):
    """docstring for RTDose"""

    def __init__(self, ds):
        super(RTDose, self).__init__(ds)

    def dose_name(self):
        return self.ds.SeriesDescription

    @property
    def shape(self):
        nx, ny = self.ds.Columns, self.ds.Rows
        nz = len(self.ds.GridFrameOffsetVector)
        return nx, ny, nz

    def grid_vectors(self):
        """Return coordinate vectors.

        Returns:
            x, y, z: numpy.ndarray
        """
        di = self.ds.PixelSpacing[0]
        dj = self.ds.PixelSpacing[1]
        cosi = self.ds.ImageOrientationPatient[:3]
        cosj = self.ds.ImageOrientationPatient[3:]
        position = self.ds.ImagePositionPatient

        x = position[0] + cosi[0] * di * np.arange(0, self.ds.Columns)
        y = position[1] + cosj[1] * dj * np.arange(0, self.ds.Rows)
        z = position[2] + cosi[0] * np.array(self.ds.GridFrameOffsetVector)

        return x, y, z

    def grid_spacing(self):
        """Return the smallest spacing present in the grid.

        Note that non-regular grids have variable spacing.

        Returns:
            dx, dy, dz: numpy.ndarray
        """
        dx, dy = self.ds.PixelSpacing
        dz = np.min(np.diff(self.ds.GridFrameOffsetVector))
        return dx, dy, dz

    def is_regular_grid(self):
        """Return whether grid has regular spacing."""
        return not any(np.diff(np.diff(self.ds.GridFrameOffsetVector)))

    def dose_array(self):
        scale = self.ds.DoseGridScaling
        distribution = np.transpose(self.ds.pixel_array, axes=(2, 1, 0))
        return scale * distribution


class RTStruct(DicomBase):
    """docstring for RTStruct"""

    def __init__(self, ds):
        super(RTStruct, self).__init__(ds)

        contour_keys = ('StructureSetROISequence', 'ROIContourSequence')
        if not all(k in self.ds for k in contour_keys):
            raise IOError('RTStruct file contains zero structures')

        self._roi_lookup = {}
        for s in self.ds.StructureSetROISequence:
            if s.ROIName in self._roi_lookup:
                logging.warning('Found multiple "%s" structures' % s.ROIName)
            else:
                self._roi_lookup[s.ROIName] = s.ROINumber

    def structure_names(self):
        return self._roi_lookup.keys()

    def is_empty_structure(self, name):
        """Return whether a given structure is empty."""

        for roi in self.ds.ROIContourSequence:
            if roi.ReferencedROINumber == self._roi_lookup[name]:
                return 'ContourSequence' not in roi

        return True

    def structure_mask(self, name, grid):
        """Compute a mask indicating the presence of a 3D structure
        upon a 3D grid.

        Parameters:
            name: structure name
            grid: 3-tuple of numpy.ndarray with shapes (m1,), (m2,) and (m3,)
                They are the x, y and z coordinate vectors.

        Returns:
            mask: numpy.ndarray of bool with shape (m1, m2, m3)
        """

        if self.is_empty_structure(name):
            return None

        for roi in self.ds.ROIContourSequence:
            if roi.ReferencedROINumber == self._roi_lookup[name]:
                contours = roi.ContourSequence

        # TODO: there may be multiple contours per z coordinate
        # so contour_points dict won't work in this case
        # would need to check if contours are inside others
        contour_points = {}
        z_contour = []
        for i, c in enumerate(contours):
            data = np.array(c.ContourData)
            z_contour.append(data[2])
            contour_points[i] = np.delete(data, np.arange(2, data.size, 3))
            contour_points[i] = np.reshape(contour_points[i], (-1,2))

        def get_polygon_mask(polygon_points, grid_points):
            """Compute a mask indicating the presence of a 2D polygon
            upon a 2D grid.

            Parameters:
                polygon_points: numpy.ndarray with shape (p, 2)
                grid_points: numpy.ndarray with shape (m, 2)

            Returns:
                mask: numpy.ndarray of bool with shape (m,)
            """
            # find bounding box for polygon
            bb_min_x, bb_min_y = np.amin(polygon_points, axis=0)
            bb_max_x, bb_max_y = np.amax(polygon_points, axis=0)

            bb_mask = (grid_points[:,0] > bb_min_x) & \
                      (grid_points[:,0] < bb_max_x) & \
                      (grid_points[:,1] > bb_min_y) & \
                      (grid_points[:,1] < bb_max_y)

            # check if points in bounding box are within polygon
            polygon = matplotlib.path.Path(polygon_points)
            bb_polygon_mask = polygon.contains_points(grid_points[bb_mask])

            # set the mask values within the bounding box on the grid
            polygon_mask = np.zeros_like(bb_mask, dtype=bool)
            polygon_mask[bb_mask] = bb_polygon_mask

            return polygon_mask

        x, y, z = grid
        points_2d = cartesian_product((x, y))
        mask_3d = np.zeros((x.size, y.size, z.size), dtype=bool)

        for i, z_i in enumerate(z):
            #  TODO: loosen this, to use contour in one slice either side
            if z_i < np.amin(z_contour) or z_i > np.amax(z_contour):
                continue

            closest_contour = np.argmin(np.fabs(z_contour - z_i))
            polygon_points = contour_points[closest_contour]
            polygon_mask = get_polygon_mask(polygon_points, points_2d)

            mask_3d[:,:,i] = polygon_mask.reshape((x.size, y.size))

        return mask_3d


class RTPlan(DicomBase):
    """docstring for RTPlan"""
    def __init__(self, ds):
        super(RTPlan, self).__init__(ds)


class CT(DicomBase):
    """docstring for CT"""
    def __init__(self, ds):
        super(CT, self).__init__(ds)

    def grid_vectors(self):
        """Return coordinate vectors.

        Returns:
            x, y, z: numpy.ndarray
        """
        di = self.ds.PixelSpacing[0]
        dj = self.ds.PixelSpacing[1]
        cosi = self.ds.ImageOrientationPatient[:3]
        cosj = self.ds.ImageOrientationPatient[3:]
        position = self.ds.ImagePositionPatient

        x = position[0] + cosi[0] * di * np.arange(0, self.ds.Columns)
        y = position[1] + cosj[1] * dj * np.arange(0, self.ds.Rows)

        if 'SliceLocation' in self.ds:
            z = self.ds.SliceLocation
        else:
            z = position[2]

        return x, y, z

    def grid_spacing(self):
        """Return the grid spacing (CT images have a single slice).

        Returns:
            dx, dy, dz: numpy.ndarray
        """
        dx, dy = self.ds.PixelSpacing
        dz = self.ds.SliceThickness
        return dx, dy, dz

    def HU_array(self):

        if 'RescaleSlope' in self.ds and 'RescaleIntercept' in self.ds:
            scale = self.ds.RescaleSlope
            offset = self.ds.RescaleIntercept
        else:
            scale = 1.0
            offset = 0.0

        distribution = np.transpose(self.ds.pixel_array)
        return scale * distribution + offset
