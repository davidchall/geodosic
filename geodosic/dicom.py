# third-party imports
from six import PY2
import numpy as np
import matplotlib.path

try:
    from pydicom.dicomio import read_file as pydicom_read_file
except ImportError:
    from dicom import read_file as pydicom_read_file

from .geometry import cartesian_product


CT_UID = '1.2.840.10008.5.1.4.1.1.2'
RTDOSE_UID = '1.2.840.10008.5.1.4.1.1.481.2'
RTPLAN_UID = '1.2.840.10008.5.1.4.1.1.481.5'
RTSTRUCT_UID = '1.2.840.10008.5.1.4.1.1.481.3'


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


class DicomImage(DicomBase):

    def __init__(self, ds):
        super(DicomImage, self).__init__(ds)

    def GetImageInfo(self):
        """Return the image metadata."""

        data = {}

        if 'ImagePositionPatient' in self.ds:
            data['position'] = self.ds.ImagePositionPatient
        if 'ImageOrientationPatient' in self.ds:
            data['orientation'] = self.ds.ImageOrientationPatient
        if 'PixelSpacing' in self.ds:
            data['pixelspacing'] = self.ds.PixelSpacing
        else:
            data['pixelspacing'] = [1, 1]
        data['rows'] = self.ds.Rows
        data['columns'] = self.ds.Columns
        data['samplesperpixel'] = self.ds.SamplesPerPixel
        data['photometricinterpretation'] = self.ds.PhotometricInterpretation
        data['littlendian'] = \
            self.ds.file_meta.TransferSyntaxUID.is_little_endian
        if 'PatientPosition' in self.ds:
            data['patientposition'] = self.ds.PatientPosition
        data['frames'] = self.GetNumberOfFrames()

        return data

    def GetNumberOfFrames(self):
        """Return the number of frames in a DICOM image file."""

        frames = 1
        if 'NumberOfFrames' in self.ds:
            frames = self.ds.NumberOfFrames.real
        else:
            try:
                self.ds.pixel_array
            except:
                return 0
            else:
                if (self.ds.pixel_array.ndim > 2):
                    if (self.ds.SamplesPerPixel == 1) and not \
                       (self.ds.PhotometricInterpretation == 'RGB'):
                        frames = self.ds.pixel_array.shape[0]
        return frames

    def GetXYZ(self):
        """Return coordinate vectors.

        Returns:
            x, y, z: numpy.ndarray
        """

        #  transformation taken from http://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.2.html#sect_C.7.6.2.1.1

        di = self.ds.PixelSpacing[0]
        dj = self.ds.PixelSpacing[1]
        cosi = self.ds.ImageOrientationPatient[:3]
        cosj = self.ds.ImageOrientationPatient[3:]
        position = self.ds.ImagePositionPatient

        x = position[0] + cosi[0] * di * np.arange(0, self.ds.Columns)
        y = position[1] + cosj[1] * dj * np.arange(0, self.ds.Rows)
        z = position[2] + cosi[0] * np.array(self.ds.GridFrameOffsetVector)

        return x, y, z


class RTDose(DicomImage):
    """docstring for RTDose"""
    def __init__(self, ds):
        super(RTDose, self).__init__(ds)

    def GetAspectRatio(self):
        """Returns the aspect ratio of the image.
        """
        if 'PixelSpacing' in self.ds:
            x, y = self.ds.PixelSpacing
        else:
            x, y = 1, 1

        return x, y

    def IsRegularGrid(self):
        """Return whether grid has regular spacing."""
        return not any(np.diff(np.diff(self.ds.GridFrameOffsetVector)))

    def GetDoseArray(self):
        scale = self.ds.DoseGridScaling
        distribution = np.transpose(self.ds.pixel_array, axes=(2, 1, 0))
        return scale * distribution


class RTStruct(DicomBase):
    """docstring for RTStruct"""
    def __init__(self, ds):
        super(RTStruct, self).__init__(ds)

        if 'StructureSetROISequence' in self.ds:
            structs = self.ds.StructureSetROISequence
            self._name_table = dict((s.ROIName, s.ROINumber) for s in structs)

    def GetAvailableStructureNames(self):
        """Return a list of available structure names."""

        return self._name_table.keys()

    def IsEmptyStructure(self, name):
        """Return whether a given structure is empty."""

        if 'ROIContourSequence' in self.ds:
            for roi in self.ds.ROIContourSequence:
                if roi.ReferencedROINumber == self._name_table[name]:
                    return 'ContourSequence' not in roi

        return True

    def GetStructureMask(self, name, grid):
        """Compute a mask indicating the presence of a structure upon a grid.

        Parameters:
            name: structure name
            grid: 3-tuple of numpy.ndarray with shapes (m1,), (m2,) and (m3,)
                They are the x, y and z coordinate vectors.

        Returns:
            mask: numpy.ndarray of bool with shape (m1, m2, m3)
        """

        if self.IsEmptyStructure(name):
            return None

        for roi in self.ds.ROIContourSequence:
            if roi.ReferencedROINumber == self._name_table[name]:
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


        x, y, z = grid
        points_2d = cartesian_product((x, y))
        mask_3d = np.empty((x.size, y.size, z.size), dtype=bool)

        for i, z_i in enumerate(z):
            #  TODO: loosen this, to use contour in one slice either side
            if z_i < np.amin(z_contour) or z_i > np.amax(z_contour):
                continue

            closest_contour = np.argmin(np.fabs(z_contour - z_i))
            polygon_points = contour_points[closest_contour]

            # find bounding box for polygon
            bb_min_x, bb_min_y = np.amin(polygon_points, axis=0)
            bb_max_x, bb_max_y = np.amax(polygon_points, axis=0)

            bb_mask_flat = (points_2d[:,0] > bb_min_x) & \
                           (points_2d[:,0] < bb_max_x) & \
                           (points_2d[:,1] > bb_min_y) & \
                           (points_2d[:,1] < bb_max_y)

            bb_mask = bb_mask_flat.reshape((x.size, y.size))

            # check if points in bounding box are within polygon
            polygon = matplotlib.path.Path(polygon_points)
            mask_2d = polygon.contains_points(points_2d[bb_mask_flat])

            mask_3d[bb_mask,i] = mask_2d

        return mask_3d


class RTPlan(DicomBase):
    """docstring for RTPlan"""
    def __init__(self, ds):
        super(RTPlan, self).__init__(ds)


class CT(DicomBase):
    """docstring for CT"""
    def __init__(self, ds):
        super(CT, self).__init__(ds)
