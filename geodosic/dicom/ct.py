# third-party imports
import numpy as np

# project imports
from geodosic.dicom.base import DicomBase


class DicomCt(DicomBase):
    """docstring for DicomCt"""
    def __init__(self, ds):
        super(DicomCt, self).__init__(ds)

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
        return float(dx), float(dy), float(dz)

    def HU_array(self):

        if 'RescaleSlope' in self.ds and 'RescaleIntercept' in self.ds:
            scale = self.ds.RescaleSlope
            offset = self.ds.RescaleIntercept
        else:
            scale = 1.0
            offset = 0.0

        distribution = np.transpose(self.ds.pixel_array)
        return scale * distribution + offset
