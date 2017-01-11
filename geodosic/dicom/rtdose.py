# third-party imports
import numpy as np

# project imports
from geodosic.dicom.base import DicomBase


class DicomRtDose(DicomBase):
    """docstring for DicomRtDose"""

    def __init__(self, ds):
        super(DicomRtDose, self).__init__(ds)

    def dose_name(self):
        if 'SeriesDescription' not in self.ds:
            return 'N/A'
        elif self.ds.SeriesDescription == '':
            return 'N/A'
        else:
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
