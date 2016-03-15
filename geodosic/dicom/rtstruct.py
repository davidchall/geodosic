# system imports
import logging

# third-party imports
import numpy as np
from matplotlib.path import Path

# project imports
from geodosic.dicom.base import DicomBase
from geodosic.geometry import cartesian_product


class DicomRtStruct(DicomBase):
    """docstring for DicomRtStruct"""

    def __init__(self, ds):
        super(DicomRtStruct, self).__init__(ds)

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
            polygon = Path(polygon_points)
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
