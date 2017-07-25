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
                all_contours = roi.ContourSequence

        # planes is a dict, where the key is the z-coordinate and the value is
        # a list of contours in that plane (each contour is a list of points)
        planes = {}
        for c in all_contours:
            contour_points_3d = np.array(c.ContourData).reshape((-1,3))
            contour_points_2d = contour_points_3d[:,0:2]
            contour_z = contour_points_3d[0,2]

            # round the float to use as a key
            contour_z = round(contour_z, 4)
            contours_list = planes.setdefault(contour_z, [])
            contours_list.append(contour_points_2d)

        x, y, z = grid
        points_2d = cartesian_product((x, y))
        mask_3d = np.zeros((x.size, y.size, z.size), dtype=bool)

        # find grid slices nearest to first and last structure planes
        z_planes = np.array(sorted(list(planes.keys())))
        min_z = z[np.fabs(z - np.amin(z_planes)).argmin()]
        max_z = z[np.fabs(z - np.amax(z_planes)).argmin()]

        # loop through z slices of the grid, find the nearest plane of contours
        # and use the structure mask for that plane
        for i, z_i in enumerate(z):
            if z_i < min_z or z_i > max_z:
                continue

            z_closest_plane = z_planes[np.fabs(z_planes - z_i).argmin()]
            tmp = np.fabs(z - z_closest_plane)
            if i not in np.argwhere(tmp == tmp.min()):
                continue

            contours_list = planes[z_closest_plane]

            if len(contours_list) == 1:
                c = contours_list[0]
                structure_mask_2d = polygon_mask(c, points_2d)
            else:
                # enclosed by  odd number of contours =>  inside structure
                # enclosed by even number of contours => outside structure
                switch = np.ones_like(points_2d[:,0]).astype(np.int8)
                for c in contours_list:
                    polygon_mask_tmp = polygon_mask(c, points_2d)
                    switch[polygon_mask_tmp] *= -1
                structure_mask_2d = (switch == -1)

            mask_3d[:,:,i] = structure_mask_2d.reshape((x.size, y.size))

        return mask_3d

    def structure_volume(self, name):
        """Compute the volume of a structure.

        Warning! This may significantly disagree with np.sum(dx*dy*dz*mask) if
        the mask is on a grid that does not contain the whole structure.

        Parameters:
            name: structure name
        """

        if self.is_empty_structure(name):
            return None

        for roi in self.ds.ROIContourSequence:
            if roi.ReferencedROINumber == self._roi_lookup[name]:
                all_contours = roi.ContourSequence

        # planes is a dict, where the key is the z-coordinate and the value is
        # a list of contours in that plane (each contour is a list of points)
        planes = {}
        for c in all_contours:
            contour_points_3d = np.array(c.ContourData).reshape((-1,3))
            contour_points_2d = contour_points_3d[:,0:2]
            contour_z = contour_points_3d[0,2]

            # round the float to use as a key
            contour_z = round(contour_z, 4)
            contours_list = planes.setdefault(contour_z, [])
            contours_list.append(contour_points_2d)

        volume = 0.0
        z = np.array(sorted(list(planes.keys())))
        dz = np.diff(z)
        for i, z_i in enumerate(z):
            contours = planes[z_i]
            area_plane = 0.0
            for i_c, c in enumerate(contours):
                area_polygon = polygon_area(c)

                n_hierarchy = 0
                for other_poly_points in contours[:i_c] + contours[i_c+1:]:
                    other_poly = Path(other_poly_points)
                    n_hierarchy += other_poly.contains_point(c[0])

                if n_hierarchy % 2 == 0:
                    area_plane += area_polygon
                else:
                    area_plane -= area_polygon

            thickness = dz[i] if i < len(dz) else dz[-1]
            volume += thickness * area_plane

        return volume


def polygon_mask(polygon_points, grid_points):
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


def polygon_area(points):
    """Computes the area of a polygon, using the shoelace formula.

    Parameters:
        points: numpy.ndarray with shape (p, 2)
    """
    x, y = points.T
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
