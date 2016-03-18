# third-party imports
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt
from six.moves import reduce


def bwperim(image):
    """Find the perimeter of objects in a binary image.

    This is used to find the surface of structure masks.

    Parameters:
        image: ndarray

    Returns:
        mask: ndarray of bool
    """
    from mahotas.labeled import bwperim
    return bwperim(image)


def distance_to_surface(mask, grid_spacing=(1, 1, 1)):
    """Return the Euclidean distance from the surface of a structure.

    Within the structure itself, the distance is set negative.

    Parameters:
        mask: 3-d numpy.ndarray of bools with shape (m1, m2, m3), indicating
            the presence of the structure upon a regular grid
        grid_spacing: 3-tuple describing the length scales of a single voxel

    Returns:
        distance: 3-d numpy.ndarray of floats with shape (m1, m2, m3)
    """
    # find surface of structure
    perim = bwperim(mask)

    # euclidean distance transform (calculated from False voxels)
    distance = distance_transform_edt(~perim, sampling=grid_spacing)

    # set negative within structure
    distance[mask] *= -1

    return distance


def bin_distance(min_dist, max_dist, width):

    dist_edges = np.arange(0., max_dist, width)
    dist_edges = np.append(dist_edges, np.inf)
    i_shell = np.arange(1, dist_edges.size)

    if min_dist < -width:
        neg_dist_edges = np.arange(width, -min_dist, width)
        neg_dist_edges = np.append(neg_dist_edges, np.inf)
        dist_edges = np.insert(dist_edges, 0, -neg_dist_edges[::-1])

        neg_i_shell = np.arange(1, neg_dist_edges.size+1)
        i_shell = np.insert(i_shell, 0, -neg_i_shell[::-1])
    else:
        dist_edges = np.insert(dist_edges, 0, -np.inf)
        i_shell = np.insert(i_shell, 0, -1)

    return i_shell, dist_edges


def centroid(density, grid, indices=False):
    """Returns the centroid of a 3-d solid.

    Parameters:
        density: 3-d ndarray used to calculate center-of-mass
        grid: (x,y,z) coordinate vectors of the mask
        indices: if True, will return indices instead of coordinates

    Returns:
        x0,y0,z0: coordinates (or indices) of centroid
    """
    x, y, z = grid
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

    x0 = np.average(xx, weights=density)
    y0 = np.average(yy, weights=density)
    z0 = np.average(zz, weights=density)

    if indices:
        x0 = np.argmin(np.fabs(x - x0))
        y0 = np.argmin(np.fabs(y - y0))
        z0 = np.argmin(np.fabs(z - z0))

    return x0, y0, z0


def interpolate_grids(values, old_grid, new_grid):
    """Interpolate data from one regular grid to another.

    Outside the bounds of old_grid, the result is set to zero.

    Parameters:
        values: numpy.ndarray with shape (m1, ..., mn)
        old_grid: tuple of numpy.ndarray, with shapes (m1,), ..., (mn,)
        new_grid: tuple of numpy.ndarray, with shapes (r1,), ..., (rn,)

    Returns:
        result: numpy.ndarray with shape (r1, ..., rn)
    """
    interpolator = RegularGridInterpolator(old_grid, values,
        bounds_error=False, fill_value=0)

    result = interpolator(cartesian_product(new_grid))
    new_shape = [d.size for d in new_grid]

    return result.reshape(new_shape)


def cartesian_product(arrays):
    """Computes the n-fold Cartesian product of the input arrays.

    This can be used to construct a grid of Cartesian coordinates from a set
    of coordinate vectors.

    Parameters:
        arrays: tuple of numpy.ndarray, with shapes (m1,), ..., (mn,)

    Returns:
        points: numpy.ndarray with shape (m1 * ... * mn, n), where each element
            is an n-tuple containing the coordinates for that point
    """
    # implementation from http://stackoverflow.com/a/11146645/2669425
    broadcastable = np.ix_(*arrays)
    broadcasted = np.broadcast_arrays(*broadcastable)
    rows, cols = reduce(np.multiply, broadcasted[0].shape), len(broadcasted)
    out = np.empty(rows * cols, dtype=broadcasted[0].dtype)
    start, end = 0, rows
    for a in broadcasted:
        out[start:end] = a.reshape(-1)
        start, end = end, end + rows
    return out.reshape(cols, rows).T
