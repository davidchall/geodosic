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
    from mahotas.labeled import borders
    return borders(image)


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
