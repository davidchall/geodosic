# system imports
import logging

# third-party imports
import numpy as np


def slice_array(array_3d, grid, i, view, origin='upper'):
    """Takes a slice through a 3D array.

    Parameters:
        array_3d: numpy.ndarray with shape (m1, m2, m3)
        grid:     (x,y,z) coordinate vectors of array_3d
        i:        sliced frame
        view:     which plane to slice {'sag', 'cor', 'tra'}
        origin:   first array index plotted in {'upper', 'lower'} left corner

    Returns:
        array_2d: numpy.ndarray with shape (mx, my) that depends upon view
        extent:   location of lower-left and upper-right corners
    """

    if origin not in ('upper', 'lower'):
        logging.warning('Unrecognized origin "%s" (using "upper")' % origin)
        origin = 'upper'

    # assumes BIPED coordinate system (DICOM C.7.6.2.1.1):
    # x-axis: increasing to patient's left side
    # y-axis: increasing to posterior
    # z-axis: increasing to superior

    # plt.imshow expects [i,j] -> [row,column]
    x, y, z = grid

    # sagittal image: {bottom, left} -> {inferior, anterior}
    if view in ('x', 'sag', 'sagittal'):

        array_2d = array_3d[i,:,:].T
        left, right = np.amin(y), np.amax(y)
        bottom, top = np.amin(z), np.amax(z)

        if np.all(np.diff(y) < 0):
            array_2d = np.fliplr(array_2d)

        if np.all(np.diff(z) < 0):
            array_2d = np.flipud(array_2d)

    # coronal image: {bottom, left} -> {inferior, patient's right}
    elif view in ('y', 'cor', 'coronal'):

        array_2d = array_3d[:,i,:].T
        left, right = np.amin(x), np.amax(x)
        bottom, top = np.amin(z), np.amax(z)

        if np.all(np.diff(x) < 0):
            array_2d = np.fliplr(array_2d)

        if np.all(np.diff(z) < 0):
            array_2d = np.flipud(array_2d)

    # transverse image: {bottom, left} -> {posterior, patient's right}
    elif view in ('z', 'tra', 'trans', 'transverse', 'axial'):

        array_2d = array_3d[:,:,i].T
        array_2d = np.flipud(array_2d)
        left, right = np.amin(x), np.amax(x)
        bottom, top = np.amax(y), np.amin(y)

        if np.all(np.diff(y) < 0):
            array_2d = np.fliplr(array_2d)

        if np.all(np.diff(x) < 0):
            array_2d = np.flipud(array_2d)

    if origin == 'upper':
        array_2d = np.flipud(array_2d)

    extent = (left, right, bottom, top)
    return array_2d, extent
