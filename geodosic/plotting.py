# system imports
import logging

# third-party imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


window_aliases = {
    'default':     ( 400,   40),
    'abdomen':     ( 300,  -10),
    'mediastinum': ( 300,  -10),
    'head':        ( 125,   45),
    'liver':       ( 305,   80),
    'lung':        (1500, -500),
    'spine':       ( 300,   30),
    'bone':        (1500,  400),
}

sagittal_aliases = ('sag', 'sagittal')
coronal_aliases = ('cor', 'coronal')
transverse_aliases = ('tra', 'trans', 'transverse', 'ax', 'axial')


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
    if view in sagittal_aliases:

        if i >= array_3d.shape[0]:
            logging.warning('index %s is out-of-bounds' % i)
            i = -1

        array_2d = array_3d[i,:,:].T
        left, right = np.amin(y), np.amax(y)
        bottom, top = np.amin(z), np.amax(z)

        if np.all(np.diff(y) < 0):
            array_2d = np.fliplr(array_2d)

        if np.all(np.diff(z) < 0):
            array_2d = np.flipud(array_2d)

    # coronal image: {bottom, left} -> {inferior, patient's right}
    elif view in coronal_aliases:

        if i >= array_3d.shape[1]:
            logging.warning('index %s is out-of-bounds' % i)
            i = -1

        array_2d = array_3d[:,i,:].T
        left, right = np.amin(x), np.amax(x)
        bottom, top = np.amin(z), np.amax(z)

        if np.all(np.diff(x) < 0):
            array_2d = np.fliplr(array_2d)

        if np.all(np.diff(z) < 0):
            array_2d = np.flipud(array_2d)

    # transverse image: {bottom, left} -> {posterior, patient's right}
    elif view in transverse_aliases:

        if i >= array_3d.shape[2]:
            logging.warning('index %s is out-of-bounds' % i)
            i = -1

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


def plot_overlay(scan_array, scan_grid, overlay_array, overlay_grid, i, view,
                 scan_window='default', scan_cbar=False,
                 overlay_alpha=0.5, overlay_cbar=True,
                 overlay_invisible_zero=False):

    # window is user-defined or an alias
    try:
        from numbers import Number
        assert len(scan_window) == 2
        assert isinstance(scan_window[0], Number)
        assert isinstance(scan_window[1], Number)
    except:
        if scan_window not in window_aliases:
            logging.warning('Unrecognized window alias "%s"' % scan_window)
            scan_window = 'default'
        window_width, window_center = window_aliases[scan_window]
    else:
        window_width, window_center = scan_window

    # plot scan
    scan_min = window_center - window_width/2.0
    scan_max = window_center + window_width/2.0
    scan, extent = slice_array(scan_array, scan_grid, i, view)
    plt.imshow(scan, extent=extent,
               cmap=cm.bone, vmin=scan_min, vmax=scan_max)
    if scan_cbar:
        plt.colorbar()

    plt.hold(True)

    # convert between coordinate vectors
    if view in sagittal_aliases:
        i = -1 if i >= scan_array.shape[0] else i
        c_scan, c_overlay = scan_grid[0], overlay_grid[0]
    elif view in coronal_aliases:
        i = -1 if i >= scan_array.shape[1] else i
        c_scan, c_overlay = scan_grid[1], overlay_grid[1]
    elif view in transverse_aliases:
        i = -1 if i >= scan_array.shape[2] else i
        c_scan, c_overlay = scan_grid[2], overlay_grid[2]

    c_slice = c_scan[i]
    # don't plot overlay if out-of-bounds
    if c_slice > np.amax(c_overlay) or c_slice < np.amin(c_overlay):
        return
    i = np.argmin(np.fabs(c_overlay - c_slice))

    # plot overlay
    overlay_max = np.amax(overlay_array)
    overlay, extent = slice_array(overlay_array, overlay_grid, i, view)

    overlay_cm = cm.jet
    if overlay_invisible_zero:
        overlay_cm.set_under('k', alpha=0)
        overlay = overlay.copy()
        overlay[overlay==0] = -1

    fig = plt.imshow(overlay, extent=extent,
                    cmap=overlay_cm, vmax=overlay_max, alpha=overlay_alpha)
    if overlay_invisible_zero:
        clim = im.get_clim()
        fig.set_clim((0, clim[1]))
    if overlay_cbar:
        plt.colorbar()

    fig.axes.get_xaxis().set_ticks([])
    fig.axes.get_yaxis().set_ticks([])
