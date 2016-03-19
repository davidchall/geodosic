# system imports
import logging

# third-party imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# project imports
from .geometry import bwperim


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


def plot_overlay(scan, grid_scan, overlay, grid_overlay, i_scan, view,
                 window='default', cbar_scan=False,
                 lim_overlay='global', cbar_overlay=True,
                 alpha=0.5, invisible_zero=False,
                 structures=[]):
    """Plot a grayscale scan with a color overlay and outlined structures.

    Warning!! When saving the figure, it may be necessary to increase the
    DPI (~200) in order for structure outlines to display correctly.

    Parameters:
        scan:           3D scan array
        grid_scan:      (x,y,z) coordinate vectors of scan
        overlay:        3D overlay array
        grid_overlay:   (x,y,z) coordinate vectors of overlay
        i_scan:         sliced frame
        view:           which plane to slice {'sag', 'cor', 'tra'}

    Optional parameters:
        window:         an alias (e.g. 'lung') or a (width, center) tuple
        cbar_scan:      display a colorbar for the scan
        lim_overlay:    'global', 'local' or a (min, max) tuple
        cbar_overlay:   display a colorbar for the overlay
        alpha:          overlay alpha blending
        invisible_zero: makes voxels containing zero transparent
        structures:     list of structure masks (should either be defined on
                        grid_scan or grid_overlay)
    """
    # be explicit
    array_scan = scan
    array_overlay = overlay

    if view in sagittal_aliases:
        dim = 0
    elif view in coronal_aliases:
        dim = 1
    elif view in transverse_aliases:
        dim = 2
    else:
        logging.error('Unrecognized view "%s"' % view)
        return

    # convert between coordinate vectors (c = generic coordinate)
    i_scan = -1 if i_scan >= array_scan.shape[dim] else i_scan
    c_scan, c_overlay = grid_scan[dim], grid_overlay[dim]
    c_slice = c_scan[i_scan]
    i_overlay = np.argmin(np.fabs(c_overlay - c_slice))

    # don't plot overlay if out-of-bounds
    plot_overlay = True
    if c_slice > np.amax(c_overlay) or c_slice < np.amin(c_overlay):
        plot_overlay = False

    ###########################
    #        plot scan        #
    ###########################
    # window is user-defined or an alias
    try:
        from numbers import Number
        assert len(window) == 2
        assert isinstance(window[0], Number)
        assert isinstance(window[1], Number)
    except:
        if window not in window_aliases:
            logging.warning('Unrecognized window alias "%s"' % window)
            window = 'default'
        window_width, window_center = window_aliases[window]
    else:
        window_width, window_center = window

    min_scan = window_center - window_width/2.0
    max_scan = window_center + window_width/2.0
    slice_scan, extent = slice_array(array_scan, grid_scan, i_scan, view)
    fig = plt.imshow(slice_scan, extent=extent,
                     cmap=cm.bone, vmin=min_scan, vmax=max_scan)
    if cbar_scan:
        plt.colorbar()

    plt.hold(True)

    ##########################
    #      plot overlay      #
    ##########################
    if plot_overlay:
        slice_overlay, extent = slice_array(array_overlay, grid_overlay,
                                            i_overlay, view)
        if lim_overlay == 'global':
            min_overlay = np.amin(array_overlay)
            max_overlay = np.amax(array_overlay)
        elif lim_overlay == 'local':
            min_overlay = np.amin(slice_overlay)
            max_overlay = np.amax(slice_overlay)
        else:
            try:
                min_overlay, max_overlay = lim_overlay
                min_overlay = float(min_overlay)
                max_overlay = float(max_overlay)
            except:
                logging.warning('Unrecognized lim_overlay')
                min_overlay = np.amin(array_overlay)
                max_overlay = np.amax(array_overlay)

        cm_overlay = cm.jet
        if invisible_zero:
            if np.amin(slice_overlay) < 0:
                logging.warning('The invisible_zero option should not be used '
                    'when the overlay contains negative values.')
            cm_overlay.set_under('k', alpha=0)
            slice_overlay = slice_overlay.copy()
            slice_overlay[slice_overlay == 0] = -1

        fig = plt.imshow(slice_overlay, extent=extent, alpha=alpha,
                         cmap=cm_overlay, vmin=min_overlay, vmax=max_overlay)
        if invisible_zero:
            clim = fig.get_clim()
            fig.set_clim((0, clim[1]))
        if cbar_overlay:
            plt.colorbar()

    #######################################
    #      plot structure perimeters      #
    #######################################
    if len(structures) > 0:
        slice_perim_scan = np.zeros_like(slice_scan, dtype=bool)
        slice_perim_overlay = np.zeros_like(slice_overlay, dtype=bool)

        for mask in structures:
            slice_mask_shape = mask.shape[:dim] + mask.shape[dim+1:]

            if set(slice_mask_shape) == set(slice_scan.shape):
                slice_mask, extent_scan = slice_array(mask, grid_scan,
                                                      i_scan, view)
                slice_perim_scan += bwperim(slice_mask)

            elif set(slice_mask_shape) == set(slice_overlay.shape):
                slice_mask, extent_overlay = slice_array(mask, grid_overlay,
                                                         i_overlay, view)
                slice_perim_overlay += bwperim(slice_mask)

            else:
                logging.error('Structure found on unrecognized grid')
                continue

        # use a masked array, since masked values are set as "bad"
        # and matplotlib makes these pixels transparent
        if np.any(slice_perim_scan):
            slice_perim_scan = np.ma.masked_equal(slice_perim_scan, False)
            fig = plt.imshow(slice_perim_scan, extent=extent_scan,
                             cmap='binary', interpolation='none')

        if np.any(slice_perim_overlay):
            slice_perim_overlay = np.ma.masked_equal(slice_perim_overlay, False)
            fig = plt.imshow(slice_perim_overlay, extent=extent_overlay,
                             cmap='binary', interpolation='none')

    # tidy up
    fig.axes.get_xaxis().set_ticks([])
    fig.axes.get_yaxis().set_ticks([])
