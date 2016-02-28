from scipy.ndimage import distance_transform_edt
from mahotas.labeled import borders


def distance_to_surface(mask, voxel_spacing=(1,1,1)):
    """Return the Euclidean distance from the surface of a structure.

    Assumes that the mask grid is uniform in distance (i.e. regular spacing).
    """
    # find surface of structure
    perim = borders(mask)

    # euclidean distance transform
    # NOTE: distance calculated from False voxels
    dist = distance_transform_edt(~perim, sampling=voxel_spacing)

    # set negative within structure
    dist[mask] *= -1

    return dist
