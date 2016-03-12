import numpy as np
import matplotlib.pyplot as plt


class DVH(object):
    """A class representing a dose-volume histogram."""

    def __init__(self, volumes, dose_edges, dDVH=False):

        if not np.all(np.diff(dose_edges) > 0):
            raise ValueError('Bin edges must be monotonically increasing')
        if len(dose_edges) != len(volumes)+1:
            raise ValueError('There must be N+1 bin edges')

        if dDVH:
            self.dDVH = np.array(volumes, float)
        else:
            if not np.all(np.diff(volumes) <= 0):
                raise ValueError('Volumes must be monotonically decreasing. '
                                 'You might want to set dDVH=True')
            self.cDVH = np.array(volumes, float)
        self.dose_edges = np.array(dose_edges, float)

        if self.cDVH[-1] != 0:
            raise ValueError('Cumulative DVH does not reach zero.')

        if self.dose_edges[0] != 0:
            raise ValueError('First dose edge must correspond to zero dose.')

    @classmethod
    def from_raw(cls, voxel_data, dose_edges):
        return cls(*np.histogram(voxel_data, bins=dose_edges), dDVH=True)

    @property
    def cDVH(self):
        return self._cDVH

    @cDVH.setter
    def cDVH(self, volumes):
        volumes = volumes / np.amax(volumes)
        self._cDVH = volumes
        self._dDVH = np.diff(volumes[::-1])[::-1]

    @property
    def dDVH(self):
        return self._dDVH

    @dDVH.setter
    def dDVH(self, volumes):
        volumes = volumes / np.sum(volumes)
        self._dDVH = volumes
        self._cDVH = np.cumsum(volumes[::-1])[::-1]

    @property
    def dose_centers(self):
        return 0.5 * (self.dose_edges[1:] + self.dose_edges[:-1])

    def plot(self, dDVH=False, label=None):
        x = self.dose_centers
        y = self.dDVH if dDVH else self.cDVH
        plt.plot(x, y, drawstyle='steps-mid', label=label)

    def min(self):
        """Return the minimum dose.

        We don't have access to the exact value, so the lower edge of the
        first non-zero bin is returned.
        """
        i = np.where(self.dDVH > 0)[0][0]
        return self.dose_edges[i]

    def max(self):
        """Return the maximum dose.

        We don't have access to the exact value, so the upper edge of the
        last non-zero bin is returned.
        """
        i = np.where(self.dDVH > 0)[0][-1]
        return self.dose_edges[i+1]

    def mean(self):
        """Return the mean dose.
        """
        return np.average(self.dose_centers, weights=self.dDVH)

    def std(self):
        """Return the standard deviation of the dose distribution.
        """
        mean = self.mean()
        variance = np.average(np.square(self.dose_centers - mean),
                              weights=self.dDVH)
        return np.sqrt(variance)

    def dose_to_volume(self, volume_threshold):
        """Return the minimum dose received by a certain volume fraction.
        """
        assert volume_threshold >= 0
        assert volume_threshold <= 1

        if volume_threshold == 0:
            return self.max()
        if volume_threshold == 1:
            return self.min()

        d, v = self.dose_edges, self.cDVH
        i = np.where(v >= volume_threshold)[0][-1]

        return d[i] + (d[i+1]-d[i]) * (volume_threshold-v[i]) / (v[i+1]-v[i])

    def volume_receiving_dose(self, dose_threshold):
        """Return the volume fraction that receives a dose above a threshold.

        Parameters:
            dose_threshold: given in same units as dose_edges
        """
        assert dose_threshold >= 0

        if dose_threshold <= self.min():
            return 1.0
        if dose_threshold >= self.max():
            return 0.0

        d, v = self.dose_edges, self.cDVH
        i = np.where(d <= dose_threshold)[0][-1]

        return v[i] + (v[i+1]-v[i]) * (dose_threshold-d[i]) / (d[i+1]-d[i])

    def eud(self, a):
        """Return the equivalent uniform dose.

        Parameters:
            a: seriality parameter
        """
        if a == 1.0:
            return self.mean()
        if a is np.inf:
            return self.max()
        if a is -np.inf:
            return self.min()

        tmp = np.average(np.power(self.dose_centers, a), weights=self.dDVH)
        return np.power(tmp, 1./a)
