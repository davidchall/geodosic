# standard imports
import inspect
from functools import wraps

# third-party imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy.stats as ss
from astropy.stats import histogram
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.optimize import curve_fit

# project imports
from .geometry import distance_to_surface
from .dvh import DVH


class skew_normal_gen(ss.rv_continuous):
    def _argcheck(self, skew):
        return np.isfinite(skew)

    def _pdf(self, x, skew):
        return 2 * ss.norm.pdf(x) * ss.norm.cdf(x * skew)

skew_normal = skew_normal_gen(name='skew-normal')


def skew_normal_pdf(x, e=0, w=1, a=0):
    """PDF for skew-normal distribution.

    Parameters:
        e: location
        w: scale
        a: shape
    """
    t = (x-e) / w
    return 2 / w * ss.norm.pdf(t) * ss.norm.cdf(a*t)


def initializer(func):
    """Decorator that automatically assigns parameters to attributes."""
    names, varargs, keywords, defaults = inspect.getargspec(func)

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        for name, arg in list(zip(names[1:], args)) + list(kwargs.items()):
            setattr(self, name, arg)

        for name, default in zip(reversed(names), reversed(defaults)):
            if not hasattr(self, name):
                setattr(self, name, default)

        func(self, *args, **kwargs)

    return wrapper


class ShellDoseFitModel(BaseEstimator, RegressorMixin):
    """docstring for DVHEstimator"""

    @initializer
    def __init__(self, dose_name=None, oar_name=None, target_name=None,
                 shell_width=3.0, method='rv_continuous'):
        pass

    def fit(self, X, y=None, plot_file=None):
        """Train the model.

        Parameters:
            X: the training cohort (a list of patient objects)
        """

        # validate model parameters
        assert self.dose_name is not None
        assert self.oar_name is not None
        assert self.target_name is not None
        assert self.shell_width > 0

        self.pp = PdfPages(plot_file) if plot_file else None
        if self.pp:
            self.iPatient = 0

        # fit dose shells for all patients
        popt_all = [self._fit_patient(p) for p in X]

        if self.pp:
            self.pp.close()
            del self.pp, self.iPatient, self.iShell

        # what range of shell distances were covered?
        min_i = min(min(popt.keys()) for popt in popt_all)
        max_i = max(max(popt.keys()) for popt in popt_all)

        # compute average best-fit parameters for each shell
        self.popt_avg_ = {}
        for i in range(min_i, max_i+1):
            if i == 0:
                continue
            popt_all_i = np.array([popt[i] for popt in popt_all if i in popt])
            if popt_all_i.size == 0:
                raise RuntimeError('Uncovered shell: {0}'.format(i))

            self.popt_avg_[i] = np.mean(popt_all_i, axis=0)

        return self

    def _fit_patient(self, p):
        """Find correlations between dose and distance-to-target for a single
        patient. This method splits the dose array into shells surrounding the
        target volume, and also selects only voxels within the organ-at-risk.
        The actual fitting is handled by _fit_shell().

        Parameters:
            p: a patient object

        Returns:
            popt: dict of (i: popt_i) pairs, where popt_i is a tuple containing
                the best-fit parameters for shell i
        """
        if self.dose_name not in p.dose_names():
            return

        if self.oar_name not in p.structure_names():
            return

        if self.pp:
            self.iPatient += 1

        dose = p.dose_array(self.dose_name)
        dose_grid = p.dose_grid_vectors(self.dose_name)
        dose_spacing = p.dose_grid_spacing(self.dose_name)

        target_mask = p.structure_mask(self.target_name, dose_grid)
        oar_mask = p.structure_mask(self.oar_name, dose_grid)

        target_dose = np.mean(dose[target_mask])
        dose = dose.copy() / target_dose

        dist = distance_to_surface(target_mask, dose_spacing)
        min_dist = np.amin(dist[oar_mask])
        max_dist = np.amax(dist[oar_mask])
        i_shell, dist_edges = self._bin_distance(min_dist, max_dist, self.shell_width)

        popt = {}
        for i, inner, outer in zip(i_shell, dist_edges[:-1], dist_edges[1:]):

            dose_shell = dose[oar_mask & (dist > inner) & (dist < outer)]

            if dose_shell.size < 4:
                continue

            if self.pp:
                self.iShell = i

            try:
                popt[i] = self._fit_shell(dose_shell)
            except:
                continue

        return popt

    def _fit_shell(self, dose):
        """Fit the dose distribution within a shell.

        Parameters:
            dose: ndarray of dose (voxel selection already applied)

        Returns:
            popt: tuple of best-fit parameters
        """

        # bin the data and add empty bins at either end to constrain fit
        counts, bin_edges = histogram(dose, bins='freedman', density=True)
        bin_edges = np.insert(bin_edges, 0, 2*bin_edges[0]-bin_edges[1])
        bin_edges = np.append(bin_edges, 2*bin_edges[-1]-bin_edges[-2])
        counts = np.insert(counts, 0, 0.)
        counts = np.append(counts, 0.)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

        # using scipy.stats.rv_continuous.fit (fit raw data)
        if self.method == 'rv_continuous':
            popt = skew_normal.fit(dose, ss.skew(dose),
                                   loc=np.mean(dose), scale=np.std(dose))
            popt = popt[-2:] + popt[:-2]

        # using scipy.optimize.curve_fit (fit a histogram)
        if self.method == 'curve_fit':
            p0 = (np.mean(dose), np.std(dose), ss.skew(dose))
            p_upper = (2, 1, 1000)
            p_lower = (-1, 0, -1000)

            popt, pcov = curve_fit(skew_normal_pdf, bin_centers, counts,
                                   p0=p0, bounds=(p_lower, p_upper))

        if self.pp:
            plt.plot(bin_centers, counts, drawstyle='steps-mid', label='Data')
            x = np.linspace(np.amin(bin_centers), np.amax(bin_centers), 100)
            plt.plot(x, skew_normal_pdf(x, *popt), label='Fit')
            plt.xlabel('Dose / Target Dose')
            plt.legend(loc='best')
            plt.title('Patient %i: Shell %i' % (self.iPatient, self.iShell))
            self.pp.savefig()
            plt.clf()

        return popt

    def _bin_distance(self, min_dist, max_dist, width):

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

    def predict(self, X):
        return [self.predict_patient(p) for p in X]

    def predict_patient(self, p):
        if self.oar_name not in p.structure_names():
            return

        # TODO: should not rely on dose array
        grid = p.dose_grid_vectors(self.dose_name)
        grid_spacing = p.dose_grid_spacing(self.dose_name)

        target_mask = p.structure_mask(self.target_name, grid)
        oar_mask = p.structure_mask(self.oar_name, grid)

        dist = distance_to_surface(target_mask, grid_spacing)
        min_dist = np.amin(dist[oar_mask])
        max_dist = np.amax(dist[oar_mask])
        i_shell, dist_edges = self._bin_distance(min_dist, max_dist, self.shell_width)

        dose_edges = np.linspace(0., 1.2, 120)
        dose_centers = 0.5 * (dose_edges[1:] + dose_edges[:-1])
        dose_counts = np.zeros_like(dose_centers)

        for i, inner, outer in zip(i_shell, dist_edges[:-1], dist_edges[1:]):

            n_voxels_shell = np.sum(oar_mask[(dist > inner) & (dist < outer)])

            popt = self.popt_avg_[i]
            dose_shell = skew_normal_pdf(dose_centers, *popt)
            dose_counts += n_voxels_shell * dose_shell

        # parametric fits yield long tails, but DVH requires last bin is zero
        dose_counts[-1] = 0
        return DVH(dose_counts, dose_edges, dDVH=True)
