# standard imports
import os
import inspect
from functools import wraps
from math import ceil
import logging

# third-party imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy.stats as ss
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from sklearn.metrics import r2_score
from statsmodels.stats.weightstats import DescrStatsW

# project imports
from .geometry import distance_to_surface, bin_distance, interpolate_grids
from .dvh import DVH


def skew_normal_pdf(x, e=0, w=1, a=0):
    """PDF for skew-normal distribution.

    Parameters:
        e: location
        w: scale
        a: shape
    """
    t = (x-e) / w
    return 2 / w * ss.norm.pdf(t) * ss.norm.cdf(a*t)


def initialize_attributes(func):
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

    @initialize_attributes
    def __init__(self, dose_name=None, oar_names=None, target_name=None,
                 grid_name=None, shell_width=3.0,
                 min_shell_size_fit=10, min_structures_for_fit=2):
        pass

    def fit(self, X, y=None, plot_shell_func=None, plot_structure_func=None, return_popt_all=False):
        """Train the model.

        Parameters:
            X: the training cohort (a list of patient objects)
        """

        # validate model parameters
        assert self.dose_name is not None
        assert self.oar_names is not None
        assert self.target_name is not None
        assert self.shell_width > 0
        assert self.min_shell_size_fit > 0
        assert self.min_structures_for_fit > 0

        if self.grid_name is None:
            self.grid_name = self.dose_name

        if isinstance(self.oar_names, str):
            self.oar_names = [self.oar_names]

        self.plot_shell_func = plot_shell_func
        self.plot_structure_func = plot_structure_func

        # parameter bounds
        self.p_upper = [2, 1, 10]
        self.p_lower = [-1, 1e-9, -10]

        # Find appropriate dose and distance-to-target binning
        i_shell, dist_edges = bin_distance(-6, 40, self.shell_width)
        dose_edges = np.linspace(0, 1.2, 100)

        # histogram dose shells between patients
        bins = (dist_edges, dose_edges)
        dose_shells_all = np.zeros((len(dist_edges)-1, len(dose_edges)-1))
        for p in X:
            for oar_name in self.oar_names:
                dose_shells = self._histogram_dose_shells(p, oar_name, bins)
                if dose_shells is None:
                    continue
                shell_sizes = np.sum(dose_shells, axis=1)[np.newaxis].T
                shell_sizes[shell_sizes == 0] = 1
                dose_shells_all += dose_shells / shell_sizes

        shell_sizes_all = np.sum(dose_shells_all, axis=1)[np.newaxis].T
        shell_sizes_all[shell_sizes_all == 0] = 1
        self.dose_shells_all = dose_shells_all / shell_sizes_all / np.diff(dose_edges)[0]

        plt.imshow(dose_shells_all.T, extent=(dist_edges[1], dist_edges[-2], dose_edges[0], dose_edges[-1]),
            aspect='auto', origin='lower', interpolation='none')
        plt.xlabel('Distance-to-target [mm]')
        plt.ylabel('Normalized dose')

        # fit accumulated dose shells
        self.failed_converge = 0
        self.attempt_converge = 0
        self.popt_avg_ = {}
        self.popt_std_ = {}
        for j, i in enumerate(i_shell):
            popt_i = self._fit_shell(dose_edges, self.dose_shells_all[j,:])
            if popt_i is not None:
                self.popt_avg_[i] = popt_i[0]
                self.popt_std_[i] = np.sqrt(np.diag(popt_i[1]))

        if self.failed_converge > 0:
            logging.warning('{0}/{1} fits failed to converge (instead used mean, std, skew)'.format(self.failed_converge, self.attempt_converge))
        del self.attempt_converge
        del self.failed_converge

        return self

    def _histogram_dose_shells(self, p, oar_name, bins):
        if self.dose_name not in p.dose_names:
            return

        if oar_name not in p.structure_names:
            return

        if self.target_name not in p.structure_names:
            logging.error('Could not find "%s" in %s' % (self.target_name, p.dicom_dir))
            return

        _, self.tmp_anon_id = os.path.split(p.dicom_dir)
        self.tmp_oar_name = oar_name

        target_mask = p.structure_mask(self.target_name, self.grid_name)
        oar_mask = p.structure_mask(oar_name, self.grid_name)
        dist = p.distance_to_surface(self.target_name, self.grid_name)

        dose = p.dose_array(self.dose_name, self.grid_name)
        target_dose = np.mean(dose[target_mask])
        dose /= target_dose

        dose_oar = dose[oar_mask]
        dist_oar = dist[oar_mask]

        counts, _, _ = np.histogram2d(dist_oar, dose_oar, bins=bins)
        return counts

    def _fit_shell(self, bin_edges, counts):
        """Fit the dose distribution within a shell.

        Parameters:
            dose: ndarray of dose (voxel selection already applied)

        Returns:
            popt: tuple of best-fit parameters
        """
        if not np.any(counts):
            return None

        # initial estimate and bounds of parameter values
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        wgt_stats = DescrStatsW(bin_centers, weights=counts)

        p0 = [wgt_stats.mean, wgt_stats.std, 0]
        for i in range(len(p0)):
            p0[i] = min(p0[i], self.p_upper[i])
            p0[i] = max(p0[i], self.p_lower[i])

        # if dose is uniform, there is no distribution to fit
        if np.all(counts == counts[0]):
            return p0

        if hasattr(self, 'attempt_converge'):
            self.attempt_converge += 1

        # fit data
        try:
            popt, pcov = curve_fit(skew_normal_pdf, bin_centers, counts,
                                   p0=p0, bounds=(self.p_lower, self.p_upper))
        except RuntimeError as e:
            # if convergence fails, just use initial parameters
            self.failed_converge += 1
            popt = p0
            pcov = np.ones((3, 3))

        except Exception as e:
            print(np.sum(counts))
            print(bin_edges.size)
            print(counts.size)
            print(bin_edges)
            print(counts)
            raise e

        if self.plot_shell_func:
            self.plot_shell_func(self, bin_centers, counts, popt)

        return popt, pcov

    def interpolate_popt(self):
        i_shell = np.array(sorted(list(self.popt_avg_.keys())))
        yp = zip(*(self.popt_avg_[i] for i in i_shell if i in self.popt_avg_))
        dyp = zip(*(self.popt_std_[i] for i in i_shell if i in self.popt_std_))
        x = np.where(i_shell > 0, self.shell_width*(i_shell-0.5), self.shell_width*(i_shell+0.5))

        splines = []
        for y, dy in zip(yp, dyp):
            dy = np.clip(dy, 0.1*np.mean(dy), np.amax(dy))
            weights = np.power(dy, -2)
            splines.append(UnivariateSpline(x, y, w=weights, s=0.8))

        return splines

    def predict(self, X, *args, **kwargs):
        if isinstance(self.oar_names, str):
            self.oar_names = [self.oar_names]

        pred = []
        for p in X:
            for oar_name in self.oar_names:
                pred.append(self.predict_structure(p, oar_name, *args, **kwargs))
        return pred

    def predict_structure(self, p, oar_name, dose_edges=None,
                          max_size_voxelwise=1000):
        if oar_name not in p.structure_names:
            return
        if self.target_name not in p.structure_names:
            return

        min_fitted_i = min(self.popt_avg_.keys())
        max_fitted_i = max(self.popt_avg_.keys())

        oar_mask = p.structure_mask(oar_name, self.grid_name)
        dist = p.distance_to_surface(self.target_name, self.grid_name)
        dist_oar = dist[oar_mask]
        min_dist = np.amin(dist_oar)
        max_dist = np.amax(dist_oar)

        if dose_edges is None:
            dose_edges = np.linspace(0., 1.2, 120)
        dose_centers = 0.5 * (dose_edges[1:] + dose_edges[:-1])
        dose_counts = np.zeros_like(dose_centers)

        oar_size = dist_oar.size
        if oar_size <= max_size_voxelwise:

            popt_splines = self.interpolate_popt()
            min_fitted_dist = min_fitted_i * self.shell_width
            max_fitted_dist = max_fitted_i * self.shell_width

            for dist_voxel in dist_oar:
                if dist_voxel < min_fitted_dist:
                    popt = self.popt_avg_[min_fitted_i]
                elif dist_voxel > max_fitted_dist:
                    popt = self.popt_avg_[max_fitted_i]
                else:
                    popt = [spline(dist_voxel) for spline in popt_splines]

                for i in range(len(popt)):
                    popt[i] = min(popt[i], self.p_upper[i])
                    popt[i] = max(popt[i], self.p_lower[i])

                if np.all(popt == 0):
                    continue
                else:
                    dose_counts += skew_normal_pdf(dose_centers, *popt)

        else:
            i_shell, dist_edges = bin_distance(min_dist, max_dist, self.shell_width)

            for i, inner, outer in zip(i_shell, dist_edges[:-1], dist_edges[1:]):

                shell = (dist_oar > inner) & (dist_oar < outer)
                n_voxels_shell = np.count_nonzero(shell)
                if n_voxels_shell == 0:
                    continue

                if i < min_fitted_i:
                    popt = self.popt_avg_[min_fitted_i]
                elif i > max_fitted_i:
                    popt = self.popt_avg_[max_fitted_i]
                elif i not in self.popt_avg_:
                    dist = 0.5 * (inner + outer)
                    popt_splines = self.interpolate_popt()
                    popt = [spline(dist) for spline in popt_splines]
                else:
                    popt = self.popt_avg_[i]

                if np.all(popt == 0):
                    continue
                else:
                    dose_shell = skew_normal_pdf(dose_centers, *popt)
                    dose_counts += n_voxels_shell * dose_shell

        # parametric fits yield long tails, but DVH requires last bin is zero
        dose_counts[-1] = 0
        return DVH(dose_counts, dose_edges, dDVH=True)

    def score(self, X, y=None,
              metric_func='mean', metric_args=[], metric_label='metric',
              normalize=False, plot=False, **kwargs):

        if isinstance(self.oar_names, str):
            self.oar_names = [self.oar_names]

        y_true, y_pred = [], []  # metric evaluated
        for p in X:
            for oar_name in self.oar_names:
                if oar_name not in p.structure_names or \
                  self.target_name not in p.structure_names or \
                  self.dose_name not in p.dose_names:
                    continue

                dose = p.dose_array(self.dose_name, self.dose_name)
                target_mask = p.structure_mask(self.target_name, self.dose_name)
                norm_factor = np.mean(dose[target_mask])

                # choose appropriate binning for DVHs
                oar_mask = p.structure_mask(oar_name, self.dose_name)
                max_oar_dose = np.amax(dose[oar_mask]) / norm_factor
                max_dvh_dose = max(1.05*max_oar_dose, 1.2)
                dose_edges = np.linspace(0, max_dvh_dose, 200)

                dvh_plan = p.calculate_dvh(oar_name, self.dose_name, dose_edges=dose_edges*norm_factor)
                dvh_pred = self.predict_structure(p, oar_name, dose_edges=dose_edges, **kwargs)

                if normalize:
                    dvh_plan.dose_edges /= norm_factor
                else:
                    dvh_pred.dose_edges *= norm_factor

                y_pred.append(getattr(dvh_pred, metric_func)(*metric_args))
                y_true.append(getattr(dvh_plan, metric_func)(*metric_args))

        r2 = r2_score(y_true, y_pred)

        if plot:
            plt.scatter(y_true, y_pred, c='k')
            max_val = 1.1*max(*y_true, *y_pred)
            plt.plot([0, max_val], [0, max_val], 'k:')
            plt.xlabel('Planned ' + metric_label)
            plt.ylabel('Predicted ' + metric_label)
            plt.figtext(0.23, 0.8, '$R^2$ = {0:.1%}'.format(r2))
            plt.axis('square')
            plt.axis([0, max_val, 0, max_val])

        return r2
