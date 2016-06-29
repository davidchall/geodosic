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

# project imports
from .geometry import distance_to_surface, bin_distance, interpolate_grids
from .dvh import DVH

# update plotting style
from matplotlib.pylab import rcParams
params = {
    'figure.figsize': (10.0, 8.0),
    'font.size': 20,
    'legend.fontsize': 20,
}
rcParams.update(params)


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
                 grid_name=None, shell_width=3.0, min_shell_size_fit=10):
        pass

    def fit(self, X, y=None, plot_fits=None, plot_params=None):
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

        if self.grid_name is None:
            self.grid_name = self.dose_name

        if isinstance(self.oar_names, str):
            self.oar_names = [self.oar_names]

        self.pp = PdfPages(plot_fits) if plot_fits else None

        # fit dose shells for all patients
        popt_all = []
        for p in X:
            for oar_name in self.oar_names:
                popt_all.append(self._fit_structure(p, oar_name))
        popt_all = [popt for popt in popt_all if popt is not None]

        if self.pp:
            self.pp.close()
            del self.pp, self.tmp_anon_id, self.tmp_oar_name, self.tmp_i_shell

        # what range of shell distances were covered?
        min_i = min(min(popt.keys()) for popt in popt_all)
        max_i = max(max(popt.keys()) for popt in popt_all)

        # compute average best-fit parameters for each shell
        self.popt_avg_, self.popt_std_ = {}, {}
        for i in range(min_i, max_i+1):
            if i == 0:
                continue

            popt_all_i = np.array([popt[i] for popt in popt_all if i in popt])
            if popt_all_i.size == 0:
                continue

            self.popt_avg_[i] = np.mean(popt_all_i, axis=0)
            self.popt_std_[i] = np.std(popt_all_i, axis=0)

        if plot_params:
            self.pp = PdfPages(plot_params)

            i_shell = np.array(sorted(list(self.popt_avg_.keys())))
            yp = zip(*(self.popt_avg_[i] for i in i_shell))
            dyp = zip(*(self.popt_std_[i] for i in i_shell))
            x = np.where(i_shell > 0, self.shell_width*(i_shell-0.5), self.shell_width*(i_shell+0.5))

            xs = np.linspace(np.amin(x), np.amax(x), 100)
            splines = self.interpolate_popt()

            for param, (spline, y, dy) in enumerate(zip(splines, yp, dyp)):
                plt.errorbar(x, y, dy, fmt='ko')
                plt.plot(xs, spline(xs))
                plt.xlabel('Distance-to-target [mm]')
                plt.ylabel('Parameter estimate $\\theta_{{{0}}}$'.format(param+1))

                self.pp.savefig()
                plt.clf()

            for param in range(len(next(iter(self.popt_avg_.values())))):
                for popt in popt_all:
                    i_shell = np.array(sorted(list(popt.keys())))
                    y = np.array([popt[i][param] for i in i_shell])
                    x = np.where(i_shell > 0, self.shell_width*(i_shell-0.5), self.shell_width*(i_shell+0.5))
                    plt.plot(x, y, 'o', markeredgewidth=0.0)
                    plt.xlabel('Distance-to-target [mm]')
                    plt.ylabel('Parameter estimate $\\theta_{{{0}}}$'.format(param+1))

                self.pp.savefig()
                plt.clf()

            self.pp.close()
            del self.pp

        return self

    def _fit_structure(self, p, oar_name):
        """Find correlations between dose and distance-to-target for a single
        organ-at-risk structure. This method splits the dose array into shells
        surrounding the target volume, and also selects only voxels within the
        OAR. The actual fitting is handled by _fit_shell().

        Parameters:
            p: a patient object
            oar_name: string to identify the OAR structure in the patient

        Returns:
            popt: dict of (i: popt_i) pairs, where popt_i is a tuple containing
                the best-fit parameters for shell i
        """
        if self.dose_name not in p.dose_names:
            return

        if oar_name not in p.structure_names:
            return

        if self.target_name not in p.structure_names:
            logging.error('Could not find "%s" in %s' % (self.target_name, p.dicom_dir))
            return

        if self.pp:
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
        min_dist = np.amin(dist_oar)
        max_dist = np.amax(dist_oar)
        i_shell, dist_edges = bin_distance(min_dist, max_dist, self.shell_width)

        # make 2D histogram
        if self.pp:
            dose_edges = np.linspace(0, 1.1, 55, True)
            tmp_dist_edges = np.hstack([dist_edges[1]-self.shell_width, dist_edges[1:-1], dist_edges[-2]+self.shell_width])
            plt.hist2d(dist_oar, dose_oar, bins=(tmp_dist_edges, dose_edges))
            plt.xlabel('Distance-to-target [mm]')
            plt.ylabel('Normalized dose')
            plt.title('%s, %s' % (self.tmp_anon_id, self.tmp_oar_name))
            self.pp.savefig()
            plt.clf()

        popt = {}
        for i, inner, outer in zip(i_shell, dist_edges[:-1], dist_edges[1:]):

            dose_shell = dose_oar[(dist_oar > inner) & (dist_oar <= outer)]

            if dose_shell.size < self.min_shell_size_fit:
                continue

            if self.pp:
                self.tmp_i_shell = i

            popt[i] = self._fit_shell(dose_shell)

        if len(popt) == 0:
            popt = None

        return popt

    def _fit_shell(self, dose):
        """Fit the dose distribution within a shell.

        Parameters:
            dose: ndarray of dose (voxel selection already applied)

        Returns:
            popt: tuple of best-fit parameters
        """
        # initial estimate and bounds of parameter values
        p0 = [np.mean(dose), np.std(dose), ss.skew(dose)]
        p_upper = [2, 1, 10]
        p_lower = [-1, 1e-9, -10]
        for i in range(len(p0)):
            p0[i] = min(p0[i], p_upper[i])
            p0[i] = max(p0[i], p_lower[i])

        # if dose is uniform, there is no distribution to fit
        if np.all(dose == dose[0]):
            return p0

        # bin the data and add empty bins at either end to constrain fit
        min_bin_width = 1e-5
        max_n_bins = 100
        if np.all(dose == dose[0]):
            bin_edges = [dose[0], dose[0]+min_bin_width]
        else:
            min_dose, max_dose = np.amin(dose), np.amax(dose)
            q75, q25 = np.percentile(dose, [75, 25])
            bin_width = 2 * (q75-q25) / np.power(dose.size, 1./3.)
            bin_width = max(bin_width, min_bin_width)
            n_bins = ceil((max_dose-min_dose) / bin_width)
            n_bins = min(n_bins, max_n_bins-2)
            bin_edges = np.linspace(min_dose, max_dose+np.finfo(float).eps, n_bins+1)

        # add empty bins at either end to further constrain fit
        bin_edges = np.insert(bin_edges, 0, 2*bin_edges[0]-bin_edges[1])
        bin_edges = np.append(bin_edges, 2*bin_edges[-1]-bin_edges[-2])

        counts, bin_edges = np.histogram(dose, bin_edges, density=True)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

        # fit data
        try:
            popt, pcov = curve_fit(skew_normal_pdf, bin_centers, counts,
                                   p0=p0, bounds=(p_lower, p_upper))
        except RuntimeError as e:
            # if convergence fails, just use initial parameters
            logging.warning('Fit did not converge - using initial parameters')
            popt = p0

        if self.pp:
            plt.plot(bin_centers, counts, 'k', drawstyle='steps-mid', label='Plan data')
            x = np.linspace(0, 1.1*np.amax(bin_centers), 200)
            plt.plot(x, skew_normal_pdf(x, *popt), 'r', label='Fit')
            plt.tick_params(labelleft='off')
            plt.xlabel('Normalized dose')
            plt.ylabel('Normalized volume')
            plt.legend(loc='upper left')
            plt.title('%s, %s, Shell %i' % (self.tmp_anon_id, self.tmp_oar_name, self.tmp_i_shell))
            s = 'k = {0}\n\n'.format(self.tmp_i_shell)
            s += '\n'.join('$\\theta_{{{0}}}$= {1:.2f}'.format(i+1, phat) for i, phat in enumerate(popt))
            plt.figtext(0.16, 0.55, s, size=20)
            self.pp.savefig()
            plt.clf()

        return popt

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

    def predict_structure(self, p, oar_name, max_size_voxelwise=1000):
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

                p_upper = [2, 1, 10]
                p_lower = [-1, 1e-9, -10]
                for i in range(len(popt)):
                    popt[i] = min(popt[i], p_upper[i])
                    popt[i] = max(popt[i], p_lower[i])

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
                target_dose = np.mean(dose[target_mask])

                dvh_pred = self.predict_structure(p, oar_name, **kwargs)
                dvh_pred.dose_edges *= target_dose
                dvh_plan = p.calculate_dvh(oar_name, self.dose_name,
                    dose_edges=dvh_pred.dose_edges)

                if normalize:
                    dvh_pred.dose_edges /= target_dose
                    dvh_plan.dose_edges /= target_dose

                y_pred.append(getattr(dvh_pred, metric_func)(*metric_args))
                y_true.append(getattr(dvh_plan, metric_func)(*metric_args))

        r2 = r2_score(y_true, y_pred)

        if plot:
            plt.scatter(y_true, y_pred, c='k')
            max_val = 1.1*max(*y_true, *y_pred)
            plt.plot([0, max_val], [0, max_val], 'k:')
            plt.xlabel('Planned ' + metric_label)
            plt.ylabel('Predicted ' + metric_label)
            plt.text(0.1*max_val, 0.9*max_val, 'R2 = {0:.1%}'.format(r2))
            plt.axis('square')
            plt.axis([0, max_val, 0, max_val])

        return r2
