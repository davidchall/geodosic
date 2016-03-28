# standard imports
import inspect
from functools import wraps

# third-party imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy.stats as ss
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# project imports
from .geometry import distance_to_surface, bin_distance
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
                 shell_width=3.0):
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
        if self.dose_name not in p.dose_names:
            return

        if self.oar_name not in p.structure_names:
            return

        if self.pp:
            self.iPatient += 1

        target_mask = p.structure_mask(self.target_name, self.dose_name)
        oar_mask = p.structure_mask(self.oar_name, self.dose_name)
        dist = p.distance_to_surface(self.target_name, self.dose_name)

        dose = p.dose_array(self.dose_name)
        target_dose = np.mean(dose[target_mask])
        dose /= target_dose

        dose_oar = dose[oar_mask]
        dist_oar = dist[oar_mask]
        min_dist = np.amin(dist_oar)
        max_dist = np.amax(dist_oar)
        i_shell, dist_edges = bin_distance(min_dist, max_dist, self.shell_width)

        popt = {}
        for i, inner, outer in zip(i_shell, dist_edges[:-1], dist_edges[1:]):

            dose_shell = dose_oar[(dist_oar > inner) & (dist_oar <= outer)]

            if dose_shell.size < 4:
                continue

            if self.pp:
                self.iShell = i

            popt[i] = self._fit_shell(dose_shell)

        return popt

    def _fit_shell(self, dose):
        """Fit the dose distribution within a shell.

        Parameters:
            dose: ndarray of dose (voxel selection already applied)

        Returns:
            popt: tuple of best-fit parameters
        """
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
            bin_edges = np.linspace(min_dose, max_dose, n_bins+1)

        # add empty bins at either end to further constrain fit
        bin_edges = np.insert(bin_edges, 0, 2*bin_edges[0]-bin_edges[1])
        bin_edges = np.append(bin_edges, 2*bin_edges[-1]-bin_edges[-2])

        counts, bin_edges = np.histogram(dose, bin_edges, density=True)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

        # fit data
        p0 = (np.mean(dose), np.std(dose), ss.skew(dose))
        p_upper = (2, 1, 1000)
        p_lower = (-1, 0, -1000)

        try:
            popt, pcov = curve_fit(skew_normal_pdf, bin_centers, counts,
                                   p0=p0, bounds=(p_lower, p_upper))
        except RuntimeError as e:
            # if convergence fails, just use initial parameters
            popt = p0

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

    def predict(self, X):
        return [self.predict_patient(p) for p in X]

    def predict_patient(self, p):
        if self.oar_name not in p.structure_names:
            return
        if self.target_name not in p.structure_names:
            return

        oar_mask = p.structure_mask(self.oar_name, 'default')
        dist = p.distance_to_surface(self.target_name, 'default')
        dist_oar = dist[oar_mask]

        min_dist = np.amin(dist_oar)
        max_dist = np.amax(dist_oar)
        i_shell, dist_edges = bin_distance(min_dist, max_dist, self.shell_width)

        dose_edges = np.linspace(0., 1.2, 120)
        dose_centers = 0.5 * (dose_edges[1:] + dose_edges[:-1])
        dose_counts = np.zeros_like(dose_centers)

        min_fitted_i = min(self.popt_avg_.keys())
        max_fitted_i = max(self.popt_avg_.keys())

        for i, inner, outer in zip(i_shell, dist_edges[:-1], dist_edges[1:]):

            if i < min_fitted_i:
                popt = self.popt_avg_[min_fitted_i]
            elif i > max_fitted_i:
                popt = self.popt_avg_[max_fitted_i]
            else:
                popt = self.popt_avg_[i]

            shell = (dist_oar > inner) & (dist_oar < outer)
            n_voxels_shell = np.count_nonzero(shell)
            dose_shell = skew_normal_pdf(dose_centers, *popt)
            dose_counts += n_voxels_shell * dose_shell

        # parametric fits yield long tails, but DVH requires last bin is zero
        dose_counts[-1] = 0
        return DVH(dose_counts, dose_edges, dDVH=True)

    def score(self, X, y=None, normalize=False, plot=False):
        y_true, y_pred = [], []  # metric evaluated
        for p in X:
            dose = p.dose_array(self.dose_name)
            target_mask = p.structure_mask(self.target_name, self.dose_name)
            target_dose = np.mean(dose[target_mask])

            dvh_pred = self.predict_patient(p)
            dvh_pred.dose_edges *= target_dose
            dvh_plan = p.calculate_dvh(self.oar_name, self.dose_name)

            if normalize:
                dvh_pred.dose_edges /= target_dose
                dvh_plan.dose_edges /= target_dose

            y_pred.append(dvh_pred.mean())
            y_true.append(dvh_plan.mean())

        r2 = r2_score(y_true, y_pred)

        if plot:
            plt.scatter(y_true, y_pred)
            max_val = 1.1*max(*y_true, *y_pred)
            plt.plot([0, max_val], [0, max_val], ':')
            plt.xlabel('Planned metric')
            plt.ylabel('Predicted metric')
            plt.text(0.1*max_val, 0.9*max_val, 'R^2 = {0:.1%}'.format(r2))
            plt.axis('square')
            plt.axis([0, max_val, 0, max_val])

        return r2
