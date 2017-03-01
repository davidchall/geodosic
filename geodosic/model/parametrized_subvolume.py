# standard imports
import logging
from math import ceil

# third-party imports
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
import scipy.stats as ss
from scipy.optimize import curve_fit
from scipy.integrate import quad

# project imports
from ..utils import initialize_attributes
from ..dvh import DVH


def skew_normal_pdf(x, e=0, w=1, a=0):
    """PDF for skew-normal distribution.

    Parameters:
        e: location
        w: scale
        a: shape
    """
    t = (x-e) / w
    return 2 / w * ss.norm.pdf(t) * ss.norm.cdf(a*t)


class BaseParametrizedSubvolumeModel(BaseEstimator, RegressorMixin):

    @initialize_attributes
    def __init__(self, dose_name=None, oar_names=None, target_name=None,
                 grid_name=None,
                 n_jobs=1,
                 normalize_to_prescribed_dose=False, max_prescribed_dose=0,
                 min_subvolume_size_for_fit=10, min_structures_for_fit=2):
        pass

    def clip_params(self, p0):
        for i in range(len(p0)):
            p0[i] = min(p0[i], self.p_upper[i])
            p0[i] = max(p0[i], self.p_lower[i])
        return p0

    def fit(self, X, y=None, plot_params_path=None):
        """Train the model.

        Parameters:
            X: the training cohort (a list of patient objects)
        """

        # validate model parameters
        assert self.dose_name is not None
        assert self.oar_names is not None
        assert self.target_name is not None
        assert self.min_subvolume_size_for_fit >= 0
        assert self.min_structures_for_fit >= 0
        assert self.n_jobs >= 0

        if not self.normalize_to_prescribed_dose:
            if self.max_prescribed_dose <= 0:
                raise AssertionError('Must set max_prescribed_dose (used in parameter bounds)')

        if self.grid_name is None:
            self.grid_name = self.dose_name

        if isinstance(self.oar_names, str):
            self.oar_names = [self.oar_names]

        # parameter bounds
        self.p_upper = [2, 1, 10]
        self.p_lower = [0, 1e-9, -10]
        if not self.normalize_to_prescribed_dose:
            self.p_upper[0] *= self.max_prescribed_dose
            self.p_lower[0] *= self.max_prescribed_dose
            self.p_upper[1] *= self.max_prescribed_dose
            self.p_lower[1] *= self.max_prescribed_dose

        # fit subvolume DVHs for all structures
        self.failed_converge = 0
        self.attempt_converge = 0

        popt_all = []
        for p in X:
            for oar_name in self.oar_names:
                popt_all.append(self._fit_structure(p, oar_name))
        popt_all = [popt for popt in popt_all if popt is not None]

        if self.failed_converge > 0:
            logging.warning('{0}/{1} fits failed to converge (used mean, std, skew instead)'.format(self.failed_converge, self.attempt_converge))
        del self.attempt_converge
        del self.failed_converge

        # compile list of subvolumes found in training set
        subvolume_keys = set(key for popt in popt_all for key in popt.keys())

        # average the subvolume parameters across structures
        self.popt_avg_, self.popt_std_ = {}, {}
        for key in subvolume_keys:

            popt_all_subvol = np.array([popt[key] for popt in popt_all if key in popt])
            n_found_structures = popt_all_subvol.shape[0]

            if n_found_structures >= self.min_structures_for_fit:
                self.popt_avg_[key] = np.mean(popt_all_subvol, axis=0)
                self.popt_std_[key] = np.std(popt_all_subvol, axis=0)

        if plot_params_path:
            self.plot_params(plot_params_path, popt_all)

        return self

    def _fit_structure(self, p, oar_name):
        """Find correlations between dose and distance-to-target for a single
        organ-at-risk structure. The dose array is split into subvolumes
        surrounding the target volume, and also selects only voxels within the
        OAR. The actual fitting is handled by _fit_subvolume().

        Parameters:
            p: a patient object
            oar_name: string to identify the OAR structure in the patient

        Returns:
            popt: dict of (name: popt) pairs, where popt is a tuple containing
                the best-fit parameters for the subvolume called name
        """
        if self.dose_name not in p.dose_names:
            return

        if oar_name not in p.structure_names:
            return

        if self.target_name not in p.structure_names:
            logging.error('Could not find "%s" in %s' % (self.target_name, p.dicom_dir))
            return

        dose = p.dose_array(self.dose_name, self.grid_name)
        mask_oar = p.structure_mask(oar_name, self.grid_name)
        dose_oar = dose[mask_oar]

        if self.normalize_to_prescribed_dose:
            dose_oar /= p.prescribed_doses[self.dose_name]

        subvolumes = ((k, m) for k, m in self._generate_subvolume_masks(p, oar_name) if np.count_nonzero(m) >= self.min_subvolume_size_for_fit)

        if self.n_jobs > 0:
            from joblib import Parallel, delayed
            popt_list = Parallel(n_jobs=self.n_jobs)(delayed(self._fit_subvolume_wrapper)(k, dose_oar[m]) for k, m in subvolumes)
            popt = {k: v for k, v in popt_list}
        else:
            popt = {k: self._fit_subvolume(dose_oar[m]) for k, m in subvolumes}

        if len(popt) == 0:
            popt = None

        return popt

    def _fit_subvolume_wrapper(self, key, dose):
        return key, self._fit_subvolume(dose)

    def _fit_subvolume(self, dose):
        """Fit the dose distribution within a subvolume.

        Parameters:
            dose: ndarray of dose (voxel selection already applied)

        Returns:
            popt: tuple of best-fit parameters
        """
        # initial estimate and bounds of parameter values
        p0 = [np.mean(dose), np.std(dose), ss.skew(dose)]
        p0 = self.clip_params(p0)

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

        self.attempt_converge += 1

        # fit data
        try:
            popt, pcov = curve_fit(skew_normal_pdf, bin_centers, counts,
                                   p0=p0, bounds=(self.p_lower, self.p_upper))
            popt = self.clip_params(popt)
        except RuntimeError as e:
            # if convergence fails, just use initial parameters
            self.failed_converge += 1
            popt = p0

        except Exception as e:
            print(dose.size)
            print(bin_edges.size)
            print(counts.size)
            print(dose)
            print(bin_edges)
            print(counts)
            raise e

        return popt

    def predict(self, X, *args, **kwargs):
        if isinstance(self.oar_names, str):
            self.oar_names = [self.oar_names]

        pred = []
        for p in X:
            for oar_name in self.oar_names:
                pred.append(self.predict_structure(p, oar_name, *args, **kwargs))
        return pred

    def predict_structure(self, p, oar_name, dose_edges=None,
                          max_size_voxelwise=100, vol_tol=1e-2):
        if oar_name not in p.structure_names:
            return
        if self.target_name not in p.structure_names:
            return

        # Set up DVH binning
        if dose_edges is None:
            dose_edges = np.linspace(0., 1.2, 120)
            if self.normalize_to_prescribed_dose:
                dose_edges *= p.prescribed_doses[self.dose_name]
            else:
                dose_edges *= self.max_prescribed_dose

        mask_oar = p.structure_mask(oar_name, self.grid_name)
        size_oar = np.count_nonzero(mask_oar)

        # normalize dose scale
        if self.normalize_to_prescribed_dose:
            dose_edges /= p.prescribed_doses[self.dose_name]

        # accumulate dose histogram
        dose_struct = np.zeros_like(dose_edges[:-1])

        if size_oar > max_size_voxelwise:
            for key_subvolume, mask_subvolume in self._generate_subvolume_masks(p, oar_name):

                size_subvolume = np.count_nonzero(mask_subvolume)
                if size_subvolume == 0:
                    continue

                popt = self._get_subvolume_popt(key_subvolume)
                dose_struct += size_subvolume * self._predict_subvolume(popt, dose_edges, vol_tol)

        else:  # voxel-wise
            for popt in self._generate_popt_voxelwise(p, oar_name):
                dose_struct += self._predict_subvolume(popt, dose_edges, vol_tol)

        # restore dose scale
        if self.normalize_to_prescribed_dose:
            dose_edges *= p.prescribed_doses[self.dose_name]

        # parametric fits yield long tails, but DVH requires last bin is zero
        if dose_struct[-1] > 0:
            overflow_volume = dose_struct[-1] / size_oar
            if overflow_volume > vol_tol:
                logging.warning('Predicted dose exceeds DVH domain ({0:.1%} of volume)'.format(overflow_volume))
            dose_struct[-2] += dose_struct[-1]
            dose_struct[-1] = 0

        return DVH(dose_struct, dose_edges, dDVH=True)

    def _predict_subvolume(self, popt, dose_edges, vol_tol):
        """Predict the dose histogram for a subvolume. That is, integrate the
        skew-normal distribution across histogram bins.

        Parameters:
            popt: skew-normal parameters
            dose_edges: histogram bin edges
            vol_tol: fractional volume tolerance (used to reduce computations
                     and identify regions exceeding the maximum dose)

        Returns:
            dose_subvolume: dose histogram
        """
        popt = tuple(popt)
        dose_subvolume = np.zeros_like(dose_edges[:-1])

        if popt[0] < dose_edges[1] and skew_normal_pdf(dose_edges[1], *popt) < vol_tol:  # all dose in first bin
            dose_subvolume[0] = 1
        elif popt[1] > 2*dose_edges[1]:
            dose_centers = 0.5 * (dose_edges[1:] + dose_edges[:-1])
            dose_subvolume = skew_normal_pdf(dose_centers, *popt)
        else:
            for dose_bin, (d_lower, d_upper) in enumerate(zip(dose_edges[:-1], dose_edges[1:])):
                integral = quad(skew_normal_pdf, d_lower, d_upper, args=popt)
                dose_subvolume[dose_bin] = integral[0]

        # identify significant overflow and integrate
        if popt[0] > dose_edges[-1] or skew_normal_pdf(dose_edges[-1], *popt) > vol_tol:
            d_upper = popt[0] + 10*popt[1]
            integral = quad(skew_normal_pdf, dose_edges[-1], d_upper, args=popt)
            dose_subvolume[-1] += integral[0]

        if np.sum(dose_subvolume) == 0:
            logging.warning('Integrated PDF to zero! popt=', popt)

        # normalize histogram to probability density function
        dose_subvolume /= np.sum(dose_subvolume)

        return dose_subvolume

    def generate_validation_dvhs(self, X, n_dose_bins=100, **kwargs):
        """Generates predicted and planned DVHs for model validation.

        Args:
            X: validation cohort of Patient objects
            n_dose_bins: resolution of DVH
            kwargs: passed to model.predict_structure
        """
        for p in X:
            if self.dose_name not in p.dose_names:
                continue
            if self.target_name not in p.structure_names:
                continue

            for oar_name in self.oar_names:
                if oar_name not in p.structure_names:
                    continue

                # choose appropriate binning for DVHs
                dose = p.dose_array(self.dose_name, self.dose_name)
                target_mask = p.structure_mask(self.target_name, self.dose_name)
                max_target_dose_plan = np.max(dose[target_mask])

                if self.normalize_to_prescribed_dose:
                    max_target_dose_pred = p.prescribed_doses[self.dose_name]
                else:
                    max_target_dose_pred = self.max_prescribed_dose

                max_dvh_dose = 1.2*max(max_target_dose_plan, max_target_dose_pred)
                dose_edges = np.linspace(0, max_dvh_dose, n_dose_bins)

                dvh_pred = self.predict_structure(p, oar_name, dose_edges=dose_edges, **kwargs)
                dvh_plan = p.calculate_dvh(oar_name, self.dose_name, dose_edges=dose_edges)

                yield p, dvh_pred, dvh_plan

    def _generate_subvolume_masks(self, p, oar_name):
        """Generate subvolume masks with unique keys.

        Parameters:
            p: Patient object
            oar_name: string to identify the OAR structure in the patient

        Returns:
            key_subvolume: hashable ID
            mask_subvolume: has shape of 1D OAR array (NOT 3D grid array)
        """
        raise NotImplementedError

    def _get_subvolume_popt(self, key_subvolume):
        """Get skew-normal parameters for a specific subvolume.

        Parameters:
            key_subvolume: hashable ID

        Returns:
            popt: skew-normal parameters
        """
        raise NotImplementedError

    def _generate_popt_voxelwise(self, p, oar_name):
        """Generate skew-normal parameters for each voxel in an OAR structure.

        Parameters:
            p: Patient object
            oar_name: string to identify the OAR structure in the patient

        Returns:
            popt: skew-normal parameters
        """
        raise NotImplementedError

    def plot_params(self, popt_all=None):
        raise NotImplementedError
