# standard imports
import logging

# third-party imports
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.stats import norm

# project imports
from ..utils import initialize_attributes
from ..dvh import DVH


class AverageSubvolumeModelBase(BaseEstimator, RegressorMixin):

    @initialize_attributes
    def __init__(self, dose_name=None, oar_names=None, target_name=None,
                 grid_name=None,
                 distn=norm, p_lower=None, p_upper=None,
                 n_jobs=1,
                 normalize_to_prescribed_dose=False, max_prescribed_dose=0,
                 min_subvolume_size_for_fit=10, min_structures_for_fit=2):
        pass

    def clip_params(self, params):
        for i, _ in enumerate(params):
            params[i] = max(params[i], self.p_lower[i])
            params[i] = min(params[i], self.p_upper[i])
        return params

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
        if not self.p_lower:
            self.p_lower = [-np.inf] * (self.distn.numargs + 2)
            self.p_lower[-1] = 1e-9
        if not self.p_upper:
            self.p_upper = [+np.inf] * (self.distn.numargs + 2)

        # fit subvolume DVHs for all structures
        popt_all = []
        for p in X:
            for oar_name in self.oar_names:
                popt_all.append(self._fit_structure(p, oar_name))
        popt_all = [popt for popt in popt_all if popt is not None]

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

        # if dose is uniform, there is no distribution to fit
        if np.all(dose == dose[0]):
            popt = [0] * (self.distn.numargs + 2)
            popt[-2] = dose[0]
        else:
            popt = self.distn.fit(dose)

        popt = self.clip_params(list(popt))

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
            if self.normalize_to_prescribed_dose:
                max_dose_expected = p.prescribed_doses[self.dose_name]
            else:
                max_dose_expected = self.max_prescribed_dose
            max_dose_dvh = 1.2 * max_dose_expected
            dose_edges = DVH.choose_dose_edges(max_dose_dvh)

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
                dose_struct += size_subvolume * self._discretize_probability(self.distn, dose_edges, popt)

        else:  # voxel-wise
            for popt in self._generate_popt_voxelwise(p, oar_name):
                dose_struct += self._discretize_probability(self.distn, dose_edges, popt)

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

    @staticmethod
    def _discretize_probability(distn, x_edges, params):
        """Integrates a probability distribution across a set of intervals.

        Note: underflow and overflow are folded into first and last intervals.

        Parameters:
            distn: continuous probability distribution (implements cdf method)
            x_edges: set of interval edges
            params: distribution parameters

        Returns:
            prob: probability of each interval
        """
        cdf = distn.cdf(x_edges, *params)

        underflow = cdf[0]
        overflow = 1 - cdf[-1]

        prob = np.diff(cdf)
        prob[0] += underflow
        prob[-1] += overflow

        # FPE can cause negative probabilities
        prob = np.maximum(prob, 0)

        # FPE can de-normalize probability
        prob /= np.sum(prob)

        return prob

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
                max_dose_target = dose[target_mask].max()

                if self.normalize_to_prescribed_dose:
                    max_dose_expected = p.prescribed_doses[self.dose_name]
                else:
                    max_dose_expected = self.max_prescribed_dose

                max_dose_dvh = max(max_dose_target, 1.2*max_dose_expected)
                dose_edges = DVH.choose_dose_edges(max_dose_dvh, n_dose_bins)

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
