# third-party imports
import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# project imports
from .parametrized_subvolume import BaseParametrizedSubvolumeModel, initialize_attributes
from ..geometry import bin_distance


class ShellModel(BaseParametrizedSubvolumeModel):

    @initialize_attributes
    def __init__(self, dose_name=None, oar_names=None, target_name=None,
                 grid_name=None,
                 n_jobs=1,
                 normalize_to_prescribed_dose=False, max_prescribed_dose=0,
                 min_subvolume_size_for_fit=10, min_structures_for_fit=2,
                 shell_width=3.0):
        pass

    def fit(self, *args, **kwargs):
        assert self.shell_width > 0
        return super(ShellModel, self).fit(*args, **kwargs)

    def _generate_subvolume_masks(self, p, oar_name):
        oar_mask = p.structure_mask(oar_name, self.grid_name)
        dist = p.distance_to_surface(self.target_name, self.grid_name)

        dist_oar = dist[oar_mask]
        min_dist = np.amin(dist_oar)
        max_dist = np.amax(dist_oar)

        i_shell, dist_edges = bin_distance(min_dist, max_dist, self.shell_width)

        key_other = None  # dummy key_other for this most simple shell model
        for i, inner, outer in zip(i_shell, dist_edges[:-1], dist_edges[1:]):

            mask_subvolume = (dist_oar > inner) & (dist_oar <= outer)
            key_subvolume = (i, key_other)

            yield key_subvolume, mask_subvolume

    def _get_subvolume_popt(self, key_subvolume):
        (i_sv, o_sv) = key_subvolume

        fitted_i = [i for i, o in self.popt_avg_.keys() if o == o_sv]
        min_fitted_i = min(fitted_i)
        max_fitted_i = max(fitted_i)

        if i_sv in self.popt_avg_:
            popt = self.popt_avg_[(i_sv, o_sv)]
        elif i_sv < min_fitted_i:
            popt = self.popt_avg_[(min_fitted_i, o_sv)]
        elif i_sv > max_fitted_i:
            popt = self.popt_avg_[(max_fitted_i, o_sv)]
        else:
            dist = self.shell_width * np.where(i_sv > 0, (i_sv-0.5), (i_sv+0.5))

            popt_splines = self.interpolate_popt(o_sv)
            popt = popt_splines(dist)

        return popt

    def interpolate_popt(self, key_other, smooth=0.8):
        i_shell = np.array(sorted(list(i for i, o in self.popt_avg_.keys() if o == key_other)))
        yp = zip(*(self.popt_avg_[(i, key_other)] for i in i_shell))
        dyp = zip(*(self.popt_std_[(i, key_other)] for i in i_shell))
        x = self.shell_width * np.where(i_shell > 0, (i_shell-0.5), (i_shell+0.5))

        splines = []
        for y, dy in zip(yp, dyp):
            dy = np.clip(dy, 0.1*np.mean(dy), np.amax(dy))
            weights = np.power(dy, -1)
            splines.append(UnivariateSpline(x, y, w=weights, s=smooth))

        return lambda x: [np.clip(spline(x), self.p_lower[i], self.p_upper[i]) for i, spline in enumerate(splines)]

    def plot_params(self, filename, popt_all=None, spline_smooth=0.8):
        pp = PdfPages(filename)

        key_other_options = set(o for i, o in self.popt_avg_.keys())

        for key_other in key_other_options:

            i_shell = np.array(sorted(list(i for i, o in self.popt_avg_.keys() if o == key_other)))
            x = self.shell_width * np.where(i_shell > 0, (i_shell-0.5), (i_shell+0.5))
            yp = zip(*(self.popt_avg_[(i, key_other)] for i in i_shell))
            dyp = zip(*(self.popt_std_[(i, key_other)] for i in i_shell))

            min_i = i_shell.min()
            max_i = i_shell.max()

            min_xs = self.shell_width*(min_i-1) if min_i > 0 else self.shell_width*min_i
            max_xs = self.shell_width*max_i if max_i > 0 else self.shell_width*(max_i+1)
            xs = np.linspace(min_xs, max_xs, 100)

            splines = self.interpolate_popt(key_other, smooth=spline_smooth)

            for param, (ys, y, dy) in enumerate(zip(splines(xs), yp, dyp)):

                ys[xs < np.amin(x)] = splines(np.amin(x))[param]
                ys[xs > np.amax(x)] = splines(np.amax(x))[param]

                plt.errorbar(x, y, dy, fmt='ko')
                plt.plot(xs, ys)
                plt.xlabel('Distance-to-target [mm]')
                plt.ylabel('Parameter estimate $\\theta_{{{0}}}$'.format(param+1))
                if key_other:
                    plt.title(key_other)

                pp.savefig()
                plt.clf()

        if popt_all:
            for key_other in key_other_options:
                for param in range(3):
                    for popt in popt_all:
                        i_shell = np.array(sorted(list(i for i, o in popt.keys() if o == key_other)))
                        y = np.array([popt[(i, key_other)][param] for i in i_shell])
                        x = self.shell_width * np.where(i_shell > 0, (i_shell-0.5), (i_shell+0.5))
                        plt.plot(x, y, 'o', markeredgewidth=0.0)
                        plt.xlabel('Distance-to-target [mm]')
                        plt.ylabel('Parameter estimate $\\theta_{{{0}}}$'.format(param+1))
                        if key_other:
                            plt.title(key_other)

                    pp.savefig()
                    plt.clf()

        pp.close()


class SimpleShellModel(ShellModel):
    """SimpleShellModel uses subvolumes:
        - shells surrounding target

    This model supports voxel-wise prediction.
    """

    @initialize_attributes
    def __init__(self, dose_name=None, oar_names=None, target_name=None,
                 grid_name=None,
                 n_jobs=1,
                 normalize_to_prescribed_dose=False, max_prescribed_dose=0,
                 min_subvolume_size_for_fit=10, min_structures_for_fit=2,
                 shell_width=3.0):
        pass

    def _generate_popt_voxelwise(self, p, oar_name):
        o_sv = None

        popt_splines = self.interpolate_popt(o_sv)

        fitted_i = [i for i, o in self.popt_avg_.keys() if o == o_sv]
        min_fitted_i = min(fitted_i)
        max_fitted_i = max(fitted_i)
        min_fitted_dist = min_fitted_i * self.shell_width
        max_fitted_dist = max_fitted_i * self.shell_width

        mask_oar = p.structure_mask(oar_name, self.grid_name)
        dist = p.distance_to_surface(self.target_name, self.grid_name)
        dist_oar = dist[mask_oar]

        for dist_voxel in dist_oar:
            if dist_voxel < min_fitted_dist:
                popt = self.popt_avg_[(min_fitted_i, o_sv)]
            elif dist_voxel > max_fitted_dist:
                popt = self.popt_avg_[(max_fitted_i, o_sv)]
            else:
                popt = popt_splines(dist_voxel)

            yield popt


class CoplanarShellModel(ShellModel):
    """CoplanarShellModel uses subvolumes:
        - shells surrounding target
        - in-field/out-of-field components based upon target z coordinates
    """

    @initialize_attributes
    def __init__(self, dose_name=None, oar_names=None, target_name=None,
                 grid_name=None,
                 n_jobs=1,
                 normalize_to_prescribed_dose=False, max_prescribed_dose=0,
                 min_subvolume_size_for_fit=10, min_structures_for_fit=2,
                 shell_width=3.0, penumbra_width=1.0):
        pass

    def fit(self, *args, **kwargs):
        assert self.penumbra_width >= 0
        return super(CoplanarShellModel, self).fit(*args, **kwargs)

    def _generate_subvolume_masks(self, p, oar_name):
        mask_target = p.structure_mask(self.target_name, self.grid_name)
        mask_oar = p.structure_mask(oar_name, self.grid_name)

        # z-position needed to find in-field and out-of-field components
        grid = p.grid_vectors(self.grid_name)
        _, _, z = np.meshgrid(*grid, indexing='ij')
        z_target = z[mask_target]
        z_oar = z[mask_oar]
        z_threshold_sup = z_target.max() + self.penumbra_width
        z_threshold_inf = z_target.min() - self.penumbra_width

        mask_infield = (z_oar <= z_threshold_sup) & (z_oar >= z_threshold_inf)

        # distance-to-target needed to construct shells
        dist = p.distance_to_surface(self.target_name, self.grid_name)
        dist_oar = dist[mask_oar]

        min_dist = np.amin(dist_oar)
        max_dist = np.amax(dist_oar)
        i_shell, dist_edges = bin_distance(min_dist, max_dist, self.shell_width)

        for i, inner, outer in zip(i_shell, dist_edges[:-1], dist_edges[1:]):

            mask_shell = (dist_oar > inner) & (dist_oar <= outer)

            # in-field
            mask_subvolume = mask_shell & mask_infield
            key_subvolume = (i, 'In-field')
            if np.count_nonzero(mask_subvolume):
                yield key_subvolume, mask_subvolume

            # out-of-field
            mask_subvolume = mask_shell & ~mask_infield
            key_subvolume = (i, 'Out-of-field')
            if np.count_nonzero(mask_subvolume):
                yield key_subvolume, mask_subvolume
