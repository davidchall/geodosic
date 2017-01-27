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

        for i, inner, outer in zip(i_shell, dist_edges[:-1], dist_edges[1:]):

            mask_subvolume = (dist_oar > inner) & (dist_oar <= outer)
            key_subvolume = i

            yield key_subvolume, mask_subvolume

    def _get_subvolume_popt(self, key_subvolume):
        i = key_subvolume

        min_fitted_i = min(self.popt_avg_.keys())
        max_fitted_i = max(self.popt_avg_.keys())

        if i in self.popt_avg_:
            popt = self.popt_avg_[i]
        elif i < min_fitted_i:
            popt = self.popt_avg_[min_fitted_i]
        elif i > max_fitted_i:
            popt = self.popt_avg_[max_fitted_i]
        else:
            dist = self.shell_width * np.where(i > 0, (i-0.5), (i+0.5))

            popt_splines = self.interpolate_popt()
            popt = popt_splines(dist)

        return popt

    def _generate_popt_voxelwise(self, p, oar_name):
        popt_splines = self.interpolate_popt()

        min_fitted_i = min(self.popt_avg_.keys())
        max_fitted_i = max(self.popt_avg_.keys())
        min_fitted_dist = min_fitted_i * self.shell_width
        max_fitted_dist = max_fitted_i * self.shell_width

        mask_oar = p.structure_mask(oar_name, self.grid_name)
        dist = p.distance_to_surface(self.target_name, self.grid_name)
        dist_oar = dist[mask_oar]

        for dist_voxel in dist_oar:
            if dist_voxel < min_fitted_dist:
                popt = self.popt_avg_[min_fitted_i]
            elif dist_voxel > max_fitted_dist:
                popt = self.popt_avg_[max_fitted_i]
            else:
                popt = popt_splines(dist_voxel)

            yield popt

    def interpolate_popt(self, smooth=0.8):
        i_shell = np.array(sorted(list(self.popt_avg_.keys())))
        yp = zip(*(self.popt_avg_[i] for i in i_shell if i in self.popt_avg_))
        dyp = zip(*(self.popt_std_[i] for i in i_shell if i in self.popt_std_))
        x = self.shell_width * np.where(i_shell > 0, (i_shell-0.5), (i_shell+0.5))

        splines = []
        for y, dy in zip(yp, dyp):
            dy = np.clip(dy, 0.1*np.mean(dy), np.amax(dy))
            weights = np.power(dy, -1)
            splines.append(UnivariateSpline(x, y, w=weights, s=smooth))

        return lambda x: [np.clip(spline(x), self.p_lower[i], self.p_upper[i]) for i, spline in enumerate(splines)]

    def plot_params(self, filename, popt_all=None, spline_smooth=0.8):
        pp = PdfPages(filename)

        i_shell = np.array(sorted(list(self.popt_avg_.keys())))
        yp = zip(*(self.popt_avg_[i] for i in i_shell))
        dyp = zip(*(self.popt_std_[i] for i in i_shell))
        x = self.shell_width * np.where(i_shell > 0, (i_shell-0.5), (i_shell+0.5))

        min_i = i_shell.min()
        max_i = i_shell.max()

        min_xs = self.shell_width*(min_i-1) if min_i > 0 else self.shell_width*min_i
        max_xs = self.shell_width*max_i if max_i > 0 else self.shell_width*(max_i+1)
        xs = np.linspace(min_xs, max_xs, 100)

        splines = self.interpolate_popt(smooth=spline_smooth)

        for param, (ys, y, dy) in enumerate(zip(splines(xs), yp, dyp)):

            ys[xs < np.amin(x)] = splines(np.amin(x))[param]
            ys[xs > np.amax(x)] = splines(np.amax(x))[param]

            plt.errorbar(x, y, dy, fmt='ko')
            plt.plot(xs, ys)
            plt.xlabel('Distance-to-target [mm]')
            plt.ylabel('Parameter estimate $\\theta_{{{0}}}$'.format(param+1))

            pp.savefig()
            plt.clf()

        if popt_all:
            for param in range(3):
                for popt in popt_all:
                    i_shell = np.array(sorted(list(popt.keys())))
                    y = np.array([popt[i][param] for i in i_shell])
                    x = self.shell_width * np.where(i_shell > 0, (i_shell-0.5), (i_shell+0.5))
                    plt.plot(x, y, 'o', markeredgewidth=0.0)
                    plt.xlabel('Distance-to-target [mm]')
                    plt.ylabel('Parameter estimate $\\theta_{{{0}}}$'.format(param+1))

                pp.savefig()
                plt.clf()

        pp.close()
