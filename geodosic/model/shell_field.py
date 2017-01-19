# third-party imports
import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# project imports
from .parametrized_subvolume import BaseParametrizedSubvolumeModel
from ..geometry import bin_distance


class ShellFieldModel(BaseParametrizedSubvolumeModel):

    def __init__(self, shell_width=3.0, penumbra_width=1.0, *args, **kwargs):
        super(ShellFieldModel, self).__init__(*args, **kwargs)
        self.shell_width = shell_width
        self.penumbra_width = penumbra_width

    def fit(self, *args, **kwargs):
        assert self.shell_width > 0
        assert self.penumbra_width > 0
        return super(ShellFieldModel, self).fit(*args, **kwargs)

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
            key_subvolume = (i, True)
            if np.count_nonzero(mask_subvolume):
                yield key_subvolume, mask_subvolume

            # out-of-field
            mask_subvolume = mask_shell & ~mask_infield
            key_subvolume = (i, False)
            if np.count_nonzero(mask_subvolume):
                yield key_subvolume, mask_subvolume

    def _get_subvolume_popt(self, key_subvolume):
        (i, is_infield) = key_subvolume

        min_fitted_i = min(k[0] for k in self.popt_avg_.keys())
        max_fitted_i = max(k[0] for k in self.popt_avg_.keys())

        if i in self.popt_avg_:
            popt = self.popt_avg_[(i, is_infield)]
        elif i < min_fitted_i:
            popt = self.popt_avg_[(min_fitted_i, is_infield)]
        elif i > max_fitted_i:
            popt = self.popt_avg_[(max_fitted_i, is_infield)]
        else:
            dist = self.shell_width * np.where(i > 0, (i-0.5), (i+0.5))

            popt_splines = self.interpolate_popt(is_infield)
            popt = popt_splines(dist)

        return popt

    def interpolate_popt(self, is_infield, smooth=0.8):
        i_shell = np.array(sorted(list(k[0] for k in self.popt_avg_.keys() if k[1] == is_infield)))
        yp = zip(*(self.popt_avg_[(i, is_infield)] for i in i_shell))
        dyp = zip(*(self.popt_std_[(i, is_infield)] for i in i_shell))
        x = self.shell_width * np.where(i_shell > 0, (i_shell-0.5), (i_shell+0.5))

        splines = []
        for y, dy in zip(yp, dyp):
            dy = np.clip(dy, 0.1*np.mean(dy), np.amax(dy))
            weights = np.power(dy, -1)
            splines.append(UnivariateSpline(x, y, w=weights, s=smooth))

        return lambda x: [np.clip(spline(x), self.p_lower[i], self.p_upper[i]) for i, spline in enumerate(splines)]

    def plot_params(self, filename, popt_all=None, spline_smooth=0.8):
        pp = PdfPages(filename)

        infield_options = [True, False]

        for is_infield in infield_options:

            i_shell = np.array(sorted(list(k[0] for k in self.popt_avg_.keys() if k[1] == is_infield)))
            yp = zip(*(self.popt_avg_[(i, is_infield)] for i in i_shell))
            dyp = zip(*(self.popt_std_[(i, is_infield)] for i in i_shell))
            x = self.shell_width * np.where(i_shell > 0, (i_shell-0.5), (i_shell+0.5))

            min_i = i_shell.min()
            max_i = i_shell.max()

            min_xs = self.shell_width*(min_i-1) if min_i > 0 else self.shell_width*min_i
            max_xs = self.shell_width*max_i if max_i > 0 else self.shell_width*(max_i+1)
            xs = np.linspace(min_xs, max_xs, 100)

            splines = self.interpolate_popt(is_infield, smooth=spline_smooth)

            for param, (ys, y, dy) in enumerate(zip(splines(xs), yp, dyp)):

                ys[xs < np.amin(x)] = splines(np.amin(x))[param]
                ys[xs > np.amax(x)] = splines(np.amax(x))[param]

                plt.errorbar(x, y, dy, fmt='ko')
                plt.plot(xs, ys)
                plt.xlabel('Distance-to-target [mm]')
                plt.ylabel('Parameter estimate $\\theta_{{{0}}}$'.format(param+1))
                plt.title('In-field' if is_infield else 'Out-of-field')

                pp.savefig()
                plt.clf()

            if popt_all:
                for param in range(3):
                    for popt in popt_all:
                        i_shell = np.array(sorted(list(k[0] for k in popt.keys() if k[1] == is_infield)))
                        y = np.array([popt[(i, is_infield)][param] for i in i_shell])
                        x = self.shell_width * np.where(i_shell > 0, (i_shell-0.5), (i_shell+0.5))
                        plt.plot(x, y, 'o', markeredgewidth=0.0)
                        plt.xlabel('Distance-to-target [mm]')
                        plt.ylabel('Parameter estimate $\\theta_{{{0}}}$'.format(param+1))
                        plt.title('In-field' if is_infield else 'Out-of-field')

                    pp.savefig()
                    plt.clf()

        pp.close()
