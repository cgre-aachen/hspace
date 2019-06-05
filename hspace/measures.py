"""
    This file is part of hspace.

    hspace is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    gempy is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with hspace.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import matplotlib.pyplot as plt
try:
    import joblib # for parallel execution
except ImportError:
    pass

try:
    import pymc

except ImportError:
    print("pymc is not installed - Bayesian entropy estimation will not work.")



# TODO at end: upload to pip, create setup.py (and conda install?) See also Tools -> Create setup.py!

# TODO: include both: a conventional calculation on the basis of probability fields,
# (as these may also be of interest), and the fast implementation based on sorting, for efficiency only


def joint_entropy(data_array, pos=None, **kwds):
    """Joint entropy between multiple points

    The data_array contains the multi-dimensional input data set. By default, the first axies
    should contain the realisations, and subsequent axes the dimensions within each realisation.

    Args:
        data_array (numpy.array): n-D array with input data
        pos (numpy.array): positions of points (n-1 -D) (in 1-D case: not required!)

    Attributes:
        extent(list):  [x_min, x_max, y_min, y_max, z_min, z_max]
        resolution ((Optional[list])): [nx, ny, nz]
        Foliations(pandas.core.frame.DataFrame): Pandas data frame with the foliations data
        Interfaces(pandas.core.frame.DataFrame): Pandas data frame with the interfaces data
        series(pandas.core.frame.DataFrame): Pandas data frame which contains every formation within each series

    """
    # TODO: implement in a way that it works on 1-D to 3-D data sets!


    # TODO: include check: first axis: relaisations; subsequent axes: dimensions within each
    # realisation

    # check possible implementation: if array is 1-D, then revert to bincount/ np.unique method
    # for faster implementation!


    # step 1: sort

    if len(data_array.shape) == 1: # 0-D case, no position argument required

        im_sort = np.sort(data_array) # , axis=0)

    # step 2: find change points/ switch points:
        switches = np.where(np.not_equal(im_sort[1:], im_sort[:-1]))[0]

    elif len(data_array.shape) == 2: # 1-D case, requires iteration over positions
        # im_sort = np.sort(data_array[:,pos], axis=0)
        # im_sort = data_array[np.argsort(data_array[:,0])]
        # new test as sorting did not return correct results:

        sub_array = np.empty((data_array.shape[0], len(pos)))
        i = 0
        for p1 in pos:
            sub_array[:, i] = data_array[:, p1]
            i += 1

        # now: sort:
        for i in range(len(pos)):
            sub_array = sub_array[sub_array[:, i].argsort(kind='mergesort')]

        switches = np.where(np.not_equal(sub_array[1:], sub_array[:-1]).any(axis=1))[0]

        # for p in pos:
        #     # data_array = data_array[data_array[:, p].argsort(kind='mergesort')]
        #     data_array = data_array[data_array[:, p].argsort(kind='mergesort')]
        #
        # # extract elements
        # # data_array = data_array[:,p]
        # switches = np.where(np.not_equal(data_array[1:], data_array[:-1]).any(axis=1))[0]
        # print(switches)

    elif len(data_array.shape) == 3: # 2-D case, requires iteration over positions
        # extract values:
        sub_array = np.empty((data_array.shape[0],len(pos)))
        i = 0
        for p1, p2 in pos:
            sub_array[:,i] = data_array[:,p1,p2]
            i += 1

        # now: sort:
        for i in range(len(pos)):
            sub_array = sub_array[sub_array[:, i].argsort(kind='mergesort')]

        switches = np.where(np.not_equal(sub_array[1:], sub_array[:-1]).any(axis=1))[0]
        #
        # for p1,p2 in pos:
        #     data_array = data_array[data_array[:, p1, p2].argsort(kind='mergesort')]
        #     # data_array = data_array[data_array[:, p].argsort(kind='mergesort')]
        # switches = np.where(np.not_equal(data_array[1:], data_array[:-1]).any(axis=1))[0]


    # determine differnces between switchpoints:
    n = data_array.shape[0]
    diff_array = np.diff(np.hstack([-1, switches, n - 1]))
    # print(tmp_switchpoints)
    # print(diff_switch)
    # print(np.sum(diff_switch))
    # determine probabilities:
    p = diff_array / n
    # calculate entropy
    H = np.sum(-p * np.log2(p))
    return H


class EntropySection(object):
    """Analyse (multivariate joint) entropy in 2-D section"""

    def __init__(self, data, pos=[], axis=0, n_jobs=1, *kwds):
        """Analyse (multivariate joint) entropy in 2-D section

        Default: entropy value at each location separately; when positions are given as argument,
        then the joint entropy between each position in the section and the position list is calculated.

        Parallel implemmentation using the python `joblib` package; the entropy itself is caculated with the
        sorting algorithm (see hspace.measures.joint_entropy())

        Args:
            data: Input data set for multiple realisations in one section (therefore: 3D)
            pos = list or array [[x1, x2, ...xn], [y1, y2, ...yn]]: list (or array)
                of fixed positions for multivariate joint entropy calculation
            axis: axis along which entropy is calculated (default: 0)
            n_jobs = int: number of processors to use for parallel execution (default: 1)
            **kwds:

        Returns:
            h : 2-D numpy array with calculated (joint) entropy values
        """
        self.data = data
        self.n_jobs = n_jobs
        self.axis = axis
        self.pos = pos

    def _calulate_entropy(self, **kwds):
        """Perform entropy calculation, in parallel if n_procs > 1

            **Optional keywords**:
            - n_jobs = int: number of processors to use for parallel execution (default: 1)
            - n_max = int : maximum number of data points (default: all)

        """
        self.n_jobs = kwds.get('n_jobs', self.n_jobs)
        n_max = kwds.get("n_max", self.data.shape[0])

        if self.n_jobs == 1:
            self.h = np.empty_like(self.data[0, :, :], dtype='float64')
            # standard sequential calculation:
            for i in range(self.data.shape[1]):
                for j in range(self.data.shape[2]):
                    self.h[i, j] = joint_entropy(self.data[:n_max, i, j])

        else:
            global data # not ideal to create global variable - but required for parallel execution
            data = self.data[:n_max]
            h_par = joblib.Parallel(n_jobs=self.n_jobs)(joblib.delayed(entropy_section_par)(i, j)
                                       for i in range(self.data.shape[1])
                                       for j in range(self.data.shape[2]))

            h_par = np.array(h_par)
            self.h = h_par.reshape((self.data.shape[1], self.data.shape[2]))

    def calc_joint_entropy_section(self, **kwds):
        """Calculate joint entropy between all values in pos and all points in domain

            **Optional keywords**:
            - n_jobs = int: number of processors to use for parallel execution (default: 1)
            - pos = list or array [[x1, x2, ...xn], [y1, y2, ...yn]]: list (or array)
                of fixed positions for multivariate joint entropy calculation
            - n_max = int : maximum number of data points (default: all)
        """
        self.n_jobs = kwds.get('n_jobs', self.n_jobs)
        self.pos = kwds.get("pos", self.pos)
        n_max = kwds.get("n_max", self.data.shape[0])
        if len(self.pos) == 0:
            raise AttributeError("No positions defined! Please set with `pos` argument.")

        self.joint_entropy_section = np.empty_like(self.data[0, :, :], dtype='float64')
        if self.n_jobs == 1:
            # standard sequential calculation:
            for i in range(self.data.shape[1]):
                for j in range(self.data.shape[2]):
                    # add position point to pos array:
                    pos_tmp = np.vstack([self.pos, np.array([i, j])])
                    self.joint_entropy_section[i, j] = joint_entropy(self.data[:n_max,:,:], pos=pos_tmp)

        else:
            global data # not ideal to create global variable - but required for parallel execution
            data = self.data[:n_max,:,:]
            # if len(pos) > 0:
            global pos # set positions as global
            pos = self.pos

            h_par = joblib.Parallel(n_jobs=self.n_jobs)(joblib.delayed(joint_entropy_section_par)(i, j)
                                       for i in range(self.data.shape[1])
                                       for j in range(self.data.shape[2]))

            h_par = np.array(h_par)
            self.joint_entropy_section = h_par.reshape((self.data.shape[1], self.data.shape[2]))

    def calc_cond_entropy_section(self, **kwds):
        """Calculate conditional entropy between all values in pos and all points in domain

            **Optional keywords**:
            - n_jobs = int: number of processors to use for parallel execution (default: 1)
            - pos = list or array [[x1, x2, ...xn], [y1, y2, ...yn]]: list (or array)
                of fixed positions for multivariate joint entropy calculation
            - n_max = int : maximum number of data points (default: all)
        """
        self.n_jobs = kwds.get('n_jobs', self.n_jobs)
        self.pos = kwds.get("pos", self.pos)
        n_max = kwds.get("n_max", self.data.shape[0])

        h_joint_pos = joint_entropy(self.data[:n_max,:,:], self.pos)
        self.calc_joint_entropy_section(n_max=n_max)

        self.cond_entropy_section = self.joint_entropy_section - h_joint_pos



    def _entropy_section_par(self, i, j):
        """Pure convencience fucntion for parallel execution!"""
        return joint_entropy(self.data[:, i, j])

    def plot_entropy(self, **kwds):
        """Create a plot of entropy

        If entropy has not been calculated (i.e. self.h does not exist), then this is automatically
        done here!

        Args:
            **kwds:
            n_jobs = int: number of processors to use for parallel execution (default: 1)
            pts = 2-D array: point positions to include in plot
            data_points = 2-D array: position of data points (e.g. used to generate realizations)

        Returns:

        """

        if not hasattr(self, "h"):
            self._calulate_entropy()

        colorbar = kwds.get("colorbar", "True")
        cmap = kwds.get("cmap", 'viridis')
        vmax = kwds.get("vmax", np.max(self.h))


        if colorbar:
            from mpl_toolkits import axes_grid1
            fig, ax = plt.subplots()
            im = ax.imshow(self.h.transpose(), origin='lower left', cmap=cmap, vmax=vmax)
            if 'pts' in kwds:
                # plot points as overlay:
                print("plot points")
                ax.scatter(kwds['pts'][:, 0], kwds['pts'][:, 1], c='w', marker='s', s=10)
            if 'data_points' in kwds:
                ax.scatter(kwds['data_points'][:, 0], kwds['data_points'][:, 1], c='w', marker='s', s=5)
            divider = axes_grid1.make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.15)
            fig.colorbar(im, cax=cax)
        else:
            plt.imshow(self.h.transpose(), origin='lower left')

    def plot_cond_entropy(self, **kwds):
        """Create a plot of conditional entropy

        Based on previously calculated conditional entropy and defined positions (self.calc_cond_entropy_section() )
        """
        if not hasattr(self, "cond_entropy_section"):
            raise(AttributeError, "Conditional entropy not yet calculated! Please use self.calc_cond_entropy_secion()")

        colorbar = kwds.get("colorbar", "True")

        if colorbar:
            from mpl_toolkits import axes_grid1
            fig, ax = plt.subplots()
            if 'vmin' in kwds and 'vmax' in kwds:
                im = ax.imshow(self.cond_entropy_section.transpose(), origin='lower left',
                               vmin=kwds['vmin'], vmax=kwds['vmax'])
            else:
                im = ax.imshow(self.cond_entropy_section.transpose(), origin='lower left')
            ax.set_xlim([0, self.data.shape[1]])
            ax.set_ylim([0, self.data.shape[2]])
            ax.scatter(self.pos[:, 0], pos[:, 1], c='w', marker='s', s=10)
            divider = axes_grid1.make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.15)
            fig.colorbar(im, cax=cax)
        else:
            plt.imshow(self.cond_entropy_section.transpose(), origin='lower left')

    def plot_mutual_info(self, **kwds):
        """Create a plot of multivariate mutual information

        Based on previously calculated conditional entropy and defined positions (self.calc_cond_entropy_section() )
        """
        if not hasattr(self, "cond_entropy_section"):
            raise(AttributeError, "Conditional entropy not yet calculated! Please use self.calc_cond_entropy_secion()")

        cmap = kwds.get("cmap", "gray")
        colorbar = kwds.get("colorbar", "True")

        if colorbar:
            from mpl_toolkits import axes_grid1
            fig, ax = plt.subplots()
            if 'vmin' in kwds and 'vmax' in kwds:
                im = ax.imshow(self.h.transpose() - self.cond_entropy_section.transpose(), origin='lower left',
                               vmin=kwds['vmin'], vmax=kwds['vmax'], cmap=cmap)
            else:
                im = ax.imshow(self.h.transpose() - self.cond_entropy_section.transpose(), origin='lower left', cmap=cmap)
            ax.set_xlim([0, self.data.shape[1]])
            ax.set_ylim([0, self.data.shape[2]])
            ax.scatter(self.pos[:, 0], pos[:, 1], c='w', marker='s', s=10)
            divider = axes_grid1.make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.15)
            fig.colorbar(im, cax=cax)
        else:
            plt.imshow(self.h.transpose() - self.cond_entropy_section.transpose(), origin='lower left', cmap=cmap)

    def plot_multiple(self, **kwds):
        """Plot multiple random section realisations in one plot

        This method can be useful to obtain a quick impression about the variability
        in the model output sections given the applied parameter distributions.
        Note that, by default, axis ticks and labels are removed for better visibility

        **Optional Keywords**:
            - *ncols* = int : number of columns (default: 8)
            - *nrows* = int : number of rows (default: 2)
            - *cmap* = matplotlib.cmap : colormap (default: YlOrRd)
            - *shuffle_events* = list of event ids : in addition to performing random draws, also
                randomly shuffle events in list
        """
        ncols = kwds.get("ncols", 6)
        nrows = kwds.get("nrows", 2)
        cmap_type = kwds.get('cmap', 'YlOrRd')
        ve = kwds.get("ve", 1.)
        savefig = kwds.get("savefig", False)
        figsize = kwds.get("figsize", (16, 5))

        k = 0  # index for image

        f, ax = plt.subplots(nrows, ncols, figsize=figsize)
        for j in range(ncols):
            for i in range(nrows):
                im = ax[i, j].imshow(self.data[k].T, interpolation='nearest',
                                     aspect=ve, cmap=cmap_type, origin='lower left')
                # remove ticks and labels
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
                # ax[i,j].imshow(im_subs_digit[j*nx+i])
                k += 1

        if savefig:
            fig_filename = kwds.get("fig_filename", "%s_random_sections_%s_pos_%d" % (self.basename, direction, cell_pos))
            plt.savefig(fig_filename, bbox_inches="tight")
        else:
            plt.show()

    def plot_cond_entropy_and_MI(self):
        """Create plots for conditional entropy and mutual information estimation

        Based on previously calculated conditional entropy and defined positions (self.calc_cond_entropy_section() )
        """
        if not hasattr(self, "cond_entropy_section"):
            raise(AttributeError, "Conditional entropy not yet calculated! Please use self.calc_cond_entropy_secion()")

        from mpl_toolkits import axes_grid1
        fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(6, 8))
        im = ax[0].imshow(self.h.T, origin='lower left')  # , vmin=-zmax, vmax=zmax)
        ax[0].set_xlim([0, self.data.shape[1]])
        ax[0].set_ylim([0, self.data.shape[2]])
        divider = axes_grid1.make_axes_locatable(ax[0])
        cax = divider.append_axes('right', size='5%', pad=0.15)
        fig.colorbar(im, cax=cax);

        im = ax[1].imshow(self.cond_entropy_section.T, origin='lower left')  # , vmin=-zmax, vmax=zmax)
        ax[1].set_xlim([0, self.data.shape[1]])
        ax[1].set_ylim([0, self.data.shape[2]])
        divider = axes_grid1.make_axes_locatable(ax[1])
        cax = divider.append_axes('right', size='5%', pad=0.15)
        fig.colorbar(im, cax=cax);
        ax[1].scatter(self.pos[:, 0], pos[:, 1], c='w', marker='s', s=10)

        im = ax[2].imshow(self.cond_entropy_section.T - self.h.T, origin='lower left',
                          cmap='RdBu', interpolation='none',
                          norm=MidpointNormalize(midpoint=0., vmin=-2, vmax=2))
        ax[2].set_xlim([0, self.data.shape[1]])
        ax[2].set_ylim([0, self.data.shape[2]])
        divider = axes_grid1.make_axes_locatable(ax[2])
        cax = divider.append_axes('right', size='5%', pad=0.15)
        fig.colorbar(im, cax=cax);
        ax[2].scatter(self.pos[:, 0], pos[:, 1], c='w', marker='s', s=10)


def entropy_section_par(i, j):
    return joint_entropy(data[:, i, j])


def joint_entropy_section_par(i, j):
    pos_tmp = np.vstack([pos, np.array([i, j])])
    return joint_entropy(data, pos=pos_tmp)



# %%timeit
def calc_parallel(data):


    h_par = joblib.Parallel(n_jobs=4)(joblib.delayed(entropy_section_par)(i,j) \
                          for i in range(data.shape[1])\
                          for j in range(data.shape[2]))
    h_par = np.array(h_par)
    h_par = h_par.reshape((data.shape[1],data.shape[2]))
    return h_par

from matplotlib import colors
# set the colormap and centre the colorbar


class MidpointNormalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))

    from http://chris35wills.github.io/matplotlib_diverging_colorbar/
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))



