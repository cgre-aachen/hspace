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

        for p in pos:
            # data_array = data_array[data_array[:, p].argsort(kind='mergesort')]
            data_array = data_array[data_array[:, p].argsort(kind='mergesort')]
        switches = np.where(np.not_equal(data_array[1:], data_array[:-1]).any(axis=1))[0]

    elif len(data_array.shape) == 3: # 1-D case, requires iteration over positions
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

