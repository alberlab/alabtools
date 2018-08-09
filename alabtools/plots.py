# Copyright (C) 2017 University of Southern California and
#                          Nan Hua
#
# Authors: Nan Hua
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
from __future__ import division, print_function
__author__  = "Nan Hua"

__license__ = "GPL"
__version__ = "0.0.1"
__email__   = "nhua@usc.edu"

import numpy as np
import matplotlib
matplotlib.use('Agg')

from scipy.ndimage.interpolation import zoom
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def make_colormap(seq,cmapname='CustomMap'):
    """
    Return a LinearSegmentedColormap

    Parameters
    ----------
    seq : list
        a sequence of floats and RGB-tuples. The floats should be increasing and in the interval (0,1).

    Example
    -------
        make_colormap([(1,1,1),(1,0,0)]) is a colormap from white to red
        make_colormap([(1,1,1),(1,0,0),0.5,(1,0,0),(0,0,0)],'wrb') is a colormap from white to red to black

    Returns
    -------

    LinearSegmentedColormap
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap(cmapname, cdict)

red = make_colormap([(1,1,1),(1,0,0)])

def plotmatrix(figurename, matrix, title=None, dpi=300, **kwargs):
    """Plot a 2D array with a colorbar.

    Parameters
    ----------

    matrix : a 2d numpy array
        A 2d array to plot
    cmap : matplotlib color map
        Color map used in matrix, e.g cm.Reds, cm.bwr
    clip_min : float, optional
        The lower clipping value. If an element of a matrix is <clip_min, it is
        plotted as clip_min.
    clip_max : float, optional
        The upper clipping value.
    label : str, optional
        Colorbar label
    ticklabels1 : list, optional
        Custom tick labels for the first dimension of the matrix.
    ticklabels2 : list, optional
        Custom tick labels for the second dimension of the matrix.
    max_resolution : int, optional
        Set a maximum resolution for the output file.
    """

    clip_min = kwargs.pop('clip_min', -np.inf)
    clip_max = kwargs.pop('clip_max', np.inf)

    cwrb = make_colormap([(1,1,1),(1,0,0),0.5,(1,0,0),(0,0,0)],'wrb')
    cmap     = kwargs.pop('cmap',cwrb)
    fig  = plt.figure()
    if 'ticklabels1' in kwargs:
        plt.yticks(range(matrix.shape[0]))
        plt.gca().set_yticklabels(kwargs.pop('ticklabels1'))

    if 'ticklabels2' in kwargs:
        plt.xticks(range(matrix.shape[1]))
        plt.gca().set_xticklabels(kwargs.pop('ticklabels2'))

    if 'max_resolution' in kwargs:
        mr = kwargs.pop('max_resolution')
        if len(matrix) > mr:
            # use linear interpolation to avoid negative values
            matrix = zoom(matrix, float(mr) / len(matrix), order=1)

    plt.imshow(np.clip(matrix, a_min=clip_min, a_max=clip_max),
                        interpolation='nearest',
                        cmap=cmap,
                        **kwargs)
    if title != None:
        plt.title(title)

    if 'label' not in kwargs:
        plt.colorbar()
    else:
        plt.colorbar().set_label(kwargs['label'])


    if figurename[-3:] == 'png':
        fig.savefig(figurename, dpi=dpi)
    elif figurename[-3:] == 'pdf':
        from matplotlib.backends.backend_pdf import PdfPages
        pp = PdfPages(figurename)
        pp.savefig(fig, dpi=dpi)
        pp.close()

    plt.show()
    plt.close(fig)
def plotxy(figurename, x, y, color='blue', linewidth=1, points=False, xlab=None, ylab=None,title=None, xlim=None, ylim=None, grid=False, xticklabels=None, yticklabels=None, vline=None, hline=None,  **kwargs):
    """xy plot
    Parameters:
    -----------
    x,y: dataset used to plot
    format: str
        format to save figure
    color: drawing color
    linewidth: 
    points : True or False, if scatter points are required
    
    xlab/ylab : string, optional
        label for x/y axis
    title : string, optional
        title of the figure
    xlim,ylim :tuples for xlim, ylim
    vline/hline: float or array, optional
        draw a vertical/horizontal line at certain position(s)
    xticks/yticks: ticks for x,y axis
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line = ax.plot(x,y,c=color,**kwargs)
    plt.setp(line,linewidth=linewidth)
    if points:
        ax.scatter(x,y, marker='o',c=color,edgecolors=color)
    
    if xlab != None:
        ax.set_xlabel(xlab)
    if ylab != None:
        ax.set_ylabel(ylab)
    if title != None:
        ax.set_title(title)
    if xticklabels != None:
        ax.set_xticklabels(xticklabels)
    if yticklabels != None:
        ax.set_yticklabels(yticklabels)
    if xlim != None:
        ax.set_xlim(xlim[0],xlim[1])
    if ylim != None:
        ax.set_ylim(ylim[0],ylim[1])
    if grid:
        ax.grid(True)
    if vline != None:
        for l in np.array([vline]).flatten():
            ax.axvline(l, color='c', linestyle='dashed', linewidth=1)
    if hline != None:
        for l in np.array([hline]).flatten():
            ax.axhline(l, color='c', linestyle='dashed', linewidth=1)
    plt.show()
    if figurename[-3:] == 'png':
        fig.savefig(figurename, dpi=600)
    elif figurename[-3:] == 'pdf':
        from matplotlib.backends.backend_pdf import PdfPages
        pp = PdfPages(figurename)
        pp.savefig(fig, dpi=600)
        pp.close()
  
    plt.close(fig)
def plot_comparison(m1, m2, chromosome=None, file=None, dpi=300, **kwargs):
    if chromosome is not None:
        m1 = m1[chromosome]
        m2 = m2[chromosome]

    cwrb = make_colormap([(1,1,1),(1,0,0),0.5,(1,0,0),(0,0,0)],'wrb')
    cmap     = kwargs.pop('cmap',cwrb)

    m = np.tril( m1.matrix.toarray() ) + np.triu( m2.matrix.toarray() )

    fig = plt.figure(figsize=(10,10))
    plt.imshow(m, cmap=cmap, **kwargs)
    plt.colorbar()
    if file is None:
        plt.show()
    else:
        plt.savefig(file, dpi=dpi)
        plt.close(fig)
