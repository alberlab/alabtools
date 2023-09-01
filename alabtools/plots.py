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
import warnings
warnings.simplefilter('ignore', UserWarning)
import matplotlib
matplotlib.use('Agg')

from scipy.ndimage.interpolation import zoom
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from .api import Contactmatrix
from .utils import isiterable


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
    cmap = kwargs.pop('cmap',cwrb)
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
    
    clipmat = np.clip(matrix, a_min=clip_min, a_max=clip_max)
    
    cmax = kwargs.pop('cmax', clipmat.max())
    cmin = kwargs.pop('cmin', clipmat.min())
    
    print("Color Range: ({}, {})".format(cmin, cmax))
    
    im = plt.imshow(clipmat,
                    interpolation='nearest',
                    cmap=cmap,
                    **kwargs)
    im.set_clim(cmin, cmax)
    
    if title != None:
        plt.title(title)

    if 'label' not in kwargs:
        plt.colorbar(im)
    else:
        plt.colorbar(im).set_label(kwargs['label'])


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


def plot_comparison(m1, m2, chromosome=None, file=None, dpi=300, labels=None, title='', **kwargs):

    if not isinstance(m1, Contactmatrix):
        m1 = Contactmatrix(m1)
    if not isinstance(m2, Contactmatrix):
        m2 = Contactmatrix(m2)


    if chromosome is not None:
        m1 = m1[chromosome]
        m2 = m2[chromosome]

    cwrb = make_colormap([(1,1,1),(1,0,0),0.5,(1,0,0),(0,0,0)],'wrb')
    cmap     = kwargs.pop('cmap',cwrb)

    m = np.tril( m1.matrix.toarray(), -1 ) + np.triu( m2.matrix.toarray(), 1 )

    fig = plt.figure(figsize=(10,10))
    plt.title(title)
    plt.imshow(m, cmap=cmap, **kwargs)
    if labels:
        plt.text(0.1, 0.1, labels[0], transform=plt.gca().transAxes)
        plt.text(0.9, 0.9, labels[1], transform=plt.gca().transAxes, horizontalalignment='right', verticalalignment='top')
    plt.colorbar()
    if file is not None:
        plt.tight_layout()
        plt.savefig(file, dpi=dpi)
    return fig


def plot_by_chromosome(data, index, xscale=1e-6, ncols=4, subplot_width=2.5, subplot_height=2.5,
                       sharey=True, subtitlesize=20, ticklabelsize=12, xgridstep=50e6,
                       datalabels=None, highlight_zones=None, highlight_colors='red', vmin=None, vmax=None,
                       suptitle=''):
    '''
    Plot tracks by chromosomes as subplots

    TODO: write docs
    '''

    if not isinstance(index, list) or isinstance(index, tuple):
        index = [index] * len(data)
        data = np.array(data)
        if len(data.shape) == 1:
            data = np.array([data])
        assert data.shape[1] == len(index[0])
    else:
        assert len(data) == len(index)
        for i, d in zip(index, data):
            assert len(i) == len(d)
        for i in range(len(data)):
            data[i] = np.array(data[i])

    if datalabels is not None:
        assert len(data) == len(datalabels)


    if highlight_zones is not None:
        highlight_zones = np.array(highlight_zones)
        if not isiterable(highlight_colors):
            highlight_colors = [highlight_colors] * len(highlight_zones)
        highlight_colors = np.array(highlight_colors)

    chroms = index[0].get_chromosomes() # multiple data better have the same chromosomes
    n_chroms = len(chroms)
    n_cols = 4
    n_rows = n_chroms // n_cols if n_chroms % n_cols == 0 else n_chroms // n_cols + 1
    f, plots = plt.subplots(n_rows, n_cols, figsize=(subplot_width * n_cols, subplot_height * n_rows), sharey=sharey)
    f.suptitle(suptitle)
    if vmin is None:
        vmin = np.nanmin([ np.nanmin(d) for d in data])
    if vmax is None:
        vmax = np.nanmax([ np.nanmax(d) for d in data])
    for i in range(n_chroms):
        col = i % n_cols
        row = i // n_cols
        if highlight_zones is not None:
            ii = np.flatnonzero(highlight_zones[:, 0] == index[0].id_to_chrom(i))
            ch = highlight_zones[ii]
            hgcolors = highlight_colors[ii]
            for (c, s, e), color in zip(ch, hgcolors):
                plots[row, col].fill_between([int(s)*xscale, int(e)*xscale], [vmin, vmin], [vmax, vmax], color=color)
        plots[row, col].set_ylim(vmin, vmax)
        plots[row, col].set_title(index[0].id_to_chrom(i), fontsize=subtitlesize)
        for k in range(len(data)):
            jj = index[k].chrom == chroms[i]
            xx = index[k].start[jj] * xscale
            label = datalabels[k] if datalabels else ''
            plots[row, col].plot(xx, data[k][jj], linewidth=2, label=label)
        if datalabels:
            plots[row, col].legend()
        # sets the grid
        if xgridstep is not None:
            xticks = np.arange(index[k].start[jj][0], index[k].end[jj][-1], xgridstep) * xscale
            plots[row, col].set_xticks(xticks)
        plots[row, col].grid(True, color='grey', linestyle='--',)
        # hide tick labels
        #plots[row, col].set_xticks( np.arange(0, scale*N, 50) )
        plots[row, col].tick_params(axis='both', which='major', labelsize=ticklabelsize, direction='in')
        #plots[row, col].set_xticklabels([''] * len(plots[row, col].get_xticklabels()))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return f, plots

def plot_mesh(mesh, points=None, **kwargs):
    """Plot a 3D mesh with points."""
    # Initialize the figure
    figsize = (8, 8)
    if 'figsize' in kwargs:
        figsize = kwargs['figsize']
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d')
    # Remove ticks
    plt.minorticks_off()
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    # Plot the mesh
    mcol = 'yellow'
    if 'mesh_color' in kwargs:
        mcol = kwargs['meshcolor']
    malpha = 0.5
    if 'mesh_alpha' in kwargs:
        malpha = kwargs['meshalpha']
    ax.plot_trisurf(*zip(*mesh.vertices), triangles=mesh.faces, color=mcol, alpha=malpha)
    # Plot the points
    if points is not None:
        pcol = 'blue'
        if 'points_color' in kwargs:
            pcol = kwargs['points_color']
        palpha = 0.3
        if 'points_alpha' in kwargs:
            palpha = kwargs['points_alpha']
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=pcol, alpha=palpha, s=1)
    # Set the title
    if 'title' in kwargs:
        ax.set_title(kwargs['title'])
    return fig, ax

