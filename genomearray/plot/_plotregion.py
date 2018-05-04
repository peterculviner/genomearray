import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec 
from matplotlib.patches import Polygon
import seaborn as sns

params = {'xtick.labelsize':13,
          'ytick.labelsize':13,
          'axes.labelsize': 15}
plt.rcParams.update(params)
plt.rcParams['figure.dpi'] = 200
plt.rcParams['pdf.fonttype'] = 42
sns.set_style(style='ticks')

class RegionPlot():
    """ Grid of plots showing a section of genomic space and associated data.
        
        Intitalization generates a user-defined number of data axes as well as a map of genomic
        space. The position in genomic space can be altered with functions beginning with
        setPosition and genome-shaped numpy arrays can then be directly plotted on the data axes.
        Positions and regions can be plotted as well.
        
        Parameters:
        ----------
        gene_names : numpy array of names of all genes
            NONE

        gene_regions : numpy array of regions for all genes
            NONE

        n_axes : number of data axes to plot
            NONE

        single_strand : NONE
            NONE

        figsize : NONE
            NONE

        Attributes:
        ----------
        NONE : NONE
            NONE

        |- 0                                                                                  100 -|
    """
    def setPosition_gene(self, name = None, spacer = 0.1, addl_5 = None, addl_3 = None):
        if name in self.gene_names is False: # check if gene exists
            raise IndexError('Gene not found in list of gene names.')
        # get genome position information for gene region
        strand, left, right = self.gene_regions[name == self.gene_names][0]
        length = right - left + 1
        if strand == 0:
            self.top_positive = True
            self.zero = left
            if addl_5 is None:
                self.gleft = int(left-length*spacer)
            else:
                self.gleft = left - addl_5
            if addl_3 is None:
                self.gright = int(right+length*spacer)
            else:
                self.gright = right + addl_3
        elif strand == 1:
            self.top_positive = False
            self.zero = right
            if addl_5 is None:
                self.gright = int(right+length*spacer)
            else:
                self.gright = right + addl_5
            if addl_3 is None:
                self.gleft = int(left-length*spacer)
            else:
                self.gleft = left - addl_3
        self._setxpos()
        self._drawgenes()
    
    def plotLine(self, axis_n = None, data = None, **kwargs):
        if self.top_positive:
            self.ax_data[0][axis_n].plot(self.xpos, data[0,self.gleft:self.gright+1], **kwargs)
            self.ax_data[0][axis_n].set_xlim(self.xpos[0],self.xpos[-1])
            if self.single_strand is False:
                self.ax_data[1][axis_n].plot(self.xpos, data[1,self.gleft:self.gright+1], **kwargs)
                self.ax_data[1][axis_n].set_xlim(self.xpos[0],self.xpos[-1])
        elif self.top_positive is False:
            self.ax_data[0][axis_n].plot(self.xpos, np.flip(data[1,self.gleft:self.gright+1],0), **kwargs)
            self.ax_data[0][axis_n].set_xlim(self.xpos[0],self.xpos[-1])
            if self.single_strand is False:
                self.ax_data[1][axis_n].plot(self.xpos, flip(data[0,self.gleft:self.gright+1],0), **kwargs)
                self.ax_data[1][axis_n].set_xlim(self.xpos[0],self.xpos[-1])
    
    def plotPositions(self, axis_n = None, positions = None, y_array = None, **kwargs):
        positions = positions[(positions[:,1] <= self.gright) & (positions[:,1] >= self.gleft)]
        position_tuple = tuple(np.asarray(positions).T)
        scatter_mask = np.zeros(y_array.shape).astype(bool)
        scatter_mask[position_tuple] = True
        if self.top_positive:
            local_mask = scatter_mask[0,self.gleft:self.gright+1]
            scatter_xpos = self.xpos[local_mask]
            scatter_yval = y_array[0,self.gleft:self.gright+1][local_mask]
            self.ax_data[0][axis_n].scatter(scatter_xpos, scatter_yval, **kwargs)
            self.ax_data[0][axis_n].set_xlim(self.xpos[0],self.xpos[-1])
            if self.single_strand is False:
                local_mask = scatter_mask[1,self.gleft:self.gright+1]
                scatter_xpos = self.xpos[local_mask]
                scatter_yval = y_array[1,self.gleft:self.gright+1][local_mask]
                self.ax_data[1][axis_n].scatter(scatter_xpos, scatter_yval, **kwargs)
                self.ax_data[1][axis_n].set_xlim(self.xpos[0],self.xpos[-1])
        elif self.top_positive is False:
            local_mask = np.flip(scatter_mask[1,self.gleft:self.gright+1],0)
            scatter_xpos = self.xpos[local_mask]
            scatter_yval = np.flip(y_array[1,self.gleft:self.gright+1],0)[local_mask]
            self.ax_data[0][axis_n].scatter(scatter_xpos, scatter_yval, **kwargs)
            self.ax_data[0][axis_n].set_xlim(self.xpos[0],self.xpos[-1])
            if self.single_strand is False:
                local_mask = np.flip(scatter_mask[0,self.gleft:self.gright+1],0)
                scatter_xpos = self.xpos[local_mask]
                scatter_yval = np.flip(y_array[0,self.gleft:self.gright+1],0)[local_mask]
                self.ax_data[1][axis_n].scatter(scatter_xpos, scatter_yval, **kwargs)
                self.ax_data[1][axis_n].set_xlim(self.xpos[0],self.xpos[-1])
    
    def plotRegions(self, axis_n = None, regions = None, y_array = None, **kwargs):
        overlapping_regions = regions[(regions[:,1] <= self.gright) &
                                      (regions[:,2] >= self.gleft)]
        if self.top_positive:
            for region in overlapping_regions:
                strand, left, right = region
                if strand == 0: # on strand if top is positive
                    ylims = self.ax_data[0][axis_n].get_ylim()
                    # draw left lines
                    self.ax_data[0][axis_n].vlines(self._getxpos(left),
                                                   y_array[0,self.gleft+np.where(self._getxpos(left) == self.xpos)[0]],
                                                   self.ax_data[0][axis_n].get_ylim()[1],**kwargs)
                    self.ax_data[0][axis_n].set_ylim(ylims)
                    # draw right lines
                    self.ax_data[0][axis_n].vlines(self._getxpos(right),
                                                   y_array[0,self.gleft+np.where(self._getxpos(right) == self.xpos)[0]],
                                                   self.ax_data[0][axis_n].get_ylim()[1],**kwargs)
                    self.ax_data[0][axis_n].set_ylim(ylims)
                    # fill between
                    self.ax_data[0][axis_n].fill_between(self.xpos,y_array[0,self.gleft:self.gright+1],ylims[1],
                                                         where=(self.xpos>=self._getxpos(left))&
                                                               (self.xpos<=self._getxpos(right)),
                                                         edgecolor='none',alpha=0.25,**kwargs)
                if (strand == 1) and (self.single_strand == False):
                    pass # not implemented yet
        if self.top_positive is False:
            for region in overlapping_regions:
                strand, right, left = region # swapped as order is reversed
                if strand == 1: # on strand if top is positive
                    ylims = self.ax_data[0][axis_n].get_ylim()
                    # draw left lines
                    self.ax_data[0][axis_n].vlines(self._getxpos(left),
                                                   y_array[1,self.gright-np.where(self._getxpos(left) == self.xpos)[0]],
                                                   self.ax_data[0][axis_n].get_ylim()[1],**kwargs)
                    self.ax_data[0][axis_n].set_ylim(ylims)
                    # draw right lines
                    self.ax_data[0][axis_n].vlines(self._getxpos(right),
                                                   y_array[1,self.gright-np.where(self._getxpos(right) == self.xpos)[0]],
                                                   self.ax_data[0][axis_n].get_ylim()[1],**kwargs)
                    self.ax_data[0][axis_n].set_ylim(ylims)
                    # fill between
                    self.ax_data[0][axis_n].fill_between(self.xpos,np.flip(y_array[1,self.gleft:self.gright+1],0),ylims[1],
                                                         where=(self.xpos>=self._getxpos(left))&
                                                               (self.xpos<=self._getxpos(right)),
                                                         edgecolor='none',alpha=0.25,**kwargs)
                if (strand == 1) and (self.single_strand == False):
                    pass # not implemented yet
    
    def _drawgenes(self):
        overlapping_regions = self.gene_regions[(self.gene_regions[:,1] <= self.gright) &
                                                (self.gene_regions[:,2] >= self.gleft)]
        overlapping_names   = self.gene_names[(self.gene_regions[:,1] <= self.gright) &
                                              (self.gene_regions[:,2] >= self.gleft)]
        for name, region in zip(overlapping_names, overlapping_regions):
            gene_strand, left, right = region
            gene_left  = min(self._getxpos(left),self._getxpos(right))
            gene_right = max(self._getxpos(left),self._getxpos(right))
            tri_len = (self.gright - self.gleft)*0.04
            if (gene_strand == 0 and self.top_positive) or (gene_strand == 1 and (self.top_positive is False)):
                self.ax_gene[0].add_patch(
                    Polygon([[gene_left,.62], # left square corner, top
                             [max(gene_left,gene_right-tri_len),.62], # right triangle corner, top
                             [gene_right,.32], # right triangle point, middle
                             [max(gene_left,gene_right-tri_len),.02], # right triangle corner, bottom
                             [gene_left,.02]], # left square corner, bottom
                            linewidth=1.5,facecolor='w'))
                if min(gene_right,self.xpos[-1]) - max(gene_left,self.xpos[0]) >= (self.gright - self.gleft) * .12:
                    self.ax_gene[0].text((min(gene_right,self.xpos[-1]) + max(gene_left,self.xpos[0]))/2,.32, name,
                                          horizontalalignment='center',verticalalignment='center',fontsize=15)
            else:
                self.ax_gene[1].add_patch(
                    Polygon([[gene_left,.68], # left point
                             [gene_left+tri_len,.98], # left triangle corner, top
                             [gene_right,.98], # right square corner, top
                             [gene_right,.38], # right square corner, bottom
                             [gene_left+tri_len,.38]], # left triangle corner, bottom
                            linewidth=1.5,facecolor='w'))
                if min(gene_right,self.xpos[-1]) - max(gene_left,self.xpos[0]) >= (self.gright - self.gleft) * .12:
                    self.ax_gene[1].text((min(gene_right,self.xpos[-1]) + max(gene_left,self.xpos[0]))/2,.68, name,
                                          horizontalalignment='center',verticalalignment='center',fontsize=15)
                    
    def _setxpos(self):
        if self.top_positive:
            self.xpos = np.arange(self.gright-self.gleft+1) - (self.zero - self.gleft)
        elif self.top_positive == False:
            self.xpos = np.arange(self.gright-self.gleft+1) - (self.gright - self.zero)
            
    def _getxpos(self, genome_pos):
        if self.top_positive:
            return genome_pos - self.zero
        else:
            return self.zero - genome_pos
                
    def __init__(self, gene_names, gene_regions, n_axes, single_strand, figsize):
        # store important variables for object function
        self.gene_names = gene_names
        self.gene_regions = gene_regions
        self.single_strand = single_strand
        self.top_positive = None # True if top plots are positive strand, starts as None
        # start making plot
        self.figure = plt.figure(figsize=figsize)
        self.ax_data = [[],[]]
        self.ax_gene = []
        if single_strand: # generate the axes for showing only a single strand
            # define axis locations
            grid = plt.GridSpec(n_axes+2, 1, height_ratios=[2 for i in range(n_axes)]+[1,1])
            # define axes, remove spines, etc.
            for i in range(n_axes): # define data axes
                if i == 0:
                    ax = plt.subplot(grid[i:i+1])
                else:
                    ax = plt.subplot(grid[i:i+1],sharex=self.ax_data[0][0])
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.xaxis.set_ticks_position('bottom')
                ax.yaxis.set_ticks_position('left')
                ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4, integer=True))
                ax.tick_params(axis='both', which='major', pad=1, length=4)
                self.ax_data[0].append(ax)
                if i != n_axes-1:
                    plt.setp(ax.get_xticklabels(),visible=False)
            for i in range(2): # define gene axes
                ax = plt.subplot(grid[n_axes+i:n_axes+i+1],sharex=self.ax_data[0][0])
                ax.set_frame_on(False)
                ax.axes.get_xaxis().set_visible(False)
                ax.axes.get_yaxis().set_visible(False)
                self.ax_gene.append(ax)
        # generate the axes for showing both strands
        elif single_strand == False:
            raise NotImplementedError('only on strand is currently availible.')
        else:
            raise ValueError('single_strand must be True or False.')