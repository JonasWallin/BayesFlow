# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 18:13:22 2015

@author: johnsson
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.colors as colors
import plot
import copy

from .utils.results_mem_efficient import DataSetClustering


class HMplot(object):

    def __init__(self, bmres, marker_lab=None):
        self.bmres = bmres
        #self.comp_colors, self.suco_colors, self.comp_ord, self.suco_ord = self.get_colors_and_order()

        #self.clp = ClustPlot(bmres.clusts, self.comp_colors, self.comp_ord)
        #if hasattr(bmres, 'clust_m'):
        #    self.clp_m = ClustPlot(bmres.clust_m, self.suco_colors, self.suco_ord)
        self.cop = CompPlot(bmres.components)
        self.trp = TracePlot(bmres.traces)

        self.mcsp = {}
        for mimic_key in bmres.mimics:
            mimic = bmres.mimics[mimic_key]
            self.mcsp[mimic_key] = MimicPlot(mimic)

        if marker_lab is None:
            try:
                marker_lab = bmres.meta_data.marker_lab
            except:
                marker_lab = ['']*bmres.d
        self.set_marker_lab(marker_lab)
        self.set_sampnames(bmres.meta_data.samp['names'])

    @property
    def comp_ord(self):
        return self.bmres.comp_ord

    @property
    def suco_ord(self):
        return self.bmres.suco_ord

    @property
    def comp_colors(self):
        return self.bmres.comp_colors

    @property
    def suco_colors(self):
        return self.bmres.suco_colors

    def set_marker_lab(self, marker_lab):
        self.marker_lab = marker_lab
        #self.clp_nm.set_marker_lab(marker_lab)
        #if hasattr(self, 'clp_m'):
        #    self.clp_m.set_marker_lab(marker_lab)
        self.cop.set_marker_lab(marker_lab)
        for mc in self.mcsp:
            self.mcsp[mc].set_marker_lab(marker_lab)

    def set_population_lab(self, pop_lab):
        order = np.argsort(self.suco_ord)
        self.pop_lab = [pop_lab[k] for k in order]
        self.clp_nm.set_population_lab(pop_lab)
        self.clp_m.set_population_lab(pop_lab)

    def set_sampnames(self, names):
        self.sampnames = names
        self.cop.set_sampnames(names)

    def component_fit(self, plotdim, name='pooled', lim=[-.2, 1.2], bins=100,
                      axs=None, **figargs):
        if axs is None:
            fig, axs = plt.subplots(len(plotdim), 4, squeeze=False, **figargs)

        #labels = self.bmres.meta_data.marker_lab
        if name == 'pooled':
            names = self.sampnames
        else:
            names = [name]

        if name in self.mcsp:
            loadsyn = True
        else:
            loadsyn = False
            print "no mimic found, generating new data"
            j = self.sampnames.index(name)
            realdata = self.bmres.data[j]
            N = self.bmres.data[j].shape[0]
            syndata = self.bmres.generate_from_mix(j, N)

        for m, dim in enumerate(plotdim):
            _ = self.cop.latent(dim, ax=axs[m, 0], plotlab=True)
            #ax.set_xlabel(labels[plotdim[m][0]], fontsize=16)
            #ax.set_ylabel(labels[plotdim[m][1]], fontsize=16)
            axs[m, 0].set_xlim(*lim)
            axs[m, 0].set_ylim(*lim)

            _ = self.cop.allsamp(dim, names=names, ax=axs[m, 1], plotlabx=True)
            #ax.set_xlabel(labels[plotdim[m][0]], fontsize=16)
            axs[m, 1].set_xlim(*lim)
            axs[m, 1].set_ylim(*lim)

            if loadsyn:
                self.mcsp[name].realplot.hist2d(dim, bins=bins, ax=axs[m, 2], lims=lim)
                if m == 0:
                    axs[m, 2].set_title(name+'(real)')

                self.mcsp[name].synplot.hist2d(dim, bins=bins, ax=axs[m, 3], lims=lim)
                if m == 0:
                    axs[m, 3].set_title(name+'(synthetic)')
            else:
                plot.hist2d(realdata, dim, bins, ax=axs[m, 2], lims=lim, labels=self.marker_lab)
                if m == 0:
                    axs[m, 2].set_title(name+'(real)')

                plot.hist2d(syndata, dim, bins, ax=axs[m, 3], lims=lim, labels=self.marker_lab)
                if m == 0:
                    axs[m, 3].set_title(name+'(synthetic)')

        return axs

    def pca_biplot(self, comp, ax=None, poplabsh=None, sampmarkers=None):
        '''
            PCA biplot of mixture component probabilities. Sample groups are
            determined by meta_data 'donorid'.

            comp        -   which principal components to plot
            ax          -   where to plot
            poplabsh    -   shift of population labels
            sampmarkers -   markers to use for samples
        '''
        #if sampmarkers is None:
        #    sampmarkers = [(4, 0, 45), (3, 0), (0, 3), (4, 2)]
        #if poplabsh is None:
        #    poplabsh = [[0, 0], [0, -.02], [0, 0], [-.1, 0], [.22, 0], [.06, -.06]]
        if not hasattr(self, 'pop_lab'):
            self.pop_lab = None
        plot.pca_biplot(self.bmres.p_merged, comp, ax, varcol=self.suco_colors, varlabels=self.pop_lab,
                        varlabsh=poplabsh, sampleid=self.bmres.meta_data.samp['donorid'], sampmarkers=sampmarkers)

    def pca_screeplot(self, ax=None):
        plot.pca_screeplot(self.bmres.p, ax)

    def pdip(self, suco=True, fig=None, colorbar=True):
        '''
            Plot p-value of Hartigan's dip test for each cluster.
            Results are plotted order by cluster size, with largest
            cluster in top.
        '''
        if fig is None:
            fig = plt.figure()
        if suco:
            order = self.suco_ord
        else:
            order = self.comp_ord

        pdiplist = self.bmres.get_pdip(suco)
        print "len(pdiplist) = {}".format(len(pdiplist))
        print "len(order) = {}".format(len(order))
        print "suco = {}".format(suco)
        for i, k in enumerate(order):
            pdip = pdiplist[k]
            ax = fig.add_subplot(1, len(pdiplist), i+1)
            p = ax.pcolormesh(pdip[::-1, :])
            p.set_clim(0, 1)
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
            ax.set_ylim(0, pdip.shape[0])
            ax.set_xlabel("k = {}".format(k))
        if colorbar:
            fig.colorbar(p)
        return fig

    def pdip_summary(self, suco=True, fig=None, colorbar=True):
        '''
            Plot p-value of Hartigan's dip test for each cluster.
            Results are plotted order by cluster size, with largest
            cluster in top.
        '''
        if fig is None:
            fig = plt.figure()
        if suco:
            order = self.suco_ord
        else:
            order = self.comp_ord
        pdipsum = self.bmres.get_pdip_summary(suco)
        S = ['Median', '25th percentile', 'Minimum']
        for i, s in enumerate(S):
            ax = fig.add_subplot(1, 3, i+1)
            p = ax.pcolormesh(pdipsum[s][order[::-1], :])
            p.set_clim(0, 1)
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
            ax.set_ylim(0, len(order))
            ax.set_xlabel(s)
        if colorbar:
            fig.colorbar(p)
        return fig

    def prob(self, suco=True, fig=None, totplots=1, plotnbr=1, ks=None):
            '''
                Plot probabilities of belonging to each cluster
            '''
            if fig is None:
                fig = plt.figure()
            nbr_cols = 2*totplots-1
            col_start = 2*(plotnbr-1)

            if suco:
                order = self.suco_ord
                prob = self.bmres.p_merged
            else:
                order = self.comp_ord
                prob = self.bmres.p

            if ks is None:
                K = prob.shape[1]
                ks = range(K)
            else:
                K = len(ks)
                if suco:
                    raise ValueError, "Selection of ks not supported for super components"

            J = prob.shape[0]
            #K = prob.shape[1]

            for i, k in enumerate(ks):
                ax = fig.add_subplot(K, nbr_cols, i*nbr_cols + col_start+1)
                ax.scatter(range(J), prob[:, order[k]])
                ax.set_yscale('log')
                ax.set_ylim(1e-3, 1)
                ax.axes.yaxis.set_ticks([1e-2, 1e-1])
                xlim = ax.get_xlim()
                ax.plot([xlim[0], xlim[1]], [1e-2, 1e-2], color='grey')
                ax.plot([xlim[0], xlim[1]], [1e-1, 1e-1], color='grey')
                ax.axes.xaxis.set_ticks([])
                ax.set_xlim(-1, J)

            return fig

    def prob_bars(self, suco=True, fig=None, js=None):
        if fig is None:
            fig = plt.figure()

        if suco:
            order = self.suco_ord
            colors = self.suco_colors
            prob = self.bmres.p_merged
        else:
            order = self.comp_ord
            colors = self.comp_colors
            prob = self.bmres.p

        if js is None:
            js = range(self.bmres.J)

        prob_list = [prob[j, :] for j in js]
        J = len(js)
        ymax = 1.2*np.max(prob_list)

        for i, j in enumerate(js):
            ax = fig.add_subplot(J, 1, i+1)
            plot.plot_pbars(prob[i], 0, ymax, order=order, colors=colors, ax=ax)

        return fig

    def scatter(self, dim, j, ax=None):
        '''
            Plots the scatter plot of the data over dim.
            Clusters are plotted with their canonical colors (see BMPlot).
        '''
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        samp_clust = self.bmres.clusts[j]
        data = samp_clust.data[:, dim]
        x = samp_clust.x_sample

        if len(dim) == 2:
            for k in range(self.bmres.K):
                ax.plot(data[x == k, 0], data[x == k, 1], '+', label='k = %d' % (k+1), color=self.comp_colors[k])
            ax.plot(data[x == self.bmres.K, 0], data[x == self.bmres.K, 1], '+', label='outliers', color='black')

        elif len(dim) == 3:
            for k in range(self.bmres.K):
                ax.plot(data[x == k, 0], data[x == k, 1], data[x == k, 2], '+', label='k = %d' % (k+1), color=self.comp_colors[k])
            ax.plot(data[x == self.bmres.K, 0], data[x == self.bmres.K, 1], data[x == self.bmres.K, 2], '+', label='outliers', color='black')

        return ax


class ClustPlot(object):

    def __init__(self, samp_clusts, colors, order):
        self.clust = DataSetClustering(samp_clusts)
        #self.clust.plot = self
        self.colors = colors
        self.order = order
        self.pop_lab = None

    def set_marker_lab(self, marker_lab):
        self.marker_lab = marker_lab

    def set_population_lab(self, pop_lab):
        self.pop_lab = pop_lab

    # def box(self, fig=None, totplots=1, plotnbr=1):
    #     '''
    #         Plot boxplots reprsenting quantiles for each cluster.
    #         NB! pooled data is used here.
    #     '''
    #     if fig is None:
    #         fig = plt.figure()
    #     quantiles = self.clust.get_quantiles((.01, .25, .5, .75, .99))
    #     #print "quantiles = {}".format(quantiles)
    #     nbr_cols = 2*totplots - 1
    #     col_start = 2*(plotnbr-1)
    #     boxloc = (np.array(range(self.clust.d)) + .5)/(self.clust.d+1)
    #     boxw = (boxloc[1] - boxloc[0])/3.5
    #     ms = 10
    #     for k in range(self.clust.K):
    #         if not np.isnan(quantiles[self.order[k], 0, 0]):
    #             ax = fig.add_subplot(self.clust.K, nbr_cols, k*nbr_cols + col_start+1)
    #             for dd in range(self.clust.d):
    #                 plot.drawbox(quantiles[self.order[k], :, dd], boxloc[dd], boxw, ms, ax)
    #             ax.axes.xaxis.set_ticks(boxloc)
    #             xlim = ax.get_xlim()
    #             ax.plot([xlim[0], xlim[1]], [.5, .5], color='grey')
    #             if k < self.clust.K-1:
    #                 ax.set_xticklabels(['']*self.clust.d)
    #             else:
    #                 ax.set_xticklabels(self.marker_lab)
    #             ax.set_ylim(-.1, 1.1)
    #             ax.axes.yaxis.set_ticks([.2, .8])
    #     if not self.pop_lab is None:
    #         ax.set_ylabel(self.pop_lab[self.order[k]])
    #     return fig
    
    def pdip_summary(self, fig=None, colorbar=True):
        '''
            Plot p-value of Hartigan's dip test for each cluster.
            Results are plotted order by cluster size, with largest cluster
            in top.
        '''
        if fig is None:
            fig = plt.figure()
        pdipsum = self.clust.get_pdip_summary()
        S = ['Median', '25th percentile', 'Minimum']
        for i, s in enumerate(S):
            ax = fig.add_subplot(1, 3, i+1)
            p = ax.pcolormesh(pdipsum[s][self.order[::-1], :])
            p.set_clim(0, 1)
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
            ax.set_ylim(0, self.clust.K)
            ax.set_xlabel(s)
        if colorbar:
            fig.colorbar(p)
        return fig
        
    def pdip(self, fig=None, colorbar=True):
        '''
            Plot p-value of Hartigan's dip test for each cluster.
            Results are plotted order by cluster size, with largest cluster
            in top.
        '''
        if fig is None:
            fig = plt.figure()
        for j, cl in self.clust.sample_clusts:
            pdip = cl.get_pdip()
            ax = fig.add_subplot(1, len(self.clust.sample_clusts), j+1)
            p = ax.pcolormesh(pdip[self.order[::-1], :])
            p.set_clim(0, 1)
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
            ax.set_ylim(0, self.clust.K)
            ax.set_title(cl.name)
        if colorbar:
            fig.colorbar(p)
        return fig

    # def cum(self, j=None, fig=None):
    #     '''
    #         Empirical CDF for all the clusters in a sample j (or the pooled data)
    #     '''
    #     if fig is None:
    #         fig = plt.figure()
    #     alpha = np.linspace(0, 1, 200)
    #     quantiles = self.clust.get_quantiles(alpha, j)
    #     d = quantiles.shape[2]
    #     K = quantiles.shape[0]
    #     for k in range(K):
    #         for dd in range(d):
    #             ax = fig.add_subplot(K, d, k*d + dd+1)
    #             ax.plot(quantiles[k, :, dd], alpha)
    #             ax.set_ylim(0, 1)
    #             ax.axes.yaxis.set_ticks([])
    #     return fig
    
    # def qhist(self, j=None, ks=None, dds=None, fig=None, totplots=1, plotnbr=1):
    #     '''
    #         Histogram of quantiles for each cluster. Useful for inspecting dip.
    #     '''
    #     if fig is None:
    #         fig = plt.figure()
      
    #     alpha = np.linspace(0, 1, 500)
    #     quantiles = self.clust.get_quantiles(alpha, j, ks, dds)
        
    #     if ks is None:
    #         ks = range(self.K)
    #     ks_ord = [self.order[k] for k in ks]
    #     ks = ks_ord
    #     if dds is None:
    #         dds = range(self.clust.d)
    #     d = len(dds)
    #     print "d = {}".format(d)
    #     nbr_cols = d*totplots + totplots-1
    #     print "nbr_cols = {}".format(nbr_cols)
    #     col_start = (d+1)*(plotnbr-1)

    #     for ik, k in enumerate(ks):
    #         for id, dd in enumerate(dds):
    #             ax = fig.add_subplot(self.clust.K, nbr_cols, k*nbr_cols + col_start + dd + 1)
    #             ax.hist(quantiles[ik, :, id], bins = 50, color=self.colors[k])
    #             ax.axes.xaxis.set_ticks([])
    #             ax.axes.yaxis.set_ticks([])
    #     return fig
        
        
    # def qhist_dipcrit(self, q=.25, fig=None, totplots=1, plotnbr=1):
    #     '''
    #         Histogram of quantiles for the sample with dip at the qth quantile.
    #         I.e. q=.25 displays quantiles for the sample which has its dip at the
    #         0.25th quantile among the samples.
    #     '''
    #     if fig is None:
    #         fig = plt.figure()
    #     alpha = np.linspace(0, 1, 500)
    #     K = self.clust.K
    #     d = self.clust.d
    #     pdiplist = self.clust.get_pdip() 

    #     nbr_cols = d*totplots + totplots-1
    #     col_start = (d+1)*(plotnbr-1)  
    #     #qind = np.int(K*q)-1
    #     jq = np.zeros((K, d), dtype='i')
    #     for ik, k in enumerate(self.order):
    #         pd = pdiplist[k]
    #         K_loc = sum(~np.isnan(pd[:, 0]))
    #         for dd in range(d):
    #             qind = max(np.int(K_loc*q)-1, 0)
    #             jq[k, dd] = np.argsort(pd[:, dd])[qind]
    #             quantiles = self.clust.get_quantiles(alpha, jq[k, dd], [k], [dd])[0, :, 0]
    #             ax = fig.add_subplot(self.clust.K, nbr_cols, ik*nbr_cols + col_start + dd + 1)
    #             ax.hist(quantiles, bins = 50, color=self.colors[k], range=(-0.1, 1.4))
    #             ax.set_xlim((-0.1, 1.4))
    #             ax.axes.xaxis.set_ticks([0, 0.5, 1])
    #             ax.axes.yaxis.set_ticks([])
    #     return fig

    # def chist_allsamp(self, min_clf, dd, ks, fig=None, ncol=4):
    #     '''
    #         Histogram of data points with at least min_clf probability of belonging to certain clusters.
    #         The clusters are displayed with their canonical colors.
            
    #         A panel of plots with ncol columns is showing this for all samples.
    #     '''
    #     if fig is None:
    #         fig = plt.figure()
    #     nrow = np.ceil(self.clust.J/ncol)
    #     for j in range(self.clust.J):
    #         ax = fig.add_subplot(nrow, ncol, j+1)
    #         self.chist(min_clf, dd, j, ks, ax)
    #     return fig

    # def chist(self, min_clf, dd, j=None, ks=None, ax=None):
    #     '''
    #         Histogram of data points with at least min_clf probability of belonging to certain clusters.
    #         The clusters are displayed with their canonical colors (see BMPlot).
    #     '''
    #     if ax is None:
    #         fig = plt.figure()
    #         ax = fig.add_subplot(111)
    #     if ks is None:
    #         ks = range(self.clust.K)

    #     for k in ks:
    #         data = self.clust.get_data_kdj(min_clf, self.order[k], dd, j)
    #         if len(data) > 0:
    #             ax.hist(data, bins=50, color=self.colors[self.order[k]], alpha = .7, range = (-0.1, 1.4))
    #     return ax

    # def scatter(self, dim, j, fig=None):
    #     '''
    #         Plots the scatter plot of the data over dim.
    #         Clusters are plotted with their canonical colors (see BMPlot).
    #     '''
    #     if fig is None:
    #         fig = plt.figure()
    #     ax = fig.add_subplot(111)
    			
    #     data = self.clust.data[j][:, dim]
    #     x = self.clust.sample_x(j)

    #     if len(dim) == 2:
    #         for k in range(self.clust.K):
    #             ax.plot(data[x==k, 0], data[x==k, 1], '+', label='k = %d'%(k+1), color=self.colors[k])
    #         ax.plot(data[x==self.clust.K, 0], data[x==self.clust.K, 1], '+', label='outliers', color='black')
        
    #     elif len(dim) == 3:
    #         ax = fig.gca(projection='3d')
    #         for k in range(self.clust.K):
    #             ax.plot(data[x==k, 0], data[x==k, 1], data[x==k, 2], '+', label='k = %d'%(k+1), color=self.colors[k])
    #         ax.plot(data[x==self.clust.K, 0], data[x==self.clust.K, 1], data[x==self.clust.K, 2], '+', label='outliers', color='black')
    						
    #     return fig, ax
        
class CompPlot(object):
    
    def __init__(self, components):
        self.comp = components
        self.comp.plot = self

    @property
    def comp_colors(self):
        return self.components.comp_colors

    @property
    def suco_colors(self):
        return self.components.suco_colors
        
    @property
    def comp_ord(self):
        return self.components.comp_ord
        
    @property
    def suco_ord(self):
        return self.components.suco_ord
        
    def set_marker_lab(self, marker_lab):
        self.marker_lab = marker_lab

    def set_sampnames(self, names):
        self.sampnames = names
        
    def center(self, suco=True, fig=None, totplots=1, plotnbr=1, yscale=False,
               ks=None, with_outliers=True, alpha=1):
        '''
            The centers of all components (mu param) are plotted along one dimension.
            
            If suco=True, components belonging to the same super component are
            plotted in the same panel.
        '''
        if fig is None:
            fig = plt.figure()
            
        colors = [col[:-1]+(alpha, ) for col in self.comp_colors]
            
        if suco:
            comps_list = copy.deepcopy(self.comp.mergeind)
            order = list(self.suco_ord)[:]
        else:
            comps_list = [[k] for k in range(self.comp.K)]
            order = list(self.comp_ord)[:]

        if not ks is None:
            ks = [self.comp_ord[k] for k in ks]
            s = 0
            while s < len(comps_list):
                comp = comps_list[s]
                for k in comp:
                    if k not in ks:
                        comp.remove(k)
                if len(comp) == 0:
                    comps_list.pop(s)
                    order.remove(s)
                    order = [o-1 if o > s else o for o in order]
                else:
                    s += 1

        if not with_outliers:
            outliers = self.comp.mu_outliers
    
        nbr_cols = 2*totplots-1
        col_start = 2*(plotnbr-1)
    
        S = len(order)
        
        for s in range(S):
            comps = comps_list[order[s]]
            #print "comps = {}".format(comps)
            ax = fig.add_subplot(S, nbr_cols, s*nbr_cols + col_start+1)
            for k in comps:
                mu_ks = self.comp.mupers[:, k, :]
                if not with_outliers:
                    mu_ks = mu_ks[~outliers[:, k], :]
                for j in range(self.comp.J):
                    ax.plot(range(self.comp.d), mu_ks[j, :], color=colors[k])
                ax.plot([0, self.comp.d-1], [.5, .5], color='grey')
            if s == S-1:
                ax.axes.xaxis.set_ticks(range(self.comp.d))
                ax.set_xticklabels(self.marker_lab)
            else:
                ax.axes.xaxis.set_ticks([])
            if not yscale:
                pass                
                #ax.axes.yaxis.set_ticks([])
                #ax.set_ylim(0, 1)
            else:
                ax.axes.yaxis.set_ticks([.2, .8])
                ax.set_ylim(-.1, 1.1)		
        return fig
    
    def center3D(self, dim, fig=None):
        '''
            Plots the centers (mu) of the components in three dimensions.
            
            dim     -   dimensions on which to project centers.
        '''
        if fig is None:
            fig = plt.figure()
            
        ax = fig.gca(projection='3d')
    
        for k in range(self.comp.K):
            mus = self.comp.mupers[:, k, :]
            ax.scatter(mus[:, dim[0]], mus[:, dim[1]], mus[:, dim[2]], marker='.', color = self.comp_colors[k], s=50) 
            
        ax.axes.xaxis.set_ticks([.1, .5, .9])
        ax.axes.zaxis.set_ticks([.1, .5, .9])
        ax.axes.yaxis.set_ticks([.1, .5, .9])
        ax.view_init(30, 165)
        
        ax.set_xlabel(self.marker_lab[dim[0]])
        ax.set_ylabel(self.marker_lab[dim[1]])
        ax.set_zlabel(self.marker_lab[dim[2]])
            
        return ax

    @staticmethod
    def set_lims(ax, lims, dim):
        try:
            xlim = lims[dim[0], :]
            ylim = lims[dim[1], :]
        except (TypeError, ValueError):
            xlim = lims
            ylim = lims
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    def latent(self, dim, ax=None, ks=None, plim=[0, 1], plotlab=False, plotlabx=False, plotlaby=False,
               plot_new_th=True, lw=2, lims=None):
        '''
            Plot visualization with ellipses of the latent components.
            Canonical colors are used (see BMplot).
            
            dim     -   which two dimensions to project on.
            ax     -   where to plot
            ks      -   which components to plot
            plim    -   only components with size within plim are plotted
            plotlab -   should labels be shown?
        '''
        #print "suco_list = {}".format(suco_list)
        if ax is None:
    		f = plt.figure()
    		ax = f.add_subplot(111)
        else:
    		f = None

        if ks is None:
            ks = range(self.comp.K)
        ks = [self.comp_ord[k] for k in ks]            
        okcl = set.intersection(set(self.within_plim(plim)), set(ks))
        
        mus = [self.comp.mulat[k, :] for k in okcl]
        Sigmas = [self.comp.Sigmalat[k, :, :] for k in okcl]
        colors = [self.comp_colors[k] for k in okcl]

        q = plot.component_plot(mus, Sigmas, dim, ax, colors=colors, lw=lw)
        
        if hasattr(self.comp, 'new_thetas') and plot_new_th:
            ax.scatter(self.comp.new_thetas[:, dim[0]], self.comp.new_thetas[:, dim[1]], s=40, c='k', marker='+')
        
        if plotlab or plotlabx:
            ax.set_xlabel(self.marker_lab[dim[0]], fontsize=16)
        if plotlab or plotlaby:
            ax.set_ylabel(self.marker_lab[dim[1]], fontsize=16)

        if not lims is None:
            self.set_lims(ax, lims, dim)

        return q

    def allsamp(self, dim, ax=None, ks=None, plim=[0, 1], js=None, names=None, plotlab=False, plotlabx=False, plotlaby=False,
                plot_new_th=True, lw=1, lims=None):

        '''
            Plot visualization of mixture components for all samples.
            Canonical colors are used (see BMplot).
            
            dim     -   which two dimensions to project on.
            ax     -   where to plot
            ks      -   which components to plot
            plim    -   only components with size within plim are plotted
            js      -   which flow cytometry samples to plot
            name    -   sampname of flow cytometry sample to plot
            plotlab -   should labels be shown?
        '''
        if ax is None:
            f = plt.figure()
            ax = f.add_subplot(111)
        else:
            f = None
            
        if ks is None:
            ks = range(self.comp.K)
        ks = [self.comp_ord[k] for k in ks]
        if not names is None:
            js = [self.sampnames.index(name) for name in names]
        else:
            if js is None:
                js = range(self.comp.J)

        okcl = set.intersection(set(self.within_plim(plim)), set(ks))
        
        muspers = [[self.comp.mupers[j, k, :] for k in okcl] for j in js]
        Sigmaspers = [[self.comp.Sigmapers[j, k, :, :] for k in okcl] for j in js]
        colors = [self.comp_colors[k] for k in okcl]

        q = plot.pers_component_plot(muspers, Sigmaspers, dim, ax, colors=colors, lw=lw)
 
        if hasattr(self.comp, 'new_thetas') and plot_new_th:
            ax.scatter(self.comp.new_thetas[:, dim[0]], self.comp.new_thetas[:, dim[1]], s=40, c='k', marker='+')

        if plotlab or plotlabx:
            ax.set_xlabel(self.marker_lab[dim[0]], fontsize=16)
        if plotlab or plotlaby:
            ax.set_ylabel(self.marker_lab[dim[1]], fontsize=16)

        if not lims is None:
            self.set_lims(ax, lims, dim)

        return q
        
    def within_plim(self, plim):
        okp = np.array([True]*self.comp.K)
        for suco in self.comp.mergeind:
            p_suco = np.mean(np.sum(self.comp.p[:, suco], axis=1))
            if p_suco < plim[0] or p_suco > plim[1]:
                okp[suco] = False
            okcl = np.nonzero(okp)[0]
        return okcl

    def latent_allsamp(self, dimlist, fig=None, ks=None, plim=[0, 1], js=None, plotlab=True):
        '''
            Plot a panel of both latent component and mixture components for 
            all samples.
            
            dimlist -  list of dimensions which to project on.
            fig     -   where to plot
            ks      -   which components to plot
            plim    -   only components with size within plim are plotted
            plotlab -   should labels be shown?
        '''

        if fig is None:
            fig = plt.figure()

        for m in range(len(dimlist)):
            ax1 = fig.add_subplot(len(dimlist), 2, 2*m+1)#plt.subplot2grid((2, 2), (m, 0))
            ql = self.latent(dimlist[m], ax1, ks, plim)
            ax2 = fig.add_subplot(len(dimlist), 2, 2*m+2)#plt.subplot2grid((2, 2), (m, 1))
            qa = self.allsamp(dimlist[m], ax2, ks, plim, js)

            if m == 0:
                ax1.set_title(self.marker_lab[dimlist[m][0]], fontsize=16)
                ax2.set_title(self.marker_lab[dimlist[m][0]], fontsize=16)
            else:
                ax1.set_xlabel(self.marker_lab[dimlist[m][0]], fontsize=16)
                ax2.set_xlabel(self.marker_lab[dimlist[m][0]], fontsize=16)
            ax1.set_ylabel(self.marker_lab[dimlist[m][1]], fontsize=16)

            plot.set_component_plot_tics([ax1, ax2], plot.mergeQ(ql, qa))
        return fig

    def bhattacharyya_overlap_quotient(self, fig=None, totplots=1, plotnbr=1):
        '''
            Diagnostic plot showing quotient between distance to correct latent
            center and distance to nearest wrong latent center.
        '''
        if fig is None:
            fig = plt.figure()
        distquo = self.comp.get_latent_bhattacharyya_overlap_quotient()
        fig = plot.plot_diagnostics(distquo, 0, 3, 1, self.comp_ord, 'Bhattacharyya overlap quotient', fig=fig, totplots=totplots, plotnbr=plotnbr)
        return fig
    
    def center_distance_quotient(self, fig=None, totplots=1, plotnbr=1):
        '''
            Diagnostic plot showing quotient between distance to correct latent
            center and distance to nearest wrong latent center.
        '''
        if fig is None:
            fig = plt.figure()
        distquo = self.comp.get_center_distance_quotient()
        fig = plot.plot_diagnostics(distquo, 0, 3, 1, self.comp_ord, 'Distance to mean quotient', fig=fig, totplots=totplots, plotnbr=plotnbr)
        return fig
    
    def cov_dist(self, norm='F', fig=None, totplots=1, plotnbr=1):
        '''
            Diagnostic plot showing distance between convariance matrices
            of the mixture components and the corresponding latent components.
            
            norm    -   which norm to use for computing the distance
        '''
        distF = self.comp.get_cov_dist(norm)
        plot.plot_diagnostics(np.log10(distF), -5, 0, -3, self.comp_ord, 'Covariance matrix distance (norm {})'.format(norm), False, fig=fig, totplots=totplots, plotnbr=plotnbr)
        return fig

class TracePlot(object):
    
    def __init__(self, traces, order):
        self.traces = traces
        self.traces.plot = self

    @property
    def order(self):
        return self.traces.comp_ord
        
    def all(self, fig=None, yscale=True):
        '''
            Plot trace plots of latent means and nus.
        '''
        if fig is None:
            fig = plt.figure()
        for k in range(self.traces.K):
            ax = plt.subplot2grid((1, self.traces.K), (0, k))
            self.mulat(k, ax, yscale)
            ax.set_title('theta_'+'{}'.format(k+1))
        # ax = plt.subplot2grid((1, self.traces.K+1), (0, k+1))
        # self.nu(ax)
        # ax.set_title('nu', fontsize=16)
        return fig, ax
        
    def mulat(self, k, ax=None, yscale=True):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        ax.plot(self.traces.ind, self.traces.get_mulat_k(self.order[k]))
        ax.set_xlim(0, self.traces.ind[-1])
        if yscale:
            ax.set_ylim(-.2, 1.2)
            ax.axes.yaxis.set_ticks([0.1, 0.9])
        plt.axvspan(0, self.traces.burnind[-1], facecolor='0.5', alpha=0.5)
        
    def nu(self, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        ax.plot(self.traces.ind, self.traces.get_nu())
        ax.set_xlim(0, self.traces.ind[-1])
        ax.set_yscale('log')
        ax.axes.yaxis.set_ticks([100, 1000])
        plt.axvspan(0, self.traces.burnind[-1], facecolor='0.5', alpha=0.5)

    def nu_sigma(self, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        ax.plot(self.traces.ind, self.traces.get_nu_sigma())
        ax.set_xlim(0, self.traces.ind[-1])
        plt.axvspan(0, self.traces.burnind[-1], facecolor='0.5', alpha=0.5)


class MimicPlot(object):
    
    def __init__(self, mimic):
        self.mimic = mimic
        self.realplot = FCplot(mimic.realsamp)
        self.synplot = FCplot(mimic.synsamp)
        
    def set_marker_lab(self, marker_lab):
        self.realplot.set_marker_lab(marker_lab)
        self.synplot.set_marker_lab(marker_lab)
        
class FCplot(object):
    
    def __init__(self, fcsample):
        self.fcsample = fcsample
        self.fcsample.plot = self
        
    def set_marker_lab(self, marker_lab):
        self.marker_lab = marker_lab

    def hist2d(self, dim, Nsamp=None, bins=50, quan=[0.5, 99.5], quan_plot=[5, 95], ax=None, lims=None):
        '''
            Plot 2D histograms of a given sample (synthetic or real).
        '''
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        data = self.fcsample.get_data(Nsamp)
        plot.hist2d(data, dim, bins, quan, quan_plot, ax, lims, labels=self.marker_lab)
        #ax.hist2d(data[:, dim[0]], data[:, dim[1]], bins = bins, norm=colors.LogNorm(), vmin=1)
        #ax.patch.set_facecolor('white')
        #if not xlim is None:
        #    ax.set_xlim(*xlim)
        #if not ylim is None:
        #    ax.set_ylim(*ylim)     

    def histnd(self, Nsamp=None, bins=50, fig=None, xlim=None, ylim=None):
        '''
            Plot panel of 1D and 2D histograms of a given sample (synthetic or real).
        '''
        if fig is None:
            fig = plt.figure()
        plot.histnd(self.fcsample.get_data(Nsamp), bins, [0, 100], [5, 95], fig,
               labels=self.marker_lab)
        if not xlim is None:
            for ax in fig.axes:
                ax.set_xlim(*xlim)
        if not ylim is None:
            for ax in fig.axes:
                if ax.get_ylim()[1] < 5:
                    ax.set_ylim(*ylim)
        
