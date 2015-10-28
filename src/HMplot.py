# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 18:13:22 2015

@author: johnsson
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

from . import plot
from .utils.results_mem_efficient import DataSetClustering


class HMplot(object):

    def __init__(self, bmres):
        self.bmres = bmres
        try:
            self.cop = CompPlot(bmres.components)
            self.trp = TracePlot(bmres.traces)

            self.mcsp = {}
            for mimic_key in bmres.mimics:
                mimic = bmres.mimics[mimic_key]
                self.mcsp[mimic_key] = MimicPlot(mimic)
        except AttributeError as e:
            print e
            pass

        if hasattr(bmres, 'meta_data'):
            self.marker_lab = bmres.meta_data.marker_lab
            self.sampnames = bmres.meta_data.samp['names']

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

    @property
    def marker_lab(self):
        if hasattr(self, '_marker_lab'):
            return self._marker_lab
        return ['']*self.bmres.d

    @marker_lab.setter
    def marker_lab(self, marker_lab):
        self._marker_lab = marker_lab
        if hasattr(self, 'cop'):
            self.cop.marker_lab = marker_lab
        if hasattr(self, 'mcsp'):
            for mc in self.mcsp:
                self.mcsp[mc].marker_lab = marker_lab

    @property
    def pop_lab(self):
        if hasattr(self, '_pop_lab'):
            return self._pop_lab
        return None

    @pop_lab.setter
    def pop_lab(self, pop_lab):
        order = np.argsort(self.suco_ord)
        self._pop_lab = [pop_lab[k] for k in order]
        if hasattr(self, 'cop'):
            self.cop.pop_lab = pop_lab

    @property
    def sampnames(self):
        if hasattr(self, '_sampnames'):
            return self._sampnames
        return None

    @sampnames.setter
    def sampnames(self, names):
        self._sampnames = names
        if hasattr(self, 'cop'):
            self.cop.sampnames = names

    def box(self, axs=None, suco=True, **figargs):
        '''
            Plot boxplots representing quantiles for each cluster.
            NB! pooled data is used here.
        '''
        quantiles = self.bmres.get_quantiles((.01, .25, .5, .75, .99), suco=suco)

        if axs is None:
            fig, axs = plt.subplots(quantiles.shape[0], **figargs)

        boxloc = (np.array(range(self.bmres.d)) + .5)/(self.bmres.d+1)
        boxw = (boxloc[1] - boxloc[0])/3.5
        ms = 10

        order = self.suco_ord if suco else self.comp_ord

        for i, (ax, k) in enumerate(zip(axs, order)):
            if not np.isnan(quantiles[k, 0, 0]):
                for dd in range(self.bmres.d):
                    plot.drawbox(quantiles[k, :, dd], boxloc[dd], boxw, ms, ax)
            ax.axes.xaxis.set_ticks(boxloc)
            xlim = ax.get_xlim()
            ax.plot([xlim[0], xlim[1]], [.5, .5], color='grey')
            if i < min(len(order), len(axs))-1:
                ax.set_xticklabels(['']*self.bmres.d)
            else:
                ax.set_xticklabels(self.marker_lab)
            ax.set_ylim(-.1, 1.1)
            ax.axes.yaxis.set_ticks([.2, .8])
        if not self.pop_lab is None:
            ax.set_ylabel(self.pop_lab[self.order[k]])
        return axs

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
            print "self.mcsp[name] = {}".format(self.mcsp[name])
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
        non_active = np.sum((self.bmres.active_komp > 0.05), axis=0) == 0
        non_active_suco = np.array([non_active[np.array(suco)].all() for suco in self.bmres.mergeind])
        plot.pca_biplot(self.bmres.p_merged[:, ~non_active_suco], comp, ax,
                        varcol=[col for k, col in enumerate(self.suco_colors)
                                if not non_active_suco[k]],
                        varlabels=self.pop_lab, varlabsh=poplabsh,
                        sampleid=self.bmres.meta_data.samp['donorid'],
                        sampmarkers=sampmarkers)

    def pca_screeplot(self, ax=None):
        plot.pca_screeplot(self.bmres.p_merged, ax)

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

    def prob(self, suco=True, axs=None, ks=None, **figargs):
            '''
                Plot probabilities of belonging to each cluster
            '''

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
                    raise ValueError("Selection of ks not supported for super components")

            if axs is None:
                fig, axs = plt.subplots(K, **figargs)

            J = prob.shape[0]

            for ax, k in zip(axs, ks):
                ax.scatter(range(J), prob[:, order[k]])
                ax.set_yscale('log')
                ax.set_ylim(1e-3, 1)
                ax.axes.yaxis.set_ticks([1e-2, 1e-1])
                xlim = ax.get_xlim()
                ax.plot([xlim[0], xlim[1]], [1e-2, 1e-2], color='grey')
                ax.plot([xlim[0], xlim[1]], [1e-1, 1e-1], color='grey')
                ax.axes.xaxis.set_ticks([])
                ax.set_xlim(-1, J)

            return axs

    def qhist(self, j, k, dd, ax=None, **figargs):
        '''
            Histogram of quantiles for each cluster. Useful for
            inspecting dip.
        '''
        if ax is None:
            fig, ax = plt.subplot(**figargs)

        alpha = np.linspace(0, 1, 500)
        quantiles = self.bmres.get_quantiles(alpha, j, [k], [dd])

        rangex = (-.2, 1.2)
        ax.hist(quantiles.reshape(-1), bins=200, color=self.suco_colors[k],
                range=rangex)
        ax.set_xlim(rangex)
        return ax

    def scatter(self, dim, j, ax=None):
        '''
            Plots the scatter plot of the data over dim.
            Clusters are plotted with their canonical
            colors (see BMPlot).
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


class CompPlot(object):

    def __init__(self, components):
        self.comp = components
        self.comp.plot = self

    @property
    def comp_colors(self):
        return self.comp.comp_colors

    @property
    def suco_colors(self):
        return self.comp.suco_colors

    @property
    def comp_ord(self):
        return self.comp.comp_ord

    @property
    def suco_ord(self):
        return self.comp.suco_ord

    def center(self, suco=True, axs=None, yscale=False,
               ks=None, with_outliers=True, alpha=1, **figargs):
        '''
            The centers of all components (mu param) are plotted along
            one dimension.

            If suco=True, components belonging to the same super
            component are plotted in the same panel.

            ks can be used to select a subset of components to be
            plotted. The input ks should be given relative to the
            canoncial order (i.e. the order that they are plotted).
        '''

        colors = [col[:-1]+(alpha, ) for col in self.comp_colors]

        if suco:
            comps_list = self.comp.mergeind
            order = self.suco_ord
        else:
            comps_list = [[k] for k in range(self.comp.K)]
            order = self.comp_ord

        non_active = np.sum(self.comp.active_komp > 0.05, axis=0) == 0

        if not ks is None or sum(non_active) > 0:
            if ks is None:
                ks = np.nonzero(~non_active)[0]
            else:
                ks = [comp for k, comp in enumerate(self.comp_ord[k])
                      if k in ks and ~non_active[comp]]
                # Input ks relative to the on canonical order of components
            comps_list = [[k for k in comp if k in ks] for comp in comps_list]
            order = [order_ for order_ in order if len(comps_list[order_]) > 0]

        if not with_outliers:
            outliers = self.comp.mu_outliers

        if axs is None:
            fig, axs = plt.subplots(len(order), **figargs)

        for i, (ax, s) in enumerate(zip(axs, order)):
            comps = comps_list[s]
            for k in comps:
                mu_ks = self.comp.mupers[:, k, :]
                if not with_outliers:
                    mu_ks = mu_ks[~outliers[:, k], :]
                for j in range(self.comp.J):
                    ax.plot(range(self.comp.d), mu_ks[j, :], color=colors[k])
                ax.plot([0, self.comp.d-1], [.5, .5], color='grey')

            if i == len(order)-1:
                ax.axes.xaxis.set_ticks(range(self.comp.d))
                ax.set_xticklabels(self.marker_lab)
            else:
                ax.axes.xaxis.set_ticks([])
            if yscale:
                ax.axes.yaxis.set_ticks([.2, .8])
                ax.set_ylim(-.1, 1.1)
        return axs

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

    def eigvectors(self, D=2, axs=None, **figargs):
        '''
            Plots the first D eigenvectors for each component.
        '''
        if D is None:
            D = self.comp.d
        if axs is None:
            fig, axs = plt.subplots(self.comp.K, D, **figargs)
        for i, k in enumerate(self.comp_ord):
            sigmas = [self.comp.Sigmapers[j, k, :, :] for j in range(self.comp.J)]
            EVs = [np.linalg.eig(sigma) for sigma in sigmas if not np.isnan(sigma[0, 0])]
            Vs = [np.sqrt(EV[0]).reshape(1, -1)*EV[1] for EV in EVs]
            for dd in range(D):
                Vs_d = [V[:, dd] for V in Vs]
                s = np.argmax(sum([abs(V_d) for V_d in Vs_d]))
                Vs_d = [V*np.sign(V[s]) if V[s] != 0 else V for V in Vs_d]
                axs[i, dd].plot(np.vstack(Vs_d).T, color=self.comp_colors[k])
                axs[i, dd].set_ylim(-.2, .2)

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

    def latent(self, dim, ax=None, ks=None, plim=[0, 1], plotlab=False,
               plotlabx=False, plotlaby=False, plot_new_th=True,
               lw=2, lims=None):
        '''
            Plot visualization with ellipses of the latent components.
            Canonical colors are used (see HMplot).

            dim     -   which two dimensions to project on.
            ax     -   where to plot
            ks      -   which components to plot
            plim    -   only components with size within plim are plotted
            plotlab -   should labels be shown?
        '''
        if ax is None:
            f = plt.figure()
            ax = f.add_subplot(111)
        else:
            f = None

        if ks is None:
            ks = range(self.comp.K)

        non_active = np.sum(self.comp.active_komp > 0.05, axis=0) == 0
        ks = [comp for k, comp in enumerate(self.comp_ord)
              if k in ks and ~non_active[comp]]
            # input ks are given relative to canonical order
        #[self.comp_ord[k] for k in ks if ~non_active[k]]
        okcl = set.intersection(set(self.within_plim(plim)), set(ks))

        mus = [self.comp.mulat[k, :] for k in okcl]
        Sigmas = [self.comp.Sigmalat[k, :, :] for k in okcl]
        colors = [self.comp_colors[k] for k in okcl]

        q = plot.component_plot(mus, Sigmas, dim, ax, colors=colors, lw=lw)

        if hasattr(self.comp, 'new_thetas') and plot_new_th:
            ax.scatter(self.comp.new_thetas[:, dim[0]],
                       self.comp.new_thetas[:, dim[1]], s=40, c='k', marker='+')

        if plotlab or plotlabx:
            ax.set_xlabel(self.marker_lab[dim[0]], fontsize=16)
        if plotlab or plotlaby:
            ax.set_ylabel(self.marker_lab[dim[1]], fontsize=16)

        if not lims is None:
            self.set_lims(ax, lims, dim)

        return q

    def allsamp(self, dim, ax=None, ks=None, plim=[0, 1], js=None, names=None,
                plotlab=False, plotlabx=False, plotlaby=False,
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
            ax1 = fig.add_subplot(len(dimlist), 2, 2*m+1)
            ql = self.latent(dimlist[m], ax1, ks, plim)
            ax2 = fig.add_subplot(len(dimlist), 2, 2*m+2, sharex=ax1, sharey=ax1)
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

    def bhattacharyya_overlap_quotient(self, axs=None, **figargs):
        '''
            Diagnostic plot showing quotient between distance to correct latent
            center and distance to nearest wrong latent center.
        '''
        distquo = self.comp.get_latent_bhattacharyya_overlap_quotient()
        self.plot_diagnostics(
            distquo, 0, 3, 1, self.comp_ord, 'Bhattacharyya overlap quotient', axs=axs)

    def center_distance_quotient(self, axs=None, **figargs):
        '''
            Diagnostic plot showing quotient between distance to correct latent
            center and distance to nearest wrong latent center.
        '''
        distquo = self.comp.get_center_distance_quotient()
        self.plot_diagnostics(
            distquo, 0, 3, 1, self.comp_ord, 'Distance to mean quotient', axs=axs)

    def cov_dist(self, norm='F', axs=None, **figargs):
        '''
            Diagnostic plot showing distance between convariance matrices
            of the mixture components and the corresponding latent components.

            norm    -   which norm to use for computing the distance
        '''
        distF = self.comp.get_cov_dist(norm)
        self.plot_diagnostics(
            np.log10(distF), -5, 0, -3, self.comp_ord,
            'Covariance matrix distance (norm {})'.format(norm), False,
            axs=axs)

    @staticmethod
    def plot_diagnostics(diagn, ymin, ymax, ybar=None, order=None, name='',
                         log=False, axs=None, **figargs):
        J = diagn.shape[0]
        K = diagn.shape[1]
        if axs is None:
            fig, axs = plt.subplots(K, **figargs)
        if order is None:
            order = range(K)

        xloc = np.arange(J) + .5
        bar_width = .35
        axs[0].set_title(name)
        for k, ax in zip(order, axs):
            ax.bar(xloc, diagn[:, k], bar_width, log=log)
            ax.set_ylim(ymin, ymax)
            if not ybar is None:
                ax.plot([xloc[0], xloc[-1]], [ybar, ybar])
            ax.axes.yaxis.set_ticks([])
            ax.axes.xaxis.set_ticks([])


class TracePlot(object):

    def __init__(self, traces):
        self.traces = traces
        self.traces.plot = self

    @property
    def order(self):
        return self.traces.comp_ord

    def all(self, axs=None, yscale=True, **figargs):
        '''
            Plot trace plots of latent means and nus.
        '''
        if axs is None:
            fig, axs = plt.subplots(1, self.traces.K, **figargs)
        for k, ax in enumerate(axs):
            if k >= self.traces.K:
                return axs
            self.mulat(k, ax, yscale)
            ax.set_title(r'$\theta_{%d}$' % (k+1))
        return axs

    def mulat(self, k, ax=None, yscale=True):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        ax.plot(self.traces.ind, self.traces.get_mulat_k(self.order[k]))
        ax.set_xlim(0, self.traces.ind[-1])
        if yscale:
            ax.set_ylim(-.2, 1.2)
            ax.axes.yaxis.set_ticks([0.1, 0.9])
            ax.axvspan(0, self.traces.burnind[-1], facecolor='0.5', alpha=0.5)

    def nu(self, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        ax.plot(self.traces.ind, self.traces.get_nu())
        ax.set_xlim(0, self.traces.ind[-1])
        ax.set_yscale('log')
        ax.axes.yaxis.set_ticks([100, 1000])
        ax.axvspan(0, self.traces.burnind[-1], facecolor='0.5', alpha=0.5)
        ax.set_title(r'$\nu$')

    def nu_sigma(self, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        ax.plot(self.traces.ind, self.traces.get_nu_sigma())
        ax.set_xlim(0, self.traces.ind[-1])
        ax.axvspan(0, self.traces.burnind[-1], facecolor='0.5', alpha=0.5)
        ax.set_title(r'$r$')


class MimicPlot(object):

    def __init__(self, mimic):
        self.mimic = mimic
        self.realplot = FCplot(mimic.realsamp)
        self.synplot = FCplot(mimic.synsamp)

    @property
    def marker_lab(self):
        return self._marker_lab

    @marker_lab.setter
    def marker_lab(self, marker_lab):
        self.realplot.marker_lab = marker_lab
        self.synplot.marker_lab = marker_lab


class FCplot(object):

    def __init__(self, fcsample):
        self.fcsample = fcsample
        self.fcsample.plot = self

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
        plot.histnd(self.fcsample.get_data(Nsamp), bins, [0, 100], [5, 95],
                    fig, labels=self.marker_lab)
        if not xlim is None:
            for ax in fig.axes:
                ax.set_xlim(*xlim)
        if not ylim is None:
            for ax in fig.axes:
                if ax.get_ylim()[1] < 5:
                    ax.set_ylim(*ylim)

