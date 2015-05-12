# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 18:13:22 2015

@author: johnsson
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import plot

class ClustPlot(object):
    
    def __init__(self,clustering,colors,order):
        self.clust = clustering
        self.clust.plot = self
        self.colors = colors
        self.order = order
        self.pop_lab = None
        
    def set_marker_lab(self,marker_lab):
        self.marker_lab = marker_lab
        
    def set_population_lab(self,pop_lab):
        self.pop_lab = pop_lab
        
    def prob(self,fig=None,totplots=1,plotnbr=1):
        '''
            Plot probabilities of belonging to each cluster
        '''
        if fig is None:
            fig = plt.figure()
        nbr_cols = 2*totplots-1
        col_start = 2*(plotnbr-1)
            
        prob = self.clust.p
        J = prob.shape[0]
        K = prob.shape[1]

        for k in range(K):
            ax = fig.add_subplot(K,nbr_cols,k*nbr_cols + col_start+1)
            ax.scatter(range(J), prob[:,self.order[k]])
            ax.set_yscale('log')
            ax.set_ylim(1e-3,1)
            ax.axes.yaxis.set_ticks([1e-2,1e-1])
            xlim = ax.get_xlim()
            ax.plot([xlim[0],xlim[1]],[1e-2,1e-2],color='grey')
            ax.plot([xlim[0],xlim[1]],[1e-1,1e-1],color='grey')
            ax.axes.xaxis.set_ticks([])
            ax.set_xlim(-1,J)
    
        return fig
        
    def box(self,fig=None,totplots=1,plotnbr=1):
        '''
            Plot boxplots reprsenting quantiles for each cluster.
            NB! pooled data is used here.
        '''
        if fig is None:
            fig = plt.figure()
        quantiles = self.clust.get_quantiles((.01,.25,.5,.75,.99))
        print "quantiles = {}".format(quantiles)
        nbr_cols = 2*totplots - 1
        col_start = 2*(plotnbr-1)
        boxloc = (np.array(range(self.clust.d)) + .5)/(self.clust.d+1)
        boxw = (boxloc[1] - boxloc[0])/3.5
        ms = 10
        for k in range(self.clust.K):
            if not np.isnan(quantiles[self.order[k],0,0]):
                ax = fig.add_subplot(self.clust.K,nbr_cols,k*nbr_cols + col_start+1)
                for dd in range(self.clust.d):
                    plot.drawbox(quantiles[self.order[k],:,dd],boxloc[dd],boxw,ms,ax)
                ax.axes.xaxis.set_ticks(boxloc)
                xlim = ax.get_xlim()
                ax.plot([xlim[0],xlim[1]],[.5, .5],color='grey')
                if k < self.clust.K-1:
                    ax.set_xticklabels(['']*self.clust.d)
                else:
                    ax.set_xticklabels(self.marker_lab)
                ax.set_ylim(-.1,1.1)
                ax.axes.yaxis.set_ticks([.2,.8])
        if not self.pop_lab is None:
            ax.set_ylabel(self.pop_lab[self.order[k]])
        return fig
    
    def pdip(self,fig=None,colorbar=True):
        '''
            Plot p-value of Hartigan's dip test for each cluster.
            Results are plotted order by cluster size, with largest cluster
            in top.
        '''
        if fig is None:
            fig = plt.figure()
        pdipsum = self.clust.get_pdip_summary()
        S = ['Median','25th percentile','Minimum']
        for i,s in enumerate(S):
            ax = fig.add_subplot(1,3,i+1)
            p = ax.pcolormesh(pdipsum[s][self.order[::-1],:])
            p.set_clim(0,1)
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
            ax.set_ylim(0,self.clust.K)
            ax.set_xlabel(s)
        if colorbar:
            fig.colorbar(p)
        return fig

    def cum(self,j=None,fig=None):
        '''
            Empirical CDF for all the clusters in a sample j (or the pooled data)
        '''
        if fig is None:
            fig = plt.figure()
        alpha = np.linspace(0,1,200)
        quantiles = self.clust.get_quantiles(alpha,j)
        d = quantiles.shape[2]
        K = quantiles.shape[0]
        for k in range(K):
            for dd in range(d):
                ax = fig.add_subplot(K,d,k*d + dd+1)
                ax.plot(quantiles[k,:,dd],alpha)
                ax.set_ylim(0,1)
                ax.axes.yaxis.set_ticks([])
        return fig
    
    def qhist(self,j=None,ks=None,dds=None,fig=None,totplots=1,plotnbr=1):
        '''
            Histogram of quantiles for each cluster. Useful for inspecting dip.
        '''
        if fig is None:
            fig = plt.figure()
      
        alpha = np.linspace(0,1,500)
        quantiles = self.clust.get_quantiles(alpha,j,ks,dds)
        
        if ks is None:
            ks = range(self.K)
        ks_ord = [self.order[k] for k in ks]
        ks = ks_ord
        if dds is None:
            dds = range(self.clust.d)
        d = len(dds)
        print "d = {}".format(d)
        nbr_cols = d*totplots + totplots-1
        print "nbr_cols = {}".format(nbr_cols)
        col_start = (d+1)*(plotnbr-1)

        for ik,k in enumerate(ks):
            for id,dd in enumerate(dds):
                ax = fig.add_subplot(self.clust.K,nbr_cols,k*nbr_cols + col_start + dd + 1)
                ax.hist(quantiles[ik,:,id],bins = 50,color=self.colors[k])
                ax.axes.xaxis.set_ticks([])
                ax.axes.yaxis.set_ticks([])
        return fig
        
        
    def qhist_dipcrit(self,q=.25,fig=None,totplots=1,plotnbr=1):
        '''
            Histogram of quantiles for the sample with dip at the qth quantile.
            I.e. q=.25 displays quantiles for the sample which has its dip at the
            0.25th quantile among the samples.
        '''
        if fig is None:
            fig = plt.figure()
        alpha = np.linspace(0,1,500)
        K = self.clust.K
        d = self.clust.d
        pdiplist = self.clust.get_pdip() 

        nbr_cols = d*totplots + totplots-1
        col_start = (d+1)*(plotnbr-1)  
        #qind = np.int(K*q)-1
        jq = np.zeros((K,d),dtype='i')
        for ik,k in enumerate(self.order):
            pd = pdiplist[k]
            K_loc = sum(~np.isnan(pd[:,0]))
            for dd in range(d):
                qind = max(np.int(K_loc*q)-1,0)
                jq[k,dd] = np.argsort(pd[:,dd])[qind]
                quantiles = self.clust.get_quantiles(alpha,jq[k,dd],[k],[dd])[0,:,0]
                ax = fig.add_subplot(self.clust.K,nbr_cols,ik*nbr_cols + col_start + dd + 1)
                ax.hist(quantiles,bins = 50,color=self.colors[k],range=(-0.1,1.4))
                ax.set_xlim((-0.1,1.4))
                ax.axes.xaxis.set_ticks([0,0.5,1])
                ax.axes.yaxis.set_ticks([])
        return fig

    def chist_allsamp(self,min_clf,dd,ks,fig=None,ncol=4):
        '''
            Histogram of data points with at least min_clf probability of belonging to certain clusters.
            The clusters are displayed with their canonical colors.
            
            A panel of plots with ncol columns is showing this for all samples.
        '''
        if fig is None:
            fig = plt.figure()
        nrow = np.ceil(self.clust.J/ncol)
        for j in range(self.clust.J):
            ax = fig.add_subplot(nrow,ncol,j+1)
            self.chist(min_clf,dd,j,ks,ax)
        return fig

    def chist(self,min_clf,dd,j=None,ks=None,ax=None):
        '''
            Histogram of data points with at least min_clf probability of belonging to certain clusters.
            The clusters are displayed with their canonical colors (see BMPlot).
        '''
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        if ks is None:
            ks = range(self.clust.K)

        for k in ks:
            data = self.clust.get_data_kdj(min_clf,self.order[k],dd,j)
            if len(data) > 0:
                ax.hist(data,bins=50,color=self.colors[self.order[k]],alpha = .7,range = (-0.1,1.4))
        return ax

    def scatter(self,dim,j,fig=None):
        '''
            Plots the scatter plot of the data over dim.
            Clusters are plotted with their canonical colors (see BMPlot).
        '''
        if fig is None:
            fig = plt.figure()
        ax = fig.add_subplot(111)
    			
        data = self.clust.data[j][:,dim]
        x = self.clust.sample_x(j)

        if len(dim) == 2:
            for k in range(self.clust.K):
                ax.plot(data[x==k,0],data[x==k, 1],'+',label='k = %d'%(k+1),color=self.colors[k])
            ax.plot(data[x==self.clust.K,0],data[x==self.clust.K,1],'+',label='outliers',color='black')
        
        elif len(dim) == 3:
            ax = fig.gca(projection='3d')
            for k in range(self.clust.K):
                ax.plot(data[x==k,0],data[x==k,1],data[x==k,2],'+',label='k = %d'%(k+1),color=self.colors[k])
            ax.plot(data[x==self.clust.K,0],data[x==self.clust.K,1],data[x==self.clust.K,2],'+',label='outliers',color='black')
    						
        return fig, ax
        
class CompPlot(object):
    
    def __init__(self,components,comp_colors,comp_ord,suco_ord):
        self.comp = components
        self.comp.plot = self
        self.comp_colors = comp_colors
        self.comp_ord = comp_ord
        self.suco_ord = suco_ord
        
    def set_marker_lab(self,marker_lab):
        self.marker_lab = marker_lab

    def set_sampnames(self,names):
        self.sampnames = names
        
    def center(self,suco=True,fig=None,totplots=1,plotnbr=1,yscale=False):
        '''
            The centers of all components (mu param) are plotted along one dimension.
            
            If suco=True, components belonging to the same super component are
            plotted in the same panel.
        '''
        if fig is None:
            fig = plt.figure()
            
        if suco:
            comps_list = self.comp.mergeind
            order = self.suco_ord
        else:
            comps_list = [[k] for k in range(self.comp.K)]
            order = self.comp_ord
    
        nbr_cols = 2*totplots-1
        col_start = 2*(plotnbr-1)
    
        S = len(order)
        
        for s in range(S):
            comps = comps_list[order[s]]
            #print "comps = {}".format(comps)
            ax = fig.add_subplot(S,nbr_cols,s*nbr_cols + col_start+1)
            for k in comps:
                mu_ks = self.comp.mupers[:,k,:]
                for j in range(self.comp.J):
                    ax.plot(range(self.comp.d),mu_ks[j,:],color=self.comp_colors[k])
                ax.plot([0,self.comp.d-1],[.5,.5],color='grey')
            if s == S-1:
                ax.axes.xaxis.set_ticks(range(self.comp.d))
                ax.set_xticklabels(self.marker_lab)
            else:
                ax.axes.xaxis.set_ticks([])
            if not yscale:
                ax.axes.yaxis.set_ticks([])
                ax.set_ylim(0,1)
            else:
                ax.axes.yaxis.set_ticks([.2,.8])
                ax.set_ylim(-.1,1.1)		
        return fig
    
    def center3D(self,dim,fig=None):
        '''
            Plots the centers (mu) of the components in three dimensions.
            
            dim     -   dimensions on which to project centers.
        '''
        if fig is None:
            fig = plt.figure()
            
        ax = fig.gca(projection='3d')
    
        for k in range(self.comp.K):
            mus = self.comp.mupers[:,k,:]
            ax.scatter(mus[:,dim[0]], mus[:,dim[1]], mus[:,dim[2]], marker='.', color = self.comp_colors[k], s=50) 
            
        ax.axes.xaxis.set_ticks([.1,.5,.9])
        ax.axes.zaxis.set_ticks([.1,.5,.9])
        ax.axes.yaxis.set_ticks([.1,.5,.9])
        ax.view_init(30,165)
        
        ax.set_xlabel(self.marker_lab[dim[0]])
        ax.set_ylabel(self.marker_lab[dim[1]])
        ax.set_zlabel(self.marker_lab[dim[2]])
            
        return ax

    def latent(self,dim,ax=None,ks=None,plim=[0,1],plotlab = False,plot_new_th=True,lw=2):
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
        okcl = set.intersection(set(self.within_plim(plim)),set(ks))
        
        mus = [self.comp.mulat[k,:] for k in okcl]
        Sigmas = [self.comp.Sigmalat[k,:,:] for k in okcl]
        colors = [self.comp_colors[k] for k in okcl]

        q = plot.component_plot(mus,Sigmas,dim,ax,colors=colors,lw=lw)
        
        if hasattr(self.comp,'new_thetas') and plot_new_th:
            ax.scatter(self.comp.new_thetas[:,dim[0]],self.comp.new_thetas[:,dim[1]],s=40,c='k',marker='+')
        
        if plotlab:
            ax.set_xlabel(self.marker_lab[dim[0]],fontsize=16)
            ax.set_ylabel(self.marker_lab[dim[1]],fontsize=16)
        return q

    def allsamp(self,dim,ax=None,ks=None,plim=[0,1],js=None,names=None,plotlab=False,plot_new_th=True,lw=1):

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

        okcl = set.intersection(set(self.within_plim(plim)),set(ks))
        
        muspers = [[self.comp.mupers[j,k,:] for k in okcl] for j in js]
        Sigmaspers = [[self.comp.Sigmapers[j,k,:,:] for k in okcl] for j in js]
        colors = [self.comp_colors[k] for k in okcl]

        q = plot.pers_component_plot(muspers,Sigmaspers,dim,ax,colors=colors,lw=lw)
 
        if hasattr(self.comp,'new_thetas') and plot_new_th:
            ax.scatter(self.comp.new_thetas[:,dim[0]],self.comp.new_thetas[:,dim[1]],s=40,c='k',marker='+')

        if plotlab:
            ax.set_xlabel(self.marker_lab[dim[0]],fontsize=16)
            ax.set_ylabel(self.marker_lab[dim[1]],fontsize=16)
        return q
        
    def within_plim(self,plim):
        okp = np.array([True]*self.comp.K)
        for suco in self.comp.mergeind:
            p_suco = np.mean(np.sum(self.comp.p[:,suco],axis=1))
            if p_suco < plim[0] or p_suco > plim[1]:
                okp[suco] = False
            okcl = np.nonzero(okp)[0]
        return okcl

    def latent_allsamp(self,dimlist,fig=None,ks=None,plim=[0,1],js=None,plotlab=True):
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
            ax1 = fig.add_subplot(len(dimlist),2,2*m+1)#plt.subplot2grid((2, 2), (m, 0))
            ql = self.latent(dimlist[m],ax1,ks,plim)
            ax2 = fig.add_subplot(len(dimlist),2,2*m+2)#plt.subplot2grid((2, 2), (m, 1))
            qa = self.allsamp(dimlist[m],ax2,ks,plim,js)

            if m == 0:
                ax1.set_title(self.marker_lab[dimlist[m][0]],fontsize=16)
                ax2.set_title(self.marker_lab[dimlist[m][0]],fontsize=16)
            else:
                ax1.set_xlabel(self.marker_lab[dimlist[m][0]],fontsize=16)
                ax2.set_xlabel(self.marker_lab[dimlist[m][0]],fontsize=16)
            ax1.set_ylabel(self.marker_lab[dimlist[m][1]],fontsize=16)

            plot.set_component_plot_tics([ax1,ax2],plot.mergeQ(ql,qa))
        return fig
    
    def center_distance_quotient(self,fig=None,totplots=1,plotnbr=1):
        '''
            Diagnostic plot showing quotient between distance to correct latent
            center and distance to nearest wrong latent center.
        '''
        if fig is None:
            fig = plt.figure()
        distquo = self.comp.get_center_distance_quotient()
        fig = plot.plot_diagnostics(distquo,0,3,1,self.comp_ord,'Distance to mean quotient',fig=fig,totplots=totplots,plotnbr=plotnbr)
        return fig
    
    def cov_dist(self,norm='F',fig=None,totplots=1,plotnbr=1):
        '''
            Diagnostic plot showing distance between convariance matrices
            of the mixture components and the corresponding latent components.
            
            norm    -   which norm to use for computing the distance
        '''
        distF = self.comp.get_cov_dist(norm)
        plot.plot_diagnostics(np.log10(distF),-5,0,-3,self.comp_ord,'Covariance matrix distance (norm {})'.format(norm),False,fig=fig,totplots=totplots,plotnbr=plotnbr)

class TracePlot(object):
    
    def __init__(self,traces,order):
        self.traces = traces
        self.traces.plot = self
        self.order = order
        
    def all(self,fig = None):
        '''
            Plot trace plots of latent means and nus.
        '''
        if fig == None:
            fig = plt.figure()
        for k in range(self.traces.K):
            ax = plt.subplot2grid((1, self.traces.K+1), (0, k))
            self.mulat(k,ax)
            ax.set_title('theta_'+'{}'.format(k+1))
        ax = plt.subplot2grid((1, self.traces.K+1), (0, k+1))
        self.nu(ax)
        ax.set_title('nu',fontsize=16)
        return fig,ax
        
    def mulat(self,k,ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        ax.plot(self.traces.ind, self.traces.get_mulat_k(self.order[k]))
        ax.set_xlim(0,self.traces.ind[-1])
        ax.set_ylim(-.2,1.2)
        ax.axes.yaxis.set_ticks([0.1,0.9])
        plt.axvspan(0, self.traces.burnind[-1], facecolor='0.5', alpha=0.5)
        
    def nu(self,ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        ax.plot(self.traces.ind, self.traces.get_nu())
        ax.set_xlim(0,self.traces.ind[-1])
        ax.set_yscale('log')
        ax.axes.yaxis.set_ticks([100, 1000])
        plt.axvspan(0, self.traces.burnind[-1], facecolor='0.5', alpha=0.5)

class MimicPlot(object):
    
    def __init__(self,mimic):
        self.mimic = mimic
        self.realplot = FCplot(mimic.realsamp)
        self.synplot = FCplot(mimic.synsamp)
        
    def set_marker_lab(self,marker_lab):
        self.realplot.set_marker_lab(marker_lab)
        self.synplot.set_marker_lab(marker_lab)
        
class FCplot(object):
    
    def __init__(self,fcsample):
        self.fcsample = fcsample
        self.fcsample.plot = self
        
    def set_marker_lab(self,marker_lab):
        self.marker_lab = marker_lab

    def hist2d(self,dim,Nsamp=None,bins=50,ax=None,xlim=None,ylim=None):
        '''
            Plot 2D histograms of a given sample (synthetic or real).
        '''
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        data = self.fcsample.get_data(Nsamp)
        ax.hist2d(data[:,dim[0]],data[:,dim[1]],bins = bins,norm=colors.LogNorm(),vmin=1)
        ax.patch.set_facecolor('white')
        if not xlim is None:
            ax.set_xlim(*xlim)
        if not ylim is None:
            ax.set_ylim(*ylim)     

    def histnd(self,Nsamp=None,bins=50,fig=None,xlim=None,ylim=None):
        '''
            Plot panel of 1D and 2D histograms of a given sample (synthetic or real).
        '''
        if fig is None:
            fig = plt.figure()
        plot.histnd(self.fcsample.get_data(Nsamp),bins,[0, 100],[5,95],fig,
               labels=self.marker_lab)
        if not xlim is None:
            for ax in fig.axes:
                ax.set_xlim(*xlim)
        if not ylim is None:
            for ax in fig.axes:
                if ax.get_ylim()[1] < 5:
                    ax.set_ylim(*ylim)
        
class BMplot(object):
    
    def __init__(self,bmres,marker_lab = None):
        self.bmres = bmres
        self.comp_colors,self.suco_colors,self.comp_ord,self.suco_ord = self.get_colors_and_order()
        
        self.clp_nm = ClustPlot(bmres.clust_nm,self.comp_colors,self.comp_ord)
        if hasattr(bmres,'clust_m'):
            self.clp_m = ClustPlot(bmres.clust_m,self.suco_colors,self.suco_ord)
        self.cop = CompPlot(bmres.components,self.comp_colors,self.comp_ord,self.suco_ord)
        self.trp = TracePlot(bmres.traces,self.comp_ord)
        
        self.mcsp = {}
        for mimic_key in bmres.mimics:
            mimic = bmres.mimics[mimic_key]
            self.mcsp[mimic_key] = MimicPlot(mimic)
            #self.mcsp.append(MimicPlot(mimic))
            
        if marker_lab is None:
            try:
                marker_lab = bmres.meta_data.marker_lab
            except:
                marker_lab = ['']*bmres.d
        self.set_marker_lab(marker_lab)
        self.set_sampnames(bmres.meta_data.samp['names'])
        
    def set_marker_lab(self,marker_lab):
        self.marker_lab = marker_lab
        self.clp_nm.set_marker_lab(marker_lab)
        if hasattr(self,'clp_m'):
            self.clp_m.set_marker_lab(marker_lab)
        self.cop.set_marker_lab(marker_lab)
        for mc in self.mcsp:
            self.mcsp[mc].set_marker_lab(marker_lab)

    def set_sampnames(self,names):
        self.sampnames = names
        self.cop.set_sampnames(names)

    def set_population_lab(self,pop_lab):
        order = np.argsort(self.suco_ord)
        self.pop_lab = [pop_lab[k] for k in order]
        self.clp_nm.set_population_lab(pop_lab)
        self.clp_m.set_population_lab(pop_lab)

    def get_colors_and_order(self):
        '''
            Get order of components (first orderd by super component size, then by
            individual component size) and colors to use for representing components.
            
            This gives canonical ordering and colors for other plots.
        '''      
        comp_col,suco_col,comp_ord,suco_ord = self.bmres.get_colors_and_order()
        
        #maxnbrsucocol = 8
        ##for s,suco in enumerate(self.bmres.mergeind):
        ##    sc_ord = np.argsort(-np.array(np.sum(self.bmres.p,axis=0)[suco]))
        ##    self.bmres.mergeind[s] = [suco[i] for i in sc_ord] # Change order within each super component
        ##prob_mer = [np.sum(self.bmres.p[:,scind]) for scind in self.bmres.mergeind]
        ##suco_ord = np.argsort(-np.array(prob_mer))
        #mergeind_sort = [self.bmres.mergeind[i] for i in suco_ord]
        ##print "mergeind_sort = {}".format(mergeind_sort)
        ##comp_ord = [ind for suco in mergeind_sort for ind in suco]
        #cm = plt.get_cmap('gist_rainbow')
        #nbrsucocol = min(maxnbrsucocol,len(suco_ord))  
        #suco_col = [(0,0,0)]*len(suco_ord)
        #colors = [(0,0,0)]*len(comp_ord)
        #for s,suco in enumerate(mergeind_sort):
        #    #print "(s % nbrsucocol)/nbrsucocol = {}".format((s % nbrsucocol)/nbrsucocol)
        #    suco_col[suco_ord[s]] = cm((s % nbrsucocol)/nbrsucocol)
        #    if s > maxnbrsucocol:
        #        suco_col[suco_ord[s]] = suco_col[suco_ord[s]][:3]+(0.5,)
        #    for i,k in enumerate(suco):
        #        colors[k] = plot.black_ip(suco_col[suco_ord[s]],i,len(suco))
        return comp_col,suco_col,comp_ord,suco_ord

    def pca_biplot(self,comp,ax=None,poplabsh=None,sampmarkers=None):
        '''
            PCA biplot of mixture component probabilities. Sample groups are 
            determined by meta_data 'donorid'.
            
            comp        -   which principal components to plot
            ax          -   where to plot
            poplabsh    -   shift of population labels
            sampmarkers -   markers to use for samples
        '''
        #if sampmarkers is None:
        #    sampmarkers = [(4,0,45),(3,0),(0,3),(4,2)]
        #if poplabsh is None:
        #    poplabsh = [[0,0],[0,-.02],[0,0],[-.1,0],[.22,0],[.06,-.06]]
        if not hasattr(self,'pop_lab'):
            self.pop_lab = None
        plot.pca_biplot(self.bmres.clust_m.p,comp,ax,varcol=self.suco_colors,varlabels=self.pop_lab,
                   varlabsh=poplabsh,sampleid=self.bmres.meta_data.samp['donorid'],sampmarkers=sampmarkers)        

    def pca_screeplot(self,ax=None):
        plot.pca_screeplot(self.bmres.p,ax)

    def component_fit(self,plotdim,name='pooled',lim=[-.2,1.2],bins=100):
        fig = plt.figure()
        labels = self.bmres.meta_data.marker_lab
        if name == 'pooled':
            names = self.sampnames
        else:
            names = [name]
        for m in range(len(plotdim)):
            ax = plt.subplot2grid((len(plotdim), 4), (m, 0))
            ql = self.cop.latent(plotdim[m],ax=ax)
            ax.set_xlabel(labels[plotdim[m][0]],fontsize=16)
            ax.set_ylabel(labels[plotdim[m][1]],fontsize=16)
            ax.set_xlim(*lim)
            ax.set_ylim(*lim)
            
            ax = plt.subplot2grid((len(plotdim), 4), (m, 1))
            qa = self.cop.allsamp(plotdim[m],names=names,ax=ax)
            ax.set_xlabel(labels[plotdim[m][0]],fontsize=16)
            ax.set_xlim(*lim)
            ax.set_ylim(*lim)
            
            ax = plt.subplot2grid((len(plotdim), 4), (m, 2))
            self.mcsp[name].realplot.hist2d(plotdim[m],bins=bins,ax=ax,xlim=lim,ylim=lim)
            if m == 0:
                ax.set_title(name+'(real)')
            #ax.set_xlim(*lim)
            #ax.set_ylim(*lim)
            
            ax = plt.subplot2grid((len(plotdim), 4), (m, 3))
            self.mcsp[name].synplot.hist2d(plotdim[m],bins=bins,ax=ax,xlim=lim,ylim=lim)
            if m == 0:
                ax.set_title(name+'(synthetic)')
            #ax.set_xlim(*lim)
            #ax.set_ylim(*lim)

        return fig



    





    







