# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 18:13:22 2015

@author: johnsson
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pyl
import matplotlib.ticker as ticker
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import colorsys

def black_ip(color,n,N):
    '''
        Interpolate between a color and black.
        
        color   -   color
        n       -   shade rank
        N       -   total number of possible shades
    '''
    if len(color) == 4:
        r,g,b,al = color
    else:
        r,g,b = color
    h,s,v = colorsys.rgb_to_hsv(r, g, b)
    return colorsys.hsv_to_rgb(h,s,(N+3-n)/(N+3)*v)
    
def drawbox(quantiles,boxloc,boxw,ms,ax):
    low = quantiles[0]
    blow = quantiles[1]
    bmid = quantiles[2]
    bupp = quantiles[3]
    upp = quantiles[4]
    plt.plot([boxloc,boxloc],[low,upp],marker = '_',color='blue',ms=ms)
    plt.plot([boxloc-boxw/2,boxloc+boxw/2,boxloc+boxw/2,boxloc-boxw/2,boxloc-boxw/2],[blow,blow,bupp,bupp,blow],color='blue')
    plt.plot([boxloc-boxw/2,boxloc+boxw/2],[bmid,bmid],color='blue')

def visualEigen(Sigma, mu, dim):
    Sigma = Sigma[np.ix_(dim,dim)]
    E, V = linalg.eig(Sigma)
    t = np.linspace(0,2*np.pi,100)
    e = np.array([np.cos(t), np.sin(t)])
    V2 = np.sqrt(E)*V
    VV = np.dot(e.transpose(),V2.transpose()) + mu[dim].transpose()
    return VV
    
def component_plot(mus,Sigmas,dim,ax,colors=None):
    q_y = [np.inf,-np.inf]
    q_x = [np.inf,-np.inf]
    K = len(mus)
    if colors is None:
        colors = ['black']*K
    for k in range(K):
        if not np.isnan(mus[k][0]):
            plres = visualEigen(Sigmas[k], mus[k], [dim[0],dim[1]]) 
            q_res = np.percentile(plres[:,0], 50)
            q_x[0] = min(q_x[0],q_res)
            q_x[1] = max(q_x[1],q_res)
            q_res = np.percentile(plres[:,1], 50)
            q_y[0] = min(q_y[0],q_res)
            q_y[1] = max(q_y[1],q_res)
            ax.plot(plres[:,0],plres[:,1],'-',color = colors[k],linewidth=2)    
    return {'q_x': q_x, 'q_y': q_y}
    
def pers_component_plot(muspers,Sigmaspers,dim,ax,colors=None):
    q_y = [np.inf,-np.inf]
    q_x = [np.inf,-np.inf]
    q = {'q_x': q_x, 'q_y': q_y}
    for j in range(len(muspers)):
        qnew = component_plot(muspers[j],Sigmaspers[j],dim,ax,colors)
        q = mergeQ(q,qnew)
    return q

def mergeQ(q1,q2):
    q_x = [0, 0]
    q_y = [0, 0]
    q_x[0] = min(q1['q_x'][0],q2['q_x'][0])
    q_x[1] = max(q1['q_x'][1],q2['q_x'][1])
    q_y[0] = min(q1['q_y'][0],q2['q_y'][0])
    q_y[1] = max(q1['q_y'][1],q2['q_y'][1])
    return {'q_x': q_x, 'q_y': q_y}

def plot_diagnostics(diagn,ymin,ymax,ybar = None,name = '',log=False,fig=None,totplots=1,plotnbr=1):
    if fig is None:
        fig = plt.figure()
    J = diagn.shape[0]
    K = diagn.shape[1]
    nbr_cols = 2*totplots - 1
    col_start = 2*(plotnbr - 1) 
    xloc = np.arange(J) + .5
    bar_width = .35
    for k in range(K):
        ax = fig.add_subplot(K,nbr_cols,k*nbr_cols + col_start + 1)
        if k == 0:
            plt.title(name)
        ax.bar(xloc,diagn[:,k],bar_width,log=log)
        ax.set_ylim(ymin,ymax)
        if not ybar is None:
            ax.plot([xloc[0],xloc[-1]],[ybar,ybar])
        ax.axes.yaxis.set_ticks([])
        ax.axes.xaxis.set_ticks([])
    return fig
    
def pca_biplot(data,comp,ax=None,varcol=None,varlabels=None,varlabsh=None,sampleid=None,sampmarkers=None):
    markers = [(4,0,45),(3,0),(0,3),(4,2)]
    if ax == None:
    	f = plt.figure()
    	ax = f.add_subplot(111)
    else:
    	f = None
     
    K = data.shape[1]
    if varcol is None:
        varcol = ['black']*K
    if varlabels is None:
        varlabels = ['']*K
    if varlabsh is None:
        varlabsh = [0,0]*K
    if sampleid is None:
        if sampmarkers is None:
            sampmarkers = [(4,2)]
        sampleid = [0]*data.shape[0]
    if sampmarkers is None:
        sampmarkers = [(k,2) for k in range(len(set(sampleid)))]
        
    # Compute plot data
    data = data - np.mean(data,0)
    u, s, v = np.linalg.svd(data,full_matrices=0)
    s /= np.max(s) #Normalizing to make plot nicer
    sw = np.dot(u[:,comp],np.diag(s[comp]))
    vw = v[comp,:].transpose()

    # Compute variance contained in two first components
    lam = s * s
    lam /= sum(lam) 
    print "Proportion of variance by components in biplot: {}".format(sum(lam[comp]))

    ax.axis('off')

    # Plot sample data
    for j in range(data.shape[0]):
        ax.scatter(sw[j,0],sw[j,1],marker=sampmarkers[sampleid[j]],edgecolors='black',facecolors='none',s=80)

    # Plot variable data
    for k in range(data.shape[1]):
        ax.plot([0,vw[k,0]],[0,vw[k,1]],color=black_ip(varcol[k],6,10))
        ax.scatter(vw[k,0],vw[k,1],s=40,color=varcol[k])
        shift = 0.04*np.sign(vw[k,:])
        shift += varlabsh[k]
        ax.annotate(varlabels[k],vw[k,:]+shift,ha='center',va='center')

    # Set plot limits
    sc = 1.15
    ax.set_xlim([sc*min(min(vw[:,0]),min(sw[:,0])),sc*max(max(vw[:,0]),max(sw[:,0]))])
    ax.set_ylim([sc*min(min(vw[:,1]),min(sw[:,1])),sc*max(max(vw[:,1]),max(sw[:,1]))])   
    plt.plot()
    return f,ax
    
def pca_screeplot(data,ax=None):
    if ax == None:
        f = plt.figure()
        ax = f.add_subplot(111)
    else:
        f = None
    data = data - np.mean(data,0)
    u, s, v = np.linalg.svd(data,full_matrices=0)
    lam = s * s
    lam /= sum(lam) 
    ax.bar(np.arange(len(lam)) + 0.5, lam)
    ax.set_ylim(0,1)
    
def histnd(dat, bins, quan = [0.5,99.5], quan_plot = [5, 95], f = None, labels = None):

    nv = np.shape(dat)[1]
    count = 0

    if f == None:
        f = plt.figure()

    gs = gridspec.GridSpec(nv, nv)
    for i in range(nv):
        ax =  f.add_subplot(gs[nv*i + i])
        count += 1	 
        index_i = (dat[:,i] > np.percentile(dat[:,i], quan[0])) * (dat[:,i] < np.percentile(dat[:,i], quan[1]))
        q1 = np.percentile(dat[index_i ,i], quan_plot[0])
        q2 = np.percentile(dat[index_i ,i], quan_plot[1])
        ax.hist(dat[index_i ,i], bins = bins)
        n_bins = ax.hist(dat[index_i ,i], bins = bins,facecolor='black',edgecolor='black')[0]
        ax.axes.xaxis.set_ticks(np.linspace(q1,q2,num=4))
        ax.axes.yaxis.set_ticks(np.ceil(np.linspace(0,np.max(n_bins),num=4)))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        ax.tick_params(axis='both', which='major', labelsize=8)
        xlims = np.percentile(dat[:,i], quan)
        ax.set_xlim(xlims[0],xlims[1])
        if not labels is None:
            ax.set_xlabel(labels[i])
        for j in range(i+1,nv):
            index_j = (dat[:,j] > np.percentile(dat[:,j], quan[0]) ) * (dat[:,j] < np.percentile(dat[:,j], quan[1]))
            q1_y = np.percentile(dat[index_j*index_i ,j], quan_plot[0])
            q2_y = np.percentile(dat[index_j*index_i ,j], quan_plot[1])
            q1_x = np.percentile(dat[index_j*index_i ,i], quan_plot[0])
            q2_x = np.percentile(dat[index_j*index_i ,i], quan_plot[1])
            ax = f.add_subplot(gs[nv*i + j])
            count += 1
            dat_j  = dat[index_i*index_j ,j]
			#dat_j[dat_j == 0] = 1.
            dat_i  = dat[index_i*index_j ,i]
			#dat_i[dat_i == 0] = 1.
            
            ax.hist2d(dat_j, dat_i, bins = bins, norm=colors.LogNorm(),vmin=1)
            ax.patch.set_facecolor('white')
            ax.axes.xaxis.set_ticks(np.linspace(q1_y, q2_y,num=4))
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
            ax.axes.yaxis.set_ticks(np.linspace(q1_x, q2_x,num=4))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
            ax.tick_params(axis='both', which='major', labelsize=8)
            if (not (labels is None)) and (j == (nv-1)):
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(labels[i])
                
    f.subplots_adjust(wspace = .25)
    f.subplots_adjust(hspace = .25)	
    
    return f
    
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
        
    def prob(fig=None,totplots=1,plotnbr=1):
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
        if fig is None:
            fig = plt.figure()
        quantiles = self.clust.get_quantiles((.01,.25,.5,.75,.99))
        nbr_cols = 2*totplots - 1
        col_start = 2*(plotnbr-1)
        boxloc = (np.array(range(self.clust.d)) + .5)/(self.clust.d+1)
        boxw = (boxloc[1] - boxloc[0])/3.5
        ms = 10
        for k in range(self.clust.K):
            if not np.isnan(quantiles[k,0,0]):
                ax = fig.add_subplot(self.clust.K,nbr_cols,k*nbr_cols + col_start+1)
                for dd in range(self.clust.d):
                    drawbox(quantiles[self.order[k],:,dd],boxloc[dd],boxw,ms,ax)
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
        if colorbar:
            fig.colorbar(p)
        return fig

    def cum(self,j=None,fig=None):
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
    
    def hist(self,j=None,ks=None,dds=None,fig=None,totplots=1,plotnbr=1):
        if fig is None:
            fig = plt.figure()
        nbr_cols = d*totplots + totplots-1
        col_start = (d+1)*(plotnbr-1)        
        alpha = np.linspace(0,1,500)
        quantiles = self.clust.get_quantiles(alpha,j,ks,dds)
        
        if ks is None:
            ks = range(self.K)
        if dds is None:
            dds = range(self.d)

        for ik,k in enumerate(ks):
            for id,dd in enumerate(dds):
                ax = fig.add_subplot(self.clust.K,nbr_cols,k*nbr_cols + col_start + dd + 1)
                ax.hist(quantiles[ik,:,id],bins = 50)
                ax.axes.xaxis.set_ticks([])
                ax.axes.yaxis.set_ticks([])
        return fig
        
        
    def hist_dipcrit(self,q=.25,fig=None,totplots=1,plotnbr=1):
        if fig is None:
            fig = plt.figure()
        K = self.clust.K
        d = self.clust.d
        pdiplist = self.clust.get_pdip() 

        qind = np.int(K*q)-1
        jq = np.zeros((K,d),dtype='i')
        for k in range(K):
            pd = pdiplist[k]
            for dd in range(d):
                jq[k,dd] = np.argsort(pd[:,dd])[qind]
                self.histplot(j=jq,ks=[k],dds=[dd],fig=fig,totplots=totplots,plotnbr=plotnbr)
        return fig

    def scatter(dim, j, fig=None):
        '''
            Plots the scatter plot of the data over dim
            and assigning each class a different color
        '''
        if fig is None:
            fig = plt.figure()
        ax = fig.add_subplot(111)
    			
        data = self.clust.data[j][:,dim]
        x = self.clust.sample_x(j)

        if len(dim) == 2:
            for k in range(self.clust.K):
                ax.plot(data[x==k,0],data[x==k, 1],'+',label='k = %d'%(k+1),color=self.colors[k])
            ax.plot(data[x==self.clust.K,0],data[x==self.clust.K,1],label='outliers',color='black')
        
        elif len(dim) == 3:
            ax = fig.gca(projection='3d')
            for k in range(self.clust.K):
                ax.plot(data[x==k,0],data[x==k,1],data[x==k,2],'+',label='k = %d'%(k+1),color=self.colors[k])
            ax.plot(data[x==self.clust.K,0],data[x==self.clust.K,1],data[x==self.clust.K,2],label='outliers',color='black')
    						
        return f, ax
        
class CompPlot(object):
    
    def __init__(self,components,comp_colors,comp_ord,suco_ord):
        self.comp = components
        self.comp.plot = self
        self.comp_colors = comp_colors
        self.comp_ord = comp_ord
        self.suco_ord = suco_ord
        
    def set_marker_lab(self,marker_lab):
        self.marker_lab = marker_lab
        
    def center(self,suco=True,fig=None,totplots=1,plotnbr=1,yscale=False):

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
            print "comps = {}".format(comps)
            ax = fig.add_subplot(S,nbr_cols,s*nbr_cols + col_start+1)
            for k in comps:
                mu_ks = self.comp.mupers[:,k,:]
                for j in range(self.components.J):
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

    def latent_components(self,dim,ks=None,plim=[0,1],ax=None):
        #print "suco_list = {}".format(suco_list)
        if ax is None:
    		f = plt.figure()
    		ax = f.add_subplot(111)
        else:
    		f = None

        if ks is None:
            ks = range(res.K)            
        okcl = np.nonzero((np.mean(self.comp.p,axis=0) > plim[0]) * (np.mean(self.comp.p,axis=0) < plim[1]))[0]
        okcl = set.intersection(set(okcl),set(ks))
        
        mus = [self.comp.mulat[k,:] for k in okcl]
        Sigmas = [self.comp.Sigmalat[k,:,:] for k in okcl]
        colors = [self.comp_colors[k] for k in okcl]

        q = component_plot(mus,Sigmas,dim,ax,colors=colors)
        return q

    def allsamp_components(self,dim,ks=None,plim=[0,1],ax=None):
        if ax == None:
            f = plt.figure()
            ax = f.add_subplot(111)
        else:
            f = None
            
        if ks is None:
            ks = range(res.K)         
        okcl = np.nonzero((np.mean(self.comp.p,axis=0) > plim[0]) * (np.mean(self.comp.p,axis=0) < plim[1]))[0]
        okcl = set.intersection(set(okcl),set(ks))
        
        muspers = [[self.comp.mupers[j,k,:] for k in okcl] for j in range(res.J)]
        Sigmaspers = [[self.comp.Sigmapers[j,k,:,:] for k in okcl] for j in range(res.J)]
        colors = [self.comp_colors[k] for k in okcl]

        q = pers_component_plot(muspers,Sigmaspers,dim,ax,colors=colors)
        return q
        
    def center_distance_quotient(self,fig=None,totplots=1,plotnbr=1):
        if fig is None:
            fig = plt.figure()
        distquo = self.comp.get_center_distance_quotient()
        fig = plot_diagnostics(distquo,0,3,1,'Distance to mean quotient',fig=fig,totplots=totplots,plotnbr=plotnbr)
        return fig
    
    def cov_dist(self,norm='F',fig=None,totplots=1,plotnbr=1):
        distF = self.comp.get_cov_dist(norm)
        plot_diagnostics(np.log10(distF),-5,0,-3,'Frobenius distance',False,fig=fig,totplots=totplots,plotnbr=plotnbr)

class TracePlot(object):
    
    def __init__(self,traces):
        self.traces = traces
        self.traces.plot = self
        
    def all(self,fig = None):
        if fig == None:
            fig = plt.figure()
        for k in range(self.K):
            ax = plt.subplot2grid((1, self.K+1), (0, k))
            self.traceplot_mulat(k,ax)
            ax.set_title('theta_'+'{}'.format(k+1))
        ax = plt.subplot2grid((1, res.K+1), (0, k+1))
        self.traceplot_nu
        ax.set_title('nu',fontsize=16)
        return f,ax
        
    def mulat(self,k,ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        ax.plot(self.traces.ind, self.get_mulat_k(k))
        ax.set_xlim(0,self.traces.ind[-1])
        ax.set_ylim(-.2,1.2)
        ax.axes.yaxis.set_ticks([0.1,0.9])
        plt.axvspan(0, self.traces.burnind[-1], facecolor='0.5', alpha=0.5)
        
    def nu(self,ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        ax.plot(self.traces.ind, self.get_nu(k))
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
        
    def histnd(self,Nsamp=None,bins=50,fig = None):
        if fig == None:
            fig = plt.figure()
        histnd(self.fcsample.get_data(Nsamp),bins,[0, 100],[5,95],fig,self.marker_lab)
#        for ax in fig.axes:
#            ax.set_xlim((-.1,1.1))
#        if ax.get_ylim()[1] < 5:
#            ax.set_ylim(-.1,1.1)
        
class BMplot(object):
    
    def __init__(self,bmres,marker_lab = None):
        self.bmres = bmres
        self.hGMM = hGMM

        self.comp_colors,self.suco_colors,self.comp_ord,self.suco_ord = self.get_colors_and_order()
        
        self.clp_nm = ClustPlot(bmres.clust_nm,self.comp_colors,self.comp_ord)
        self.clp_m = ClustPlot(bmres.clust_m,self.suco_colors,self.suco_ord)
        self.cop = CompPlot(bmres.components,self.comp_colors,self.comp_ord,self.suco_ord)
        self.trp = TracePlot(bmres.traces)
        
        self.mcsp = []
        for mimic in bmres.mimics:
            self.mcsp.append(MimicPlot(mimic))
            
        if marker_lab is None:
            marker_lab = ['']*bmres.d
        self.set_marker_lab(marker_lab)
        
    def set_marker_lab(self,marker_lab):
        self.marker_lab = marker_lab
        self.clp_nm.set_marker_lab(marker_lab)
        self.clp_m.set_marker_lab(marker_lab)
        self.cop.se_marker_lab(marker_lab)
        for mc in self.mcsp:
            mc.set_marker_lab(marker_lab)

    def set_population_lab(self,pop_lab):
        self.pop_lab = pop_lab
        self.clp_nm.set_population_lab(pop_lab)
        self.clp_m.set_population_lab(pop_lab)

    def get_colors_and_order(self):
        '''
            Get order of components (first orderd by super component size, then by
            individual component size) and colors to use for representing components.
        '''
        for s,suco in enumerate(self.bmres.mergeind):
            sc_ord = np.argsort(-np.array(self.bmres.p[suco]))
            self.bmres.mergeind[s] = [suco[i] for i in sc_ord]
        prob_mer = [sum(self.bmres.p[scind]) for scind in self.bmres.mergeind]
        suco_ord = np.argsort(-np.array(prob_mer))
        mergeind_sort = [self.bmres.mergeind[i] for i in suco_ord]
        print "mergeind_sort = {}".format(mergeind_sort)
        comp_ord = [ind for suco in mergeind_sort for ind in suco]
        cm = plt.get_cmap('gist_rainbow')
        colors = [(0,0,0)]*len(comp_ord)
        for s,suco in enumerate(mergeind_sort):
            suco_col = cm(s/len(suco_ord))
            for i,k in enumerate(suco):
                colors[k] = black_ip(suco_col,i,len(suco))
        return colors,suco_col,comp_ord,suco_ord

    def pca_biplot(self,comp,ax=None,poplabsh=None,sampmarkers=None):
        if sampmarkers is None:
            sampmarkers = [(4,0,45),(3,0),(0,3),(4,2)]
        if poplabsh is None:
            poplabsh = [[0,0],[0,-.02],[0,0],[-.1,0],[.22,0],[.06,-.06]]
        pca_biplot(self.p,comp,ax,varcol=self.suco_colors,varlabels=self.pop_lab,
                   varlabsh=poplabsh,sampleid=self.bmres.meta_data.group_id,sampmarkers=sampmarkers)        

    def pca_screeplot(self,ax=None):
        pca_screeplot(self.p,ax)






    





    







