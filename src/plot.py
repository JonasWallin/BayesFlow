'''
Created on Aug 10, 2014

@author: jonaswallin
'''
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
from .utils.plot_util import black_ip

def autocorr(x_in, lag=100):
    """
        returning the autocorrelation of x_in (1D)
        Lazy implimentation
    """
    x = np.zeros_like(x_in)
    x[:] = x_in[:]
    x -= np.mean(x)
    n = min(len(x),lag)
    res = np.zeros(n)
    for t in range(n):
        res[t] = np.corrcoef(np.array([x[0:len(x)-t], x[t:len(x)]]))[0,1]
    return res

def hist2d(dat,dim,bins,quan=[0.5,99.5],quan_plot = [5, 95],ax=None,lims=None,labels=None):

    if ax is None:
        ax = plt.figure().add_subplot(111)

    i,j = dim

    if lims is None:
        index_i = (dat[:,i] > np.percentile(dat[:,i], quan[0])) * (dat[:,i] < np.percentile(dat[:,i], quan[1]))
        index_j = (dat[:,j] > np.percentile(dat[:,j], quan[0]) ) * (dat[:,j] < np.percentile(dat[:,j], quan[1]))
        q1_y = np.percentile(dat[index_j*index_i ,j], quan_plot[0])
        q2_y = np.percentile(dat[index_j*index_i ,j], quan_plot[1])
        q1_x = np.percentile(dat[index_j*index_i ,i], quan_plot[0])
        q2_x = np.percentile(dat[index_j*index_i ,i], quan_plot[1])
    else:
        try:
            q1_x =lims[i, 0]
            q2_x =lims[i, 1]
            q1_y =lims[j, 0]
            q2_y =lims[j, 1]
        except TypeError:
            q1_x,q2_x = lims
            q1_y,q2_y = lims
        index_i = (dat[:,i] > q1_x) * (dat[:,i] < q2_x)
        index_j = (dat[:,j] > q1_y) * (dat[:,j] < q2_y)        


    dat_j  = dat[index_i*index_j ,j]
    #dat_j[dat_j == 0] = 1.
    dat_i  = dat[index_i*index_j ,i]
    #dat_i[dat_i == 0] = 1.
    
    ax.hist2d(dat_i, dat_j, bins = bins, norm=colors.LogNorm(),vmin=1)
    ax.patch.set_facecolor('white')
    ax.axes.xaxis.set_ticks(np.linspace(q1_x, q2_x,num=4))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax.axes.yaxis.set_ticks(np.linspace(q1_y, q2_y,num=4))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax.tick_params(axis='both', which='major', labelsize=8)
    if not labels is None:
        ax.set_xlabel(labels[i])
        ax.set_ylabel(labels[j])


def histnd(dat, bins, quan = [0.5,99.5], quan_plot = [5, 95], f = None, 
           lims = None, labels = None):
    
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
        if lims is None:
            ax.set_xlim(xlims[0],xlims[1])
        else:
            try:
                ax.set_xlim(lims[i, 0],lims[i, 1])
            except TypeError:
                ax.set_xlim(lims[0],lims[1])
        if not labels is None:
            ax.set_xlabel(labels[i])                
        for j in range(i+1,nv):
            index_j = (dat[:,j] > np.percentile(dat[:,j], quan[0]) ) * (dat[:,j] < np.percentile(dat[:,j], quan[1]))
            if lims is None:
                q1_y = np.percentile(dat[index_j*index_i ,j], quan_plot[0])
                q2_y = np.percentile(dat[index_j*index_i ,j], quan_plot[1])
                q1_x = np.percentile(dat[index_j*index_i ,i], quan_plot[0])
                q2_x = np.percentile(dat[index_j*index_i ,i], quan_plot[1])
            else:
                try:
                    q1_x =lims[i, 0]
                    q2_x =lims[i, 1]
                    q1_y =lims[j, 0]
                    q2_y =lims[j, 1]
                except TypeError:
                    q1_x,q2_x = lims
                    q1_y,q2_y = lims
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

def plot_GMM_scatter_all(hGMM, dim):
    
    for GMM in hGMM.GMMs:
        plt.figure()
        ax = plt.subplot(111)
        plot_GMM_scatter(GMM, ax ,dim)

def plot_GMM_scatter(GMM, ax ,dim):
    """
        Plots the scatter plot of the data over dim
        and assigning each class a different color
    """
    data= GMM.data[:,dim]
    x = GMM.x
    cm = plt.get_cmap('gist_rainbow')
    ax.set_color_cycle([cm(1.*i/GMM.K) for i in range(GMM.K)])
    if len(dim) == 2:
        for k in range(GMM.K):
            plt.plot(data[x==k,0],data[x==k,1],'+',label='k = %d'%(k+1))
            
def drawbox(quantiles,boxloc,boxw,ms,ax):
    '''
        Draw box in boxplot
        
        quantiles   -   quantiles to use for box
        boxloc      -   x-values where to draw box
        boxw        -   width of box
        ms          -   size of whiskers
    '''
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    low = quantiles[0]
    blow = quantiles[1]
    bmid = quantiles[2]
    bupp = quantiles[3]
    upp = quantiles[4]
    ax.plot([boxloc,boxloc],[low,upp],marker = '_',color='blue',ms=ms)
    ax.plot([boxloc-boxw/2,boxloc+boxw/2,boxloc+boxw/2,boxloc-boxw/2,boxloc-boxw/2],[blow,blow,bupp,bupp,blow],color='blue')
    ax.plot([boxloc-boxw/2,boxloc+boxw/2],[bmid,bmid],color='blue')
    
def drawboxes(quantiles,ax=None,std_ylim=True):

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    d = len(quantiles[0])
    boxloc = (np.array(range(d)) + .5)/(d+1)
    boxw = (boxloc[1] - boxloc[0])/3.5
    ms = 10
    for dd in range(d):
        quan_dd = [quan[dd] for quan in quantiles]
        drawbox(quan_dd,boxloc[dd],boxw,ms,ax)
    ax.axes.xaxis.set_ticks(boxloc)
    if std_ylim:
        xlim = ax.get_xlim()
        ax.plot([xlim[0],xlim[1]],[.5, .5],color='grey')
        ax.set_ylim(-.1,1.1)
        ax.axes.yaxis.set_ticks([.2,.8])

    return ax

def probscatter(p,ax):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    J = len(p)
    ax.scatter(range(J), p)
    ax.set_yscale('log')
    ax.set_ylim(1e-3,1)
    ax.axes.yaxis.set_ticks([1e-2,1e-1])
    xlim = ax.get_xlim()
    ax.plot([xlim[0],xlim[1]],[1e-2,1e-2],color='grey')
    ax.plot([xlim[0],xlim[1]],[1e-1,1e-1],color='grey')
    ax.axes.xaxis.set_ticks([])
    ax.set_xlim(-1,J)


def visualEigen(Sigma, mu, dim):
    if np.isnan(Sigma[0,0]) or np.isinf(Sigma[0,0]):
        return None
    Sigma = Sigma[np.ix_(dim,dim)]
    try:
        E, V = np.linalg.eig(Sigma)
    except np.linalg.LinAlgError:
        print "Exception caught, Sigma = {}".format(Sigma)
        return None
    t = np.linspace(0,2*np.pi,100)
    e = np.array([np.cos(t), np.sin(t)])
    V2 = np.sqrt(E)*V
    VV = np.dot(e.transpose(),V2.transpose()) + mu[dim].transpose()
    return VV
    
def component_plot(mus,Sigmas,dim,ax,colors=None,lw=2):
    '''
        Viusalize mixture component
        
        mu,Sigma    -   mixture component parameters
        dim         -   dimensions for projection
    '''
    
    q_y = [np.inf,-np.inf]
    q_x = [np.inf,-np.inf]
    K = len(mus)
    if colors is None:
        colors = ['black']*K
    for k in range(K):
        if not np.isnan(mus[k][0]):
            plres = visualEigen(Sigmas[k], mus[k], [dim[0],dim[1]])
            if plres is None:
                continue
            q_res = np.percentile(plres[:,0], 50)
            q_x[0] = min(q_x[0],q_res)
            q_x[1] = max(q_x[1],q_res)
            q_res = np.percentile(plres[:,1], 50)
            q_y[0] = min(q_y[0],q_res)
            q_y[1] = max(q_y[1],q_res)
            ax.plot(plres[:,0],plres[:,1],'-',color = colors[k],linewidth=lw)    
    return {'q_x': q_x, 'q_y': q_y}
    
def pers_component_plot(muspers,Sigmaspers,dim,ax,colors=None,lw=1):
    '''
        Visualize mixture components for all samples
    '''
    q_y = [np.inf,-np.inf]
    q_x = [np.inf,-np.inf]
    q = {'q_x': q_x, 'q_y': q_y}
    for j in range(len(muspers)):
        qnew = component_plot(muspers[j],Sigmaspers[j],dim,ax,colors,lw)
        q = mergeQ(q,qnew)
    return q

def set_component_plot_tics(axlist,q):
    q_x = q['q_x']
    q_y = q['q_y']
    for ax in axlist:
        ax.axes.yaxis.set_ticks(np.linspace(q_y[0], q_y[1],num=3))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        ax.axes.xaxis.set_ticks(np.linspace(q_x[0], q_x[1],num=3))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

def mergeQ(q1,q2):
    q_x = [0, 0]
    q_y = [0, 0]
    q_x[0] = min(q1['q_x'][0],q2['q_x'][0])
    q_x[1] = max(q1['q_x'][1],q2['q_x'][1])
    q_y[0] = min(q1['q_y'][0],q2['q_y'][0])
    q_y[1] = max(q1['q_y'][1],q2['q_y'][1])
    return {'q_x': q_x, 'q_y': q_y}

def plot_diagnostics(diagn,ymin,ymax,ybar = None,order=None,name = '',log=False,fig=None,totplots=1,plotnbr=1):
    if fig is None:
        fig = plt.figure()
    J = diagn.shape[0]
    K = diagn.shape[1]
    if order is None:
        order = range(K)
    nbr_cols = 2*totplots - 1
    col_start = 2*(plotnbr - 1) 
    xloc = np.arange(J) + .5
    bar_width = .35
    for k in range(K):
        ax = fig.add_subplot(K,nbr_cols,k*nbr_cols + col_start + 1)
        if k == 0:
            plt.title(name)
        ax.bar(xloc,diagn[:,order[k]],bar_width,log=log)
        ax.set_ylim(ymin,ymax)
        if not ybar is None:
            ax.plot([xloc[0],xloc[-1]],[ybar,ybar])
        ax.axes.yaxis.set_ticks([])
        ax.axes.xaxis.set_ticks([])
    return fig

def plot_pbars(data,ymin,ymax,order=None,colors=None,log=False,ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    K = len(data)
    if order is None:
        order = range(K)
    if colors is None:
        colors = (1,0,0)*K
    xloc = np.arange(K)+.5
    bar_width = 0.35
    for k,xl in enumerate(xloc):
        ax.bar(xl,data[order[k]],color=colors[order[k]],log=log)
    ax.set_ylim(ymin,ymax)
    ax.axes.yaxis.set_ticks([])
    ax.axes.xaxis.set_ticks([])
    return ax
    
def pca_biplot(data,comp,ax=None,varcol=None,varlabels=None,varlabsh=None,sampleid=None,sampmarkers=None):
    '''
        PCA biplot
        
        data        -   data which to use
        comp        -   which two principal components to display
        ax          -   where to plot
        varcol      -   colors for variables
        varlabels   -   labels for variables
        varlabsh    -   shift for labels for variables
        sampleid    -   which group does each sample belong to?
        sampmarkers -   which plotmarkers shoud be used for the samples
    '''
    
    if ax == None:
    	f = plt.figure()
    	ax = f.add_subplot(111)
    else:
    	f = None
     
    K = data.shape[1]
    print "K = {}".format(K)
    if varcol is None:
        varcol = [(0,0,0)]*K
    if varlabels is None:
        varlabels = ['']*K
    if varlabsh is None:
        varlabsh = [0,0]*K
    if sampleid is None:
        if sampmarkers is None:
            sampmarkers = [(4,2)]
        sampleid = [0]*data.shape[0]
    else:
        sampleids = list(set(sampleid))
        iddict = dict(zip(sampleids,range(len(sampleids))))
        sampleid_new = [iddict[sid] for sid in sampleid]
        sampleid = sampleid_new
        if sampmarkers is None:
            sampmarkers = [(k,2) for k in range(3,(len(set(sampleids)))+3)]
        
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