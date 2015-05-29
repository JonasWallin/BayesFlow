import matplotlib.pyplot as plt
import BayesFlow.utils.load_and_save as ls
from BayesFlow import BMplot

from example_util import set_donorid,load_setup_postproc_HF

if expdir[-1] != '/':
    expdir += '/'
loaddirres = expdir+'run'+str(run)+'/'

'''
    Load results.
'''

try:
    res = hmres
    res.plot = BMplot(res)
except:
    _,setup_postproc = load_setup_postproc_HF(loaddirres,setupno)
    postpar = setup_postproc()  
    res = ls.load_HMres(loaddirres,postpar.mergemeth)  


#print "res.mergeind = {}".format(res.mergeind)
#print "res.components.p = {}".format(res.components.p)
#print "res comp suco p sum: {}".format([np.sum(res.components.p[:,suco]) for suco in res.mergeind])
#print "res clust_m p: {}".format(np.sum(res.clust_m.p,axis=0))
#print "res clust_m classif_freq sum: {}".format(np.sum(np.vstack(res.clust_m.classif_freq),axis=0))

''''
    Plotting
'''

plotdim = [[0,1],[3,2]]
mimicnames = res.mimics.keys()
print "mimicnames = {}".format(mimicnames)

### Convergence

if 'conv' in toplot:
    res.traces.plot.all()
    res.traces.plot.nu()
 
### Marginals

if 'marg' in toplot:
    print "Mimic samples = {}".format(res.mimics)
    print "res marker lab = {}".format(res.meta_data.marker_lab)
    print "Marker lab = {}".format(res.mimics[mimicnames[0]].realsamp.plot.marker_lab)
    maxmimics = 4
    i = 0
    for name in mimicnames:
        res.mimics[name].realsamp.plot.histnd()
        res.mimics[name].synsamp.plot.histnd()
        i += 1
        if i > maxmimics:
            break

### Component fit

if 'compfit' in toplot:
    res.plot.component_fit(plotdim,name=mimicnames[0])
    res.plot.component_fit(plotdim,name='pooled')

### 1D centers

if 'cent' in toplot:

    res.components.plot.center(yscale=True)
    res.components.plot.center(suco=False,yscale=True)

### Quantiles

if 'quan' in toplot:
    res.clust_m.plot.box()
    res.clust_nm.plot.box()

### Probabilities

if 'prob' in toplot:
    res.clust_m.plot.prob()
    res.clust_nm.plot.prob()


### Mixture components

if 'mix' in toplot:
    figmix = plt.figure(figsize=(7,2*len(plotdim)))
    res.components.plot.latent_allsamp(plotdim,figmix)#,ks=range(8))
    for ax in figmix.axes:
        ax.set_xlim([-0.2,1.2])
        ax.set_ylim([-0.2,1.2])
        
    figmix2 = plt.figure(figsize=(7,2*len(plotdim)))
    res.components.plot.latent_allsamp(plotdim,figmix2)#,ks=range(8))
    for ax in figmix2.axes:
        ax.set_xlim([-2,2])
        ax.set_ylim([-2,2])
    #res.components.plot.latent(plotdim[1],plim=[0.1,1],plotlab=True)
    #res.components.plot.allsamp(plotdim[0],ks=[7],plotlab=True)


if 'sampmix' in toplot:
    for j in range(res.J):
        figsampmix = plt.figure(figsize=(7,2*len(plotdim)))
        res.components.plot.latent_allsamp(plotdim,js=[j],fig=figsampmix)
        figsampmix.suptitle(res.meta_data.samp['names'][j])
        for ax in figsampmix.axes:
            ax.set_xlim([-0.2,1.2])
            ax.set_ylim([-0.2,1.2])

# ### PCA biplot

if 'pca' in toplot:

     res.plot.pca_screeplot()
     res.plot.set_population_lab(['CD4 T cells','CD8 T cells','B cells','','',''])
     res.plot.pca_biplot([0,1])
     res.plot.pca_biplot([2,3])


### Dip test

if 'dip' in toplot:

    res.clust_m.plot.pdip()
    res.clust_m.plot.qhist_dipcrit(q=.25)
    res.clust_nm.plot.pdip()
    res.clust_nm.plot.qhist_dipcrit(q=.25)


### Histograms showing overlap between clusters

if 'overlap' in toplot:
    res.clust_m.plot.chist_allsamp(min_clf=0.3,dd=0,ks=[0,1])
    res.clust_m.plot.chist_allsamp(min_clf=0.3,dd=1,ks=[0,1])
    res.clust_m.plot.chist_allsamp(min_clf=0.3,dd=2,ks=[0,1])
    res.clust_m.plot.chist_allsamp(min_clf=0.3,dd=3,ks=[0,1])

### Diagnostic plots

if 'diagn' in toplot:
    res.components.plot.center_distance_quotient()
    res.components.plot.cov_dist()

if 'scatter' in toplot:
    res.clust_m.plot.scatter([0,1],0)

# ###

print "id(res.plot.cop) = {}".format(id(res.plot.cop))
print "id(res.components.plot) = {}".format(id(res.components.plot))

plt.show()
