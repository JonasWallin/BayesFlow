import matplotlib.pyplot as plt
import BayesFlow.utils.load_and_save as ls
import BayesFlow.BMplot as bmp

from setup_util import get_dir_setup_HF_art,set_donorid

'''
    Define which results to load.
'''

dataset = 'HF'

expname = 'test'
run = 11
setupno = 0

datadir,expdir,_,__ = get_dir_setup_HF_art(dataset,expname,str(setupno))
loaddirres = expdir+'run'+str(run)+'/'
mergemeth = 'bhat_hier_dip'

'''
    Define which plots to make.
'''

toplot = ['conv','marg','diagn']
#toplot = ['cent','quan','prob']
#toplot = ['mix','dip']
#toplot = ['sampmix']
#toplot = ['cent','pca']
#toplot = ['overlap']
#toplot = ['scatter']

'''
    Load results.
'''

res = ls.load_HMres(loaddirres,mergemeth)    

#print "res.mergeind = {}".format(res.mergeind)
#print "res.components.p = {}".format(res.components.p)
#print "res comp suco p sum: {}".format([np.sum(res.components.p[:,suco]) for suco in res.mergeind])
#print "res clust_m p: {}".format(np.sum(res.clust_m.p,axis=0))
#print "res clust_m classif_freq sum: {}".format(np.sum(np.vstack(res.clust_m.classif_freq),axis=0))

'''
    Plotting
'''

### Convergence

if 'conv' in toplot:
    res.traces.plot.all()
    res.traces.plot.nu()
 
### Marginals

if 'marg' in toplot:
    print "Mimic samples = {}".format(res.mimics)
    print "res marker lab = {}".format(res.meta_data.marker_lab)
    print "Marker lab = {}".format(res.mimics['sample3'].realsamp.plot.marker_lab)
    res.mimics['sample3'].realsamp.plot.histnd()
    res.mimics['sample3'].synsamp.plot.histnd()
    res.mimics['sample6'].realsamp.plot.histnd()
    res.mimics['sample6'].synsamp.plot.histnd()
    res.mimics['pooled'].realsamp.plot.histnd()
    res.mimics['pooled'].synsamp.plot.histnd()


### 1D centers

if 'cent' in toplot:
    fig = plt.figure(figsize=(2,5))
    fig_nm = plt.figure(figsize=(2,8))
    res.components.plot.center(yscale=True,fig=fig)
    res.components.plot.center(suco=False,yscale=True,fig=fig_nm)

### Quantiles

if 'quan' in toplot:
    fig = plt.figure(figsize=(2,5))
    fig_nm = plt.figure(figsize=(2,8))
    res.clust_m.plot.box(fig=fig)
    res.clust_nm.plot.box(fig=fig_nm)

### Probabilities

if 'prob' in toplot:
    fig = plt.figure(figsize=(2,5))
    fig_nm = plt.figure(figsize=(2,8))
    res.clust_m.plot.prob(fig=fig)
    res.clust_nm.plot.prob(fig=fig_nm)


### Mixture components

if 'mix' in toplot:

    figmix = plt.figure(figsize=(7,7))
    plotdim = [[1,0],[3,2]]
    res.components.plot.latent_allsamp(plotdim,figmix)

    #res.components.plot.latent(plotdim[1],plim=[0.1,1],plotlab=True)
    #res.components.plot.allsamp(plotdim[0],ks=[7],plotlab=True)


if 'sampmix' in toplot:
    plotdim = [[1,0],[3,2]]
    for j in range(res.J):
        res.components.plot.latent_allsamp(plotdim,js=[j])

# ### PCA biplot

if 'pca' in toplot:
    set_donorid(res.meta_data)
    res.plot.pca_screeplot()
    res.plot.set_population_lab([str(k) for k in range(res.clust_m.K)])
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
    fig = res.clust_m.plot.chist_allsamp(min_clf=0.3,dd=0,ks=[0,1])
    fig.suptitle('Overlap in marker: '+res.meta_data.marker_lab[0])
    res.clust_m.plot.chist_allsamp(min_clf=0.3,dd=1,ks=[0,1])
    fig.suptitle('Overlap in marker: '+res.meta_data.marker_lab[1])
    res.clust_m.plot.chist_allsamp(min_clf=0.3,dd=2,ks=[0,1])
    fig.suptitle('Overlap in marker: '+res.meta_data.marker_lab[2])
    res.clust_m.plot.chist_allsamp(min_clf=0.3,dd=3,ks=[0,1])
    fig.suptitle('Overlap in marker: '+res.meta_data.marker_lab[3])

### Diagnostic plots

if 'diagn' in toplot:
    res.components.plot.center_distance_quotient()
    res.components.plot.cov_dist()

if 'scatter' in toplot:
    res.clust_m.plot.scatter([0,1],0)

plt.show()

