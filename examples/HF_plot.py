from __future__ import division
import os
import imp
import matplotlib
import matplotlib.pyplot as plt
import load_and_save as ls
import BayesFlow.BMplot as bmp


'''
    Define home directory
'''

if 0:
    homedir = '/Users/johnsson/'
else:
    homedir = '/home/johnsson/'


'''
    Define file with experimental setup and directory where results are saved
'''

expdir = homedir+'Forskning/Experiments/FlowCytometry/BHM/HF/informed/'

expname = 'test'
rond = 1
loaddirres = expdir+expname+'/' + 'rond'+str(rond)+'/'
setupfile = loaddirres+'exp_setup_HF.py'

'''
    Load experimental setup
'''
setup = imp.load_source('setup', setupfile)

prior = setup.Prior()
simpar = setup.SimulationParam(prior)
postpar = setup.PostProcParam()

'''
    Define load and save directories, create save directories and copy experiment setup
'''
savedirfig = loaddirres+'fig/'

for dr in [savedirfig]:
    if not os.path.exists(dr):
        os.makedirs(dr)

'''
	Define plot parameters
'''

toplot = ['cent','dip','mix']#['sampmix']#['conv','cent','mix','dip','diagn']#['conv','cent','mix']#['cent','mix']#['pca','dip']#['quan']#['marg']
latexplot = 0

'''
	Load results
'''

res = ls.load_HMres(loaddirres,postpar.mergemeth)    
res.merge(method=postpar.mergemeth,**postpar.mergekws)
res.plot = bmp.BMplot(res,res.meta_data.marker_lab) ## A hack. Should not be needed, but somehow some references are lost otherwise.
#res.plot.set_marker_lab(res.meta_data.marker_lab)
#print "res.plot.marker_lab = {}".format(res.plot.marker_lab)
#print "res.components.plot.marker_lab = {}".format(res.components.plot.marker_lab)

print "res.clust_m.plot.colors = {}".format(res.clust_m.plot.colors)

print "id(res.plot.cop) = {}".format(id(res.plot.cop))
print "id(res.components.plot) = {}".format(id(res.components.plot))

#print "res.mergeind = {}".format(res.mergeind)
#print "res.components.p = {}".format(res.components.p)
#print "res comp suco p sum: {}".format([np.sum(res.components.p[:,suco]) for suco in res.mergeind])
#print "res clust_m p: {}".format(np.sum(res.clust_m.p,axis=0))
#print "res clust_m classif_freq sum: {}".format(np.sum(np.vstack(res.clust_m.classif_freq),axis=0))

'''
	Plotting
'''

if latexplot:
	matplotlib.rc('text', usetex=True)
	matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
	matplotlib.rcParams['ps.useafm'] = True
	matplotlib.rcParams['pdf.use14corefonts'] = True
	matplotlib.rcParams['text.usetex'] = True

### Convergence

if 'conv' in toplot:
	res.traces.plot.all()
	res.traces.plot.nu()
 
### Marginals

if 'marg' in toplot:
	print "Mimic samples = {}".format(res.mimics)
	print "res marker lab = {}".format(res.meta_data.marker_lab)
	print "Marker lab = {}".format(res.mimics['sample1'].realsamp.plot.marker_lab)
	res.mimics['sample1'].realsamp.plot.histnd()
	res.mimics['sample1'].synsamp.plot.histnd()
 	res.mimics['sample2'].realsamp.plot.histnd()
	res.mimics['sample2'].synsamp.plot.histnd()
	res.mimics['pooled'].realsamp.plot.histnd()
	res.mimics['pooled'].synsamp.plot.histnd()


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

 	figmix = plt.figure(figsize=(7,7))
 	plotdim = [[1,0],[3,2]]
 	res.components.plot.latent_allsamp(plotdim,figmix)

	res.components.plot.latent(plotdim[1],plim=[0.1,1],plotlab=True)
	res.components.plot.allsamp(plotdim[0],ks=[7],plotlab=True)


if 'sampmix' in toplot:
    plotdim = [[1,0],[3,2]]
    for j in range(res.J):
        res.components.plot.latent_allsamp(plotdim,js=[j])

# ### PCA biplot

if 'pca' in toplot:
#     if latexplot:
#         lb = ' \n '
#         dim = r'$^\text{dim}$'
#     else:
#         lb = '\n'
#         dim = 'dim'
#     figpca = plt.figure(figsize=(5,5))
#     ax = figpca.add_subplot(111)

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

if not latexplot:
 	plt.show()


#res.clust_nm.get_bh_dist_data()

