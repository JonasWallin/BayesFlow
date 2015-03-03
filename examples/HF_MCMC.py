from __future__ import division
from mpi4py import MPI
import os
import inspect
import numpy as np
import imp
import BayesFlow as bf
import BayesFlow.data
import load_and_save as ls
import timer

timer = timer.Timer()


'''
    Define file with experimental setup and directory where results are saved
'''
setupfile = 'exp_setup_HF.py'

expdir = '../results/HF/'

'''
    Load experimental setup
'''
setup = imp.load_source('setup', setupfile)

prior = setup.Prior()
simpar = setup.SimulationParam(prior)
postpar = setup.PostProcParam()

'''
    Set seed
'''
np.random.seed(simpar.seed)

'''
    Define save and load directories, create save directories and copy experiment setup
'''

savedir = expdir+simpar.expname+'/' + 'rond'+str(simpar.rond)+'/'
if simpar.loadinit:
    loaddir = expdir+simpar.expname+'/' + 'rond'+str(simpar.loadrond)+'/hGMM/burn/'

if MPI.COMM_WORLD.Get_rank() == 0:
    for dr in [savedir,savedir+'hGMM/burn/',savedir+'hGMM/prod/']:
        if not os.path.exists(dr):
            os.makedirs(dr)
    currfile = inspect.getfile(inspect.currentframe())
    os.system("cp "+setupfile+" "+savedir)
    os.system("cp "+currfile+" "+savedir)

'''
    Load data
'''
if MPI.COMM_WORLD.Get_rank() == 0:
    data,metadata = bf.data.healthyFlowData.load(scale=True)
    sampnames = metadata['samp']['names']
    print "sampnames = {}".format(sampnames)
else:
    data = None
    sampnames = None
    metadata = None
    
'''
    Initialize hGMM
'''
hGMM = bf.hierarical_mixture_mpi(data = data, sampnames = sampnames, prior = prior)

timer.timepoint('initialization')

'''
    MCMC sampling
'''

'''
        Burn in iterations
'''

blog1 = hGMM.simulate(simpar.phaseB1,'Burnin phase 1')
hGMM.deactivate_outlying_components()
blog2 = hGMM.simulate(simpar.phaseB2, 'Burnin phase 2')
burnlog = blog1.cat(blog2)

if MPI.COMM_WORLD.Get_rank() == 0:
    ls.save_object(burnlog,savedir)
hGMM.save_to_file(savedir+'hGMM/burn/')  

timer.timepoint('burnin iterations ({}) and postproc'.format(simpar.nbriter*simpar.qburn))

'''
        Production iterations
'''
prodlog = hGMM.simulate(simpar.phaseP, 'Production phase')

if MPI.COMM_WORLD.Get_rank() == 0:
    ls.save_object(prodlog,savedir)
hGMM.save_to_file(savedir+'hGMM/prod/') 

timer.timepoint('production iterations ({}) and postproc'.format(simpar.nbriter*simpar.qprod))


'''
    Post-processing: merging components and computing pdip
    NB! Computed on a single core
'''

if MPI.COMM_WORLD.Get_rank() == 0 and postpar.postproc:
    bmres = bf.BMres(prodlog,burnlog,data,metadata)
    bmres.merge(method=postpar.mergemeth,**postpar.mergekws)
    timer.timepoint("merging")
    
    bmres.clust_nm.get_pdip()
    bmres.clust_m.get_pdip()
    timer.timepoint("computing dip test")

    ls.save_object(bmres,savedir)

if MPI.COMM_WORLD.Get_rank() == 0:
    timer.print_timepoints()

