from __future__ import division
from mpi4py import MPI
import sys

import BayesFlow as bf
from BayesFlow.utils import Timer
import BayesFlow.utils.load_and_save as ls

from setup_util import fcdata,read_argv_HF_art,get_dir_setup_HF_art,get_J

'''
    Experiment parameters
'''
dataset = 'HF'
hGMMtiming = False
Nevent = 2000
testrun = False

'''
    Initialization
'''
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

timer = Timer()

expname,setupno,seed = read_argv_HF_art(sys.argv)
datadir,expdir,setupfile,setup = get_dir_setup_HF_art(dataset,expname,setupno)
savedir,run = bf.setup_sim(expdir,seed,setupfile) # copies experiment setup and set seed

data,metadata = fcdata(dataset,Nevent=Nevent,scale='percentilescale',datadir=datadir)

prior,simpar,postpar = setup(get_J(data),Nevent,testrun)

if rank == 0:
    print "n_theta set to {}, n_Psi set to {}".format(prior.n_theta[0],prior.n_Psi[0])
    print "Q set to {}, H set to {}".format(prior.Q[0],prior.H[0])

hGMM = bf.hierarical_mixture_mpi(data = data, sampnames = metadata['samp']['names'],
                                 prior = prior)
for j,GMM in enumerate(hGMM.GMMs):
    print "rank {} sample {} has name {}".format(rank, j, GMM.name)

timer.timepoint('initialization')
timer.print_timepoints()
    
'''
    MCMC sampling
'''

'''
        Burn in iterations
'''

if hGMMtiming:
    hGMM.toggle_timing(on=True)

print "simpar.phases['B'] = {}".format(simpar.phases['B'])  
burnlog = hGMM.simulate(simpar.phases['B'],'Burnin phase')

ls.save_object(burnlog,savedir)
hGMM.save_to_file(savedir+'hGMM/burn/')  

timer.timepoint('burnin iterations ({}) and postproc'.format(simpar.nbriter*simpar.qburn))

'''
        Production iterations
'''
prodlog = hGMM.simulate(simpar.phases['P'], 'Production phase',stop_if_cl_off=False)
ls.save_object(prodlog,savedir)
hGMM.save_to_file(savedir+'hGMM/prod/') 

timer.timepoint('production iterations ({}) and postproc'.format(simpar.nbriter*simpar.qprod))
timer.print_timepoints()

'''
    Post-processing: merging components and computing pdip
    NB! Computed on a single core
'''
if postpar.postproc:

    hmres = bf.HMres(prodlog,burnlog,data,metadata)
    hmres.merge(method=postpar.mergemeth,**postpar.mergekws)
    timer.timepoint("merging")
 
    if rank == 0:
        hmres.clust_nm.get_pdip()
        hmres.clust_m.get_pdip()
    timer.timepoint("computing dip test")

    ls.save_object(hmres,savedir)

    timer.print_timepoints()

