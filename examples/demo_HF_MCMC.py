# -*- coding: utf-8 -*-
"""
Created on Fri May  8 16:04:29 2015

@author: johnsson
"""
from __future__ import division

import BayesFlow as bf
from BayesFlow.utils import Timer
import BayesFlow.utils.load_and_save as ls

from example_util import  retrieve_healthyFlowData,load_setup_HF,HF,get_J


'''
    Initialization
'''

timer = Timer()

retrieve_healthyFlowData(datadir)

timer.timepoint('retrieve data')

setupfile,setup = load_setup_HF(setupdir,setupno)
savedir,run = bf.setup_sim(expdir,seed,setupfile) # copies experiment setup and set seed

data,metadata = HF(datadir,Nevent=Nevent,scale='percentilescale')

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
