from mpi4py import MPI
from BayesFlow.utils.seed import get_seed

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

'''
    Experiment parameters
'''
dataset = 'HF'
hGMMtiming = False
Nevent = 2000
testrun = True

expname = 'test'
setupno = '0'
seed = get_seed(expname)

'''
    Directories for data, experiment results and setup
'''

datadir = 'data/'+dataset
expparentdir = 'experiments/'
expdir = expparentdir+expname
setupdir = 'exp_setup/HF/'

'''
    Run MCMC sampler
'''
if 1:
    execfile('demo_HF_MCMC.py',globals())
else:
    run = 1
    
'''
    Post-processing: merging components and computing pdip
    NB! Computed on a single core

'''
if 1:
    execfile('demo_HF_merge.py',globals())

'''
    Plot
'''    
'''
    Define which plots to make.
'''
toplot = ['conv','marg','diagn']
#toplot = ['cent','quan','prob']
#toplot = ['compfit']
#toplot = ['mix','dip']
#toplot = ['sampmix']
#toplot = ['cent','pca']
#toplot = ['overlap']
#toplot = ['scatter']

if 1 and rank == 0:
    execfile('demo_HF_plot.py',globals())