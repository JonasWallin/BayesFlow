from mpi4py import MPI
import imp
import os
import numpy as np

from BayesFlow.utils.seed import get_seed
from BayesFlow.utils.dat_util import load_fcdata_MPI as load_fcdata
import BayesFlow.data.healthyFlowData as hf

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def retrieve_healthyFlowData(datadir):
    if rank == 0:
        if datadir[-1] != '/':
            datadir += '/'
        if not os.path.exists(datadir):
            os.makedirs(datadir)
        if len(os.listdir(datadir)) > 0:
            print "Directory {} not empty, no data retrieved".format(datadir)
            return
        data,metadata = hf.load(scale=False)
        for j,dat in enumerate(data):
            np.savetxt(datadir+metadata['samp']['names'][j]+'.dat',data[j])

def load_setup_HF(setupdir,setupno):
    if setupdir[-1] != '/':
        setupdir += '/'
    setupfile = setupdir+str(setupno)+'.py'
    return setupfile,imp.load_source('', setupfile).setup
    
def load_setup_postproc_HF(setupdir,setupno):
    if setupdir[-1] != '/':
        setupdir += '/'
    setupfile = setupdir+str(setupno)+'.py'
    return setupfile,imp.load_source('', setupfile).setup_postproc

def get_J(data):
    try:
        return len(data)
    except:
        return 0

def read_argv(args):
    '''
        Read input arguments for informed setup
    '''
    expname = 'test'
    setupno = '0'
    seed = get_seed(expname)

    try:
        expname = args[1]
        setupno = args[2]
        seed = int(args[3])
    except IndexError:
        pass

    print "Experiment {} run with setup {} at rank {}".format(expname,setupno,rank)
    return expname,setupno,seed

'''
    Data load functions
'''
        
def HF(datadir,**kw):
    if datadir[-1] != '/':
        datadir += '/'
    ext = '.dat'
    loadfilef = lambda filename: np.loadtxt(filename)
    startrow = 0
    startcol = 0
    marker_lab = ['CD4','CD8','CD3','CD19']
    return load_fcdata(ext,loadfilef,startrow,startcol,marker_lab,datadir,**kw)

def set_donorid(metadata):
    sampnames = metadata.samp['names']
    #print "sampnames = {}".format(sampnames)
    sampnbr = [int(name.replace('sample','')) for name in sampnames]
    #print "sampnbr = {}".format(sampnbr)
    donorid = [(nbr-1)/5 for nbr in sampnbr]
    #print "donorid = {}".format(donorid)
    metadata.samp['donorid'] = donorid