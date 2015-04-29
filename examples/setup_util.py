from mpi4py import MPI
import socket
import imp
import os
import numpy as np

from BayesFlow.utils.seed import get_seed
from BayesFlow.utils.dat_util import load_fcdata_MPI as load_fcdata

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

homedirs = {'Ke':'/Users/johnsson/','ta':'/home/johnsson/'}
datadirs = {'HF': {'Ke': '/Users/johnsson/Forskning/Data/healthyFlowData/'}}
exppaths = {'HF': 'HF/demo/'}

def get_dir_setup_HF_art(dataset,expname,setupno):
	host = socket.gethostname()
	homedir = homedirs[host[:2]]
	datadir = datadirs[dataset][host[:2]]
	exppath = exppaths[dataset]
	setupfile = './exp_setup/'+dataset+'/'+setupno+'.py'
	#print "os.listdir('../'+dataset+'/exp_setup/'): {}".format(os.listdir('./'+dataset+'/exp_setup/')) 
	#print "setupfile {} exists {}".format(setupfile,os.path.exists(setupfile))
	expparentdir = homedir+'Forskning/Experiments/FlowCytometry/BHM/'+exppath
	expdir = expparentdir+expname+'/'
	return datadir,expdir,setupfile,imp.load_source('', setupfile).setup

def get_J(data):
	try:
		return len(data)
	except:
		return 0

def read_argv_HF_art(args):
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
    ext = '.dat'
    loadfilef = lambda filename: np.loadtxt(filename)
    startrow = 0
    startcol = 0
    marker_lab = ['CD4','CD8','CD3','CD19']
    return load_fcdata(ext,loadfilef,startrow,startcol,marker_lab,datadir,**kw)

loadfunnames = {'HF': HF}

def fcdata(dataset,**kw):
    return loadfunnames[dataset](**kw)

def set_donorid(metadata):
	sampnames = metadata.samp['names']
	#print "sampnames = {}".format(sampnames)
	sampnbr = [int(name.replace('sample','')) for name in sampnames]
	#print "sampnbr = {}".format(sampnbr)
	donorid = [(nbr-1)/5 for nbr in sampnbr]
	#print "donorid = {}".format(donorid)
	metadata.samp['donorid'] = donorid