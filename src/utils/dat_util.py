# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 16:32:31 2015

@author: johnsson
"""
from __future__ import division
import numpy as np
import numpy.random as npr
import glob
from mpi4py import MPI
import load_and_save as ls

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def load_fcdata_MPI(ext,loadfilef,startrow,startcol,marker_lab,datadir,verbose=True,**kw):

    data = None
    metasamp = {'names': None}
    metadata = {'samp':metasamp,'marker_lab': None}  

    if rank == 0:
        data,metadata = load_fcdata(ext,loadfilef,startrow,startcol,marker_lab,datadir,**kw)

    if rank == 0 and verbose:
        print "sampnames = {}".format(metadata['samp']['names'])
        print "data sizes = {}".format([dat.shape for dat in data])

    return data,metadata


def load_fcdata(ext,loadfilef,startrow,startcol,marker_lab,datadir,
                Nsamp=None,Nevent=None,scale='percentilescale',
                rm_extreme=True,perturb_extreme=False,eventind=[],
                eventind_dic = {},datanames=None,load_eventind=True,
                i_eventind_load = 0):
    """
        Load flow cytometry data. Will default arguments the indices of the subsampled data will 
        be saved in a file "eventind_'Nevent'_0.json" in datadir and when load_fcdata is called
        subsequently, the eventindices will be loaded from this file.

        ext             -   file extension for fc data files.
        loadfilef       -   function for loading one flow cytometry data file into a np.array.
        startrow        -   first row that could be used.
        startcol        -   first column that should be used.
        marker_lab      -   names of markers. Should equal the number of columns used.
        datadir         -   directory where fc data files have been saved.
        Nsamp           -   number of flow cytometry samples to use. If None, all files matching
                            the file extension in datadir will be used.
        Nevent          -   number of events that should be sampled from the fc data file.
        scale           -   'maxminscale','percentilescale' or None. If 'maxminscale' each sample is scaled so that
                            the minimal value in each dimension go to 0 and the maximal value in each
                            dimension go to 1. If 'percentilescale' each sample is scaled so that the 1st percentile of
                            the pooled data goes to 0 and the 99th percentile of the pooled data go to 1.
                            If False, no scaling is done.
        rm_extreme      -   boolean. Should extreme values be removed? (If not removed or perturbed they
                            might lead to singular covariance matrices.)
        perturb_extreme -   boolean. Should extreme values be perturbed? (If not removed or perturbed they
                            might lead to singular covariance matrices.)
        eventind        -   list. The ith element contains a list of the indices that will be loaded
                            for the ith sample.
        eventind_dic    -   dictionary. eventind_dic[sampname] contains the indices that will be loaded for 
                            the sample with filename sampname.
        datanames       -   list of fc data file names that will be loaded.
        load_eventind   -   bool. Should eventind be loaded from data directory? If eventind.json does
                            not exist in data directory it will be saved to it.
        i_eventind_load -   index of eventindfile to load.
    """

    if datanames is None:
        datafiles = glob.glob(datadir+'*'+ext) 
        #print "datafiles = {}".format(datafiles)
    else:
        datafiles = [datadir + name + ext for name in datanames]
    J = len(datafiles)
    if not Nsamp is None:
        J = min(J,Nsamp)
        
    data = []
    sampnames = []
    print "J = {} samples will be loaded".format(J)
    for j in range(J):
        sampnames.append(datafiles[j].replace(datadir,'').replace(' ','').replace(ext,''))
        data.append(loadfilef(datafiles[j])[startrow:,startcol:])

    metasamp = {'names':sampnames}
    metadata = {'samp':metasamp,'marker_lab':marker_lab}          

    compute_new_eventind = True
    if load_eventind:
        eventind_dic = ls.load_eventind(datadir,Nevent,i_eventind_load)
        if len(eventind_dic) > 0:
            compute_new_eventind = False
    elif len(eventind_dic) > 0:
        for j,name in enumerate(sampnames):
            data[j] = data[j][eventind_dic[name],:]
        compute_new_eventind = False
    elif len(eventind) > 0:
        for j,dat in enumerate(data):
            data[j] = dat[eventind[j],:] 
        compute_new_eventind = False

    if perturb_extreme or (rm_extreme and compute_new_eventind):
        ok_inds = []
        for j,dat in enumerate(data):
            ok = np.ones(dat.shape[0],dtype='bool')
            for dd in range(dat.shape[1]):
                ok *= dat[:,dd] > 1.001*np.min(dat[:,dd]) 
                ok *= dat[:,dd] < 0.999*np.max(dat[:,dd])
            if perturb_extreme:
                data[j][~ok,:] = data[j][~ok,:] + npr.normal(0,0.01,data[j][~ok,:].shape)
                ok_inds.append(np.arange(data[j].shape[0]))
            else:
                ok_inds.append(np.nonzero(ok)[0]) 

    if not Nevent is None:
        for j,dat in enumerate(data):
            if compute_new_eventind:
                indices_j = npr.choice(ok_inds[j],Nevent,replace=False)
                eventind.append(indices_j)
                eventind_dic[sampnames[j]] = indices_j
            else:
                try:
                    indices_j = eventind_dic[sampnames[j]]
                except:
                    indices_j = eventind[j]
            data[j] = dat[indices_j,:]

    if compute_new_eventind:
        ls.save_eventind(eventind_dic,datadir,Nevent)  

    if scale == 'maxminscale':
        maxminscale(data)
    elif scale == 'percentilescale':
        percentilescale(data)
    elif not scale is None:
        raise ValueError, "Scaling {} is unsupported".format(scale)

    return data,metadata
    
def percentilescale(data, q = (1.,99.)):
    '''
        Scales the data sets in data so that given quantiles of the pooled data ends up at 0 and 1 respectively.

        data	-	list of data sets
	       q	-	percentiles. q[0] is the percentile will be scaled to 0, 
                    q[1] is the percentile that will be scaled to 1 (in the pooled data).
    '''
    alldata = np.vstack(data)
    datq = np.percentile(alldata,q,0)
    for j in range(len(data)):
        for m in range(data[0].shape[1]):
            data[j][:,m] = (data[j][:,m]-datq[0][m])/(datq[1][m]-datq[0][m])
    return datq

def maxminscale(data):
    """
        Each data set is scaled so that the minimal value in each dimension go to 0 
        and the maximal value in each dimension go to 1.

        data - list of data sets
    """
    d = data[0].shape[1]
    for j in range(len(data)):
        for m in range(d):
            data[j][:,m] = (data[j][:,m]-np.min(data[j][:,m]))/(np.max(data[j][:,m])-np.min(data[j][:,m]))        
