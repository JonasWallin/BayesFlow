# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 16:32:31 2015

@author: johnsson
"""
from __future__ import division
import numpy as np
import numpy.random as npr
import copy
import glob
from mpi4py import MPI
import load_and_save as ls
import cPickle as pickle
import mpiutil
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def sampnames_mpi(datadir,ext,Nsamp=None):
    if rank == 0:
        if datadir[-1] != '/':
            datadir += '/'
        datafiles = glob.glob(datadir+'*'+ext)
        sampnames_all = [datafile.replace(datadir,'').replace(' ','').replace(ext,'') for datafile in datafiles]
        #print "sampnames_all = {}".format(sampnames_all)
        if not Nsamp is None:
            sampnames_all = sampnames_all[:Nsamp]
        send_name = np.array_split(np.array(sampnames_all),size)
    else:
        send_name = None
    names_dat = comm.scatter(send_name, root= 0)  # @UndefinedVariable
    return names_dat

def total_number_samples(sampnames):
    J = np.sum(mpiutil.collect_int(len(sampnames)))
    return mpiutil.bcast_int(J) 

def total_number_events_and_samples(sampnames,data=None,**kw):
    J = total_number_samples(sampnames)
    #print "J at rank {} = {}".format(rank,J)
    try:
        Nevent = kw['Nevent']
        print "J*Nevent = {}".format(J*Nevent)
        return J*Nevent
    except:
        pass
    if data is None:
        kw_ = copy.deepcopy(kw)
        kw_['scale'] = None
        N_loc = 0
        for name in sampnames:
            N_loc += load_fcsample(name,**kw_).shape[0]
    else:
        N_loc = np.sum([dat.shape[0] for dat in data])
    N = np.sum(mpiutil.collect_int(N_loc))
    N = mpiutil.bcast_int(N)
    return N,J

def partition_all_columns(data,N):
    for d in range(data.shape[1]):
        data[:,d] = np.partition(data[:,d],N)

def pooled_percentile_mpi(q,sampnames,data=None,Nevent=None,i_eventind_load=0,load=True,save=True,**kw):
    if load or save:
        scaleKey = scale_key(kw['datadir'],sampnames,Nevent,i_eventind_load)
        try:
            percentile = ls.load_percentile(kw['datadir'],q,scaleKey)
            print "percentile = {}".format(percentile)
            return percentile
        except IOError:
            pass

    if q > 50:
        if data is None:
            kw_ = copy.deepcopy(kw)
            kw_['loadfilef'] = lambda name: -kw['loadfilef'](name)
            percentiles = -pooled_percentile_mpi(100-q,sampnames,load=False,save=False,**kw_)
        else:
            #print "pooled_percentile_mpi(100-q,-[dat for dat in data],load=False,save=False)[0] = {}".format(pooled_percentile_mpi(100-q,-[dat for dat in data],load=False,save=False)[0])
            percentiles = -pooled_percentile_mpi(100-q,sampnames,[-dat for dat in data],load=False,save=False,**kw)
        if rank == 0 and save:
            ls.save_percentile(percentiles,kw['datadir'],q,scaleKey) 
        return percentiles

    if rank == 0:
        print "Computing new percentiles"
    NN,_ = total_number_events_and_samples(sampnames,data,**kw)
    N = int(np.round(NN*q/100))
    if data is None:
        kw_ = copy.deepcopy(kw)
        kw_['scale'] = None
        data_list = []
        for name in sampnames:
            dat = load_fcsample(name,**kw_)
            partition_all_columns(dat,N-1)
            data_list.append(dat[:N,:])
        data_loc = np.vstack(data_list)
    else:
        data_loc = np.vstack(data)
    partition_all_columns(data_loc,N-1)
    data_all = mpiutil.collect_data(data_loc[:N,:],data_loc.shape[1],'d',MPI.DOUBLE)
    if rank == 0:
        partition_all_columns(data_all,N-1)
        percentiles = data_all[N-1,:]
        print "percentiles = {}".format(percentiles)
        if save:
            ls.save_percentile(percentiles,kw['datadir'],q,scaleKey)
    else:
        percentiles = None

    #return mpiutil.bcast_array_1d(percentiles,'d',MPI.DOUBLE),scaleKey
    #percentiles = np.array(comm.bcast(percentiles))
    #print "percentiles = {}".format(percentiles)
    return comm.bcast(percentiles)

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

def load_fcsample(name,ext,loadfilef,startrow,startcol,datadir,
                  Nevent=None,
                  rm_extreme=True,perturb_extreme=False,
                  i_eventind_load = 0):

    if datadir[-1] != '/':
        datadir += '/'
    datafile = datadir + name + ext
    data = loadfilef(datafile)[startrow:,startcol:]

    try:
        eventind_dic = ls.load_eventind(datadir,Nevent,i_eventind_load,name)
    except:
        ok = non_extreme_ind(data)
        if perturb_extreme:
            data[~ok,:] = add_noise(data[~ok,:])
        if rm_extreme:
            ok_inds = np.nonzero(ok)[0]
        else:
            ok_inds = np.arange(data.shape[0])
            
        if Nevent is None:
            indices = ok_inds
        else:
            indices = npr.choice(ok_inds,Nevent,replace=False)
        eventind_dic = {name: indices}
        ls.save_eventind(eventind_dic,datadir,Nevent,name)  

    data = data[eventind_dic[name],:]

    return data
            

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

    metadata = meta_data(sampnames,marker_lab)         

    compute_new_eventind = True
    if load_eventind:
        try:
            eventind_dic = ls.load_eventind(datadir,Nevent,i_eventind_load)
        except:
            eventind_dic = {}

    if all([name in eventind_dic.keys() for name in sampnames]) or len(eventind) > J:
        compute_new_eventind = False

    if perturb_extreme or (rm_extreme and compute_new_eventind):
        if rm_extreme:
            ok_inds = []
        for j,dat in enumerate(data):
            if rm_extreme and (sampnames[j] in eventind_dic.keys()):
                ok_inds.append([])
            else:
                ok = non_extreme_ind(dat)
                if perturb_extreme:
                    data[j][~ok,:] = add_noise(data[j][~ok,:])
                if rm_extreme:
                    ok_inds.append(np.nonzero(ok)[0]) 

    if not rm_extreme:
        ok_inds = [np.arange(dat.shape[0]) for dat in data]

    for j,dat in enumerate(data):
        if compute_new_eventind and not sampnames[j] in eventind_dic.keys():
            if Nevent is None:
                indices_j = ok_inds[j]
            else:
                indices_j = npr.choice(ok_inds[j],Nevent,replace=False)
            eventind.append(indices_j)
            eventind_dic[sampnames[j]] = indices_j
        else:
            try:
                indices_j = eventind_dic[sampnames[j]]
            except KeyError as e:
                print "Key error({0}): {1}".format(e.errno, e.strerror)
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
    
def percentilescale(data,q = (1.,99.), qvalues = None):
    '''
        Scales the data sets in data so that given quantiles of the pooled data ends up at 0 and 1 respectively.

        data	-	list of data sets
	    q       -	percentiles to be computed. q[0] is the percentile will be scaled to 0, 
                    q[1] is the percentile that will be scaled to 1 (in the pooled data).
        qvalues -   tuple of percentile values. If provided percentiles does not have to be computed.
    '''
    if qvalues is None:
        alldata = np.vstack(data)
        lower,upper = np.percentile(alldata,q,0)
    else:
        lower,upper = qvalues
    
    intercept = lower
    slope = upper - lower

    for j in range(len(data)):
        for m in range(data[0].shape[1]):
            data[j][:,m] = (data[j][:,m]-intercept[m])/slope[m]
    return lower,upper

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

def meta_data(sampnames,marker_lab):
    metasamp = {'names':sampnames}
    metadata = {'samp':metasamp,'marker_lab':marker_lab}    
    return metadata

def non_extreme_ind(data):
    ok = np.ones(data.shape[0],dtype='bool')
    for dd in range(data.shape[1]):
        ok *= data[:,dd] > 1.001*np.min(data[:,dd]) 
        ok *= data[:,dd] < 0.999*np.max(data[:,dd])
    return ok

def add_noise(data,sd=0.01):
    return data + npr.normal(0,sd,data.shape)

def scale_key(datadir,sampnames,Nevent,i_eventind_load):
    sampnames_all = comm.gather(sampnames)#mpiutil.collect_strings(sampnames)
    if rank == 0:
        sampnames_all = list(np.hstack(sampnames_all))
    comm.Barrier()
    sampnames_all = comm.bcast(sampnames_all)
    #print "sampnames_all after bcast at rank {}= {}".format(rank,sampnames_all)
    if rank == 0:
        if datadir[-1] != '/':
            datadir += '/'
        if not os.path.exists(datadir+'scale_dat/'):
            os.mkdir(datadir+'scale_dat/')
        keyfile = datadir+'scale_dat/scale_keys.pkl'
        if os.path.exists(keyfile):
            with open(keyfile,'r') as f:
                parent_dict = pickle.load(f)
        else:
            parent_dict = {}
        curr_dict = parent_dict
        #print "curr_dict = {}".format(curr_dict)
        samp_tuple = tuple(sorted(sampnames_all))
        key = ''
        for j,dat in enumerate([samp_tuple,Nevent,i_eventind_load]):
            try:
                i,curr_dict = curr_dict[dat]
            except:
                curr_dict[dat] = (len(curr_dict),{})
                i,curr_dict = curr_dict[dat]
            key += '_%d' % i
        #print "parent_dict = {}".format(parent_dict)
        with open(keyfile,'w') as f:
            pickle.dump(parent_dict,f)
    else:
        key = None
    return mpiutil.bcast_string(key)
    