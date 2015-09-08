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

def add_noise(data,sd=0.01):
    return data + npr.normal(0,sd,data.shape)

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

def load_fcdata(sampnames=None,scale='percentilescale',q=(1,99),comm=MPI.COMM_WORLD,**kw):

    if sampnames is None:
        sampnames = sampnames_mpi(comm,kw['datadir'],kw['ext'],kw['Nsamp'])

    data = []
    for name in sampnames:
        data.append(load_fcsample(name,**kw))

    if scale == 'percentilescale':
        lower = PercentilesMPI.percentiles_pooled_data(comm,q[0],sampnames,data,**kw)
        upper = PercentilesMPI.percentiles_pooled_data(comm,q[1],sampnames,data,**kw)
        percentilescale(data,qvalues=(lower,upper))

    if scale == 'maxminscale': #not recommended in general to use this option
        maxminscale(data)

    return data

def load_fcsample(name,ext,loadfilef,startrow=0,startcol=0,datadir=None,
                  Nevent=None,
                  rm_extreme=True,perturb_extreme=False,
                  i_eventind_load = 0):
    '''
        Load one fc sample in reproducible way, i.e. so that if 
        subsampling is used, the indices (eventind) are saved and will 
        be automatically loaded when a new subsampling of the same 
        size is requested.
    '''

    if datadir is None:
        raise ValueError('No datadir provided')

    datafile = os.path.join(datadir, name + ext)
    data = loadfilef(datafile)[startrow:,startcol:]

    try:
        eventind = EventInd.load(name,datadir,Nevent,i_eventind_load)
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

        eventind = EventInd(name,Nevent,indices,i_eventind_load)
        eventind.save(datadir)
    else:
        if perturb_extreme:
            ok = non_extreme_ind(data)
            data[~ok,:] = add_noise(data[~ok,:])    

    return data[eventind.indices,:]

def percentilescale(data,q = (1.,99.), qvalues = None):
    '''
        Scales the data sets in data so that given quantiles of the 
        pooled data ends up at 0 and 1 respectively.

        data    -   list of data sets
        q       -   percentiles to be computed. q[0] is the percentile
                    will be scaled to 0, q[1] is the percentile that 
                    will be scaled to 1 (in the pooled data).
        qvalues -   tuple of percentile values. If provided percentiles 
                    does not have to be computed.
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

def sampnames_mpi(comm,datadir,ext,Nsamp=None,namerule=None):
    rank = comm.Get_rank()
    if rank == 0:
#        if datadir[-1] != '/':
#            datadir += '/'
        datafiles = glob.glob(os.path.join(datadir, '*'+ext)) #datadir+'*'+ext)
        sampnames_all = [os.path.basename(datafile).replace(' ', '').\
                            replace(ext, '') for datafile in datafiles]
#        sampnames_all = [datafile.replace(datadir,'').replace(' ','').\
#                                  replace(ext,'') for datafile in datafiles]
        #print "sampnames_all = {}".format(sampnames_all)
        if not namerule is None:
            sampnames_all = [name for name in sampnames_all if namerule(name)]
        if not Nsamp is None:
            sampnames_all = sampnames_all[:Nsamp]
        send_name = np.array_split(np.array(sampnames_all),comm.Get_size())
    else:
        send_name = None
    names_dat = comm.scatter(send_name, root= 0)  # @UndefinedVariable
    return list(names_dat)

def total_number_samples(comm,sampnames):
    print "id(comm) = {}".format(id(comm))
    J = np.sum(mpiutil.collect_int(len(sampnames),comm))
    return mpiutil.bcast_int(J,comm) 

def total_number_events_and_samples(comm,sampnames,data=None,**kw):
    print "computing tot samples at rank {}".format(comm.Get_rank())
    J = total_number_samples(comm,sampnames)
    print "nbr samples computed"
    #print "J at rank {} = {}".format(rank,J)
    try:
        Nevent = kw['Nevent']
        if Nevent is None:
            raise KeyError
        print "J*Nevent = {}".format(J*Nevent)
        return J*Nevent
    except KeyError:
        pass
    if data is None:
        N_loc = 0
        kw_cp = kw.copy()
        if 'scale' in kw_cp:
            del kw_cp['scale']
        for name in sampnames:
            N_loc += load_fcsample(name,**kw_cp).shape[0]
    else:
        N_loc = np.sum([dat.shape[0] for dat in data])
    print "N_loc at rank {} = {}".format(comm.Get_rank(),N_loc)
    N = int(np.sum(mpiutil.collect_int(N_loc,comm)))
    print "N = {}".format(N)
    comm.Barrier()
    print "N at rank {} = {}".format(comm.Get_rank(),N)
    N = mpiutil.bcast_int(N,comm)
    return N,J

class EventInd(object):

    def __init__(self,sampname,Nevent,indices,i=0):
        self.sampname = sampname
        self.indices = indices
        self.Nevent = Nevent
        self.i = i

    def __str__(self):
        return self.name(self.sampname,self.Nevent,self.i)

    @staticmethod
    def name(sampname,Nevent,i):
        s = 'eventind_'+sampname
        if not Nevent is None:
            s += '_' + str(Nevent)
        s += '_' + str(i)
        return s

    @classmethod
    def load(cls,sampname,datadir,Nevent,i=0):
        if not datadir[-1] == '/':
            datadir += '/'
        with open(datadir+'eventinds/'+cls.name(sampname,Nevent,i)+'.npy','r') as f:
            indices = np.load(f)
        return cls(sampname,Nevent,indices)

    def save(self,datadir):
        if not datadir[-1] == '/':
            datadir += '/'
        savedir = datadir + 'eventinds/'
        if not os.path.exists(savedir):
            os.mkdir(savedir)
        fpath = savedir+str(self)+'.npy'
        while os.path.exists(fpath):
            print fpath+' already exists, increasing i'
            self.i += 1
            fpath = savedir+str(self)+'.npy'
        with open(fpath,'w') as f:
            np.save(f,self.indices)

class NoDataError(Exception):
    pass

class PercentilesMPI(object):
    def __init__(self,comm,sampnames,Nevent=None,i_eventind_load=0,datadir=None):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.sampnames = sampnames
        self.Nevent = Nevent
        self.i_eventind_load = i_eventind_load
        self.datadir = datadir
        self.savedir_ = self.savedir(datadir)
        self.key_file_ = self.savedir_+'scale_keys.pkl'
        self.key_dict_ = self.key_dict()
        self.key_ = self.key()

    @classmethod
    def percentiles_pooled_data(cls,comm,q,sampnames,data=None,Nevent=None,i_eventind_load=0,datadir=None,**kw):
        percMPI = cls(comm,sampnames,Nevent,i_eventind_load,datadir)
        return percMPI.percentiles_pooled_data_method(q,data,**kw)

    def percentiles_pooled_data_method(self,q,data=None,load=True,save=True,**kw):
        if load:
            try:
                return self.load_values(q)
            except NoDataError:
                pass
        if q > 50:
            if data is None:
                kw_ = kw.copy()
                kw_['loadfilef'] = lambda name: -kw['loadfilef'](name)
                percentile_values = -self.percentiles_pooled_data_method(100-q,load=False,save=False,**kw_)
            else:
                #print "pooled_percentile_mpi(100-q,-[dat for dat in data],load=False,save=False)[0] = {}".format(pooled_percentile_mpi(100-q,-[dat for dat in data],load=False,save=False)[0])
                percentile_values = -self.percentiles_pooled_data_method(100-q,[-dat for dat in data],load=False,save=False,**kw)
            if save:
                self.save(q,percentile_values)#ls.save_percentile(percentiles,datadir,q,key) 
            return percentile_values

        if self.rank == 0:
            print "Computing new percentiles"
        NN,_ = total_number_events_and_samples(self.comm,self.sampnames,data,datadir=self.datadir,**kw)
        N = int(np.round(NN*q/100))
        if data is None:
            data_list = []
            for name in self.sampnames:
                dat = load_fcsample(name,datadir=self.datadir,**kw)
                self.partition_all_columns(dat,N-1)
                data_list.append(dat[:N,:])
            data_loc = np.vstack(data_list)
        else:
            data_loc = np.vstack(data)
        self.partition_all_columns(data_loc,N-1)
        data_all = mpiutil.collect_data(data_loc[:N,:],data_loc.shape[1],'d',MPI.DOUBLE,self.comm)
        if self.rank == 0:
            self.partition_all_columns(data_all,N-1)
            percentile_values = data_all[N-1,:]
            print "percentiles = {}".format(percentile_values)
            if save:
                self.save(q,percentile_values)
        else:
            percentile_values = None

        #return mpiutil.bcast_array_1d(percentiles,'d',MPI.DOUBLE),key
        #percentiles = np.array(comm.bcast(percentiles))
        #print "percentiles = {}".format(percentiles) 
        return self.comm.bcast(percentile_values)

    @staticmethod
    def partition_all_columns(data,N):
        for d in range(data.shape[1]):
            data[:,d] = np.partition(data[:,d],N)

    def key_dict(self):
        try:
            with open(self.key_file_,'r') as f:
                return pickle.load(f)
        except:
            pass
        return {}

    def key(self):
        sampnames_all = self.comm.gather(self.sampnames)
        if self.rank == 0:
            sampnames_all = list(np.hstack(sampnames_all))
        #comm.Barrier()
        sampnames_all = self.comm.bcast(sampnames_all)
        #print "sampnames_all after bcast at rank {}= {}".format(rank,sampnames_all)
        if self.rank == 0:
            curr_dict = self.key_dict_
            #print "curr_dict = {}".format(curr_dict)
            samp_frozen = frozenset(sampnames_all)
            key_ = ''
            for j,dat in enumerate([samp_frozen,self.Nevent,self.i_eventind_load]):
                try:
                    i,curr_dict = curr_dict[dat]
                except:
                    curr_dict[dat] = (len(curr_dict),{})
                    i,curr_dict = curr_dict[dat]
                key_ += '_%d' % i
            #print "parent_dict = {}".format(parent_dict)
            #with open(self.key_file_,'w') as f:
            #    pickle.dump(self.key_dict_,f)
        else:
            key_ = None
        return mpiutil.bcast_string(key_,self.comm)

    @staticmethod
    def name(q,key):
        return 'percentile_'+str(q)+key

    @staticmethod
    def savedir(datadir):
        if not datadir[-1] == '/':
            datadir += '/'
        return datadir+'scale_dat/'

    def load_values(self,q):
        if self.rank == 0:
            try:
                values = np.loadtxt(self.savedir_+self.name(q,self.key_)+'.txt')
                data_found = True
            except:
                data_found = False
        else:
            values = None
            data_found = False
        data_found = self.comm.bcast(data_found)
        if not data_found:
            raise NoDataError     
        return self.comm.bcast(values)

    def save(self,q,values):
        if self.rank == 0:
            if not os.path.exists(self.savedir_):
                os.mkdir(self.savedir_)
            np.savetxt(self.savedir_+self.name(q,self.key_)+'.txt',values)
            with open(self.savedir_+'scale_keys.pkl','w') as f:
                pickle.dump(self.key_dict_,f,-1)

def load_fcdata_to_root(ext,loadfilef,startrow,startcol,marker_lab,datadir,verbose=True,comm=MPI.COMM_WORLD,**kw):
    '''
        As load_fcdata below, but only loading data to first rank when MPI is used.
    '''
 
    rank = comm.Get_rank()

    data = None
    metasamp = {'names': None}
    metadata = {'samp':metasamp,'marker_lab': None}  

    if rank == 0:
        data,metadata = load_fcdata_no_mpi(ext,loadfilef,startrow,startcol,marker_lab,datadir,**kw)

    if rank == 0 and verbose:
        print "sampnames = {}".format(metadata['samp']['names'])
        print "data sizes = {}".format([dat.shape for dat in data])

    return data,metadata            

def load_fcdata_no_mpi(ext,loadfilef,startrow,startcol,marker_lab,datadir,
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
