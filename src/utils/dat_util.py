# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 16:32:31 2015

@author: johnsson
"""
from __future__ import division
import numpy as np
import numpy.random as npr
#import copy
import glob
from mpi4py import MPI
import pickle
import os
from pprint import pformat

from ..exceptions import NoDataError, OldFileError
from . import mpiutil


def meta_data(sampnames, marker_lab):
    metasamp = {'names': sampnames}
    metadata = {'samp': metasamp, 'marker_lab': marker_lab}
    return metadata


def load_fcdata(datadirs, ext, loadfilef, comm=MPI.COMM_WORLD, sampnames=None,
                Nsamp=None, Nevent=None, i_eventind_load=0,
                scale='percentilescale', q=(1, 99), **kw):

    if sampnames is None:
        sampnames = sampnames_scattered(comm, datadirs, ext, Nsamp)

    data = []
    sampnames_in_dir = []
    for datadir in datadirs:
        dirfiles = os.listdir(datadir)
        for name in sampnames:
            if name+ext in dirfiles:
                data.append(load_fcsample(name, datadir, ext, loadfilef, Nevent,
                                          i_eventind_load, **kw))
                sampnames_in_dir.append(name)
    if set(sampnames) != set(sampnames_in_dir):
        raise ValueError('Not all required sampnames found in given directories')

    if scale == 'percentilescale':
        perc = PercentilesMPI(datadirs, ext, comm, sampnames,
                              Nevent, i_eventind_load, kw)
        lower = perc.percentiles_pooled_data(q[0], data)
        upper = perc.percentiles_pooled_data(q[1], data)
        percentilescale(data, qvalues=(lower, upper))

    if scale == 'maxminscale':  # Not recommended in general to use this option
        maxminscale(data)

    return data


def load_fcsample(name, datadir, ext, loadfilef, Nevent=None, i_eventind_load=0,
                  startrow=0, startcol=0, rm_extreme=True, perturb_extreme=False,
                  overwrite_eventind=False, selectcol=None):
    '''
        Load one fc sample in reproducible way, i.e. so that if
        subsampling is used, the indices (eventind) are saved and will
        be automatically loaded when a new subsampling of the same
        size is requested.

        If rm_extreme=True, data points with exreme values in any
        dimension will be removed (selectcol does not affect this).
        If preturb_extreme is true, these data points will instead be
        perturbed slightly (to avoid singularities).
    '''

    datafile = os.path.join(datadir, name + ext)
    data = loadfilef(datafile)[startrow:, startcol:]

    try:
        eventind = EventInd.load(name, datadir, Nevent, i_eventind_load)
    except (IOError, OldFileError):
        ok = non_extreme_ind(data)
        if perturb_extreme:
            data[~ok, :] = add_noise(data[~ok, :])
        if rm_extreme:
            ok_inds = np.nonzero(ok)[0]
        else:
            ok_inds = np.arange(data.shape[0])

        if Nevent is None:
            indices = ok_inds
        else:
            indices = npr.choice(ok_inds, Nevent, replace=False)

        eventind = EventInd(name, Nevent, indices, i_eventind_load, rm_extreme)
        eventind.save(datadir, overwrite_eventind)
    else:
        if perturb_extreme:
            ok = non_extreme_ind(data)
            data[~ok, :] = add_noise(data[~ok, :])

    if not selectcol is None:
        data = np.ascontiguousarray(data[:, selectcol])

    return data[eventind.indices, :]


def add_noise(data, sd=0.01):
    return data + npr.normal(0, sd, data.shape)


def maxminscale(data):
    """
        Each data set is scaled so that the minimal value in each dimension go to 0
        and the maximal value in each dimension go to 1.

        data - list of data sets
    """
    d = data[0].shape[1]
    for j in range(len(data)):
        for m in range(d):
            data[j][:, m] = (data[j][:, m]-np.min(data[j][:, m]))/(np.max(data[j][:, m])-np.min(data[j][:, m]))


def non_extreme_ind(data):
    ok = np.ones(data.shape[0], dtype='bool')
    for dd in range(data.shape[1]):
        ok *= data[:, dd] > 1.001*np.min(data[:, dd])
        ok *= data[:, dd] < 0.999*np.max(data[:, dd])
    return ok


def percentilescale(data, q=(1., 99.), qvalues=None):
    '''
        Scales the data sets in data so that given quantiles of the
        pooled data ends up at 0 and 1 respectively.

        data    -   list of data sets
        q       -   percentiles to be computed. q[0] is the percentile
                    will be scaled to 0, q[1] is the percentile that
                    will be scaled to 1 (in the pooled data).
        qvalues -   tuple of percentile values. If provided, percentiles
                    does not have to be computed.
    '''
    if qvalues is None:
        alldata = np.vstack(data)
        lower, upper = np.percentile(alldata, q, 0)
    else:
        lower, upper = qvalues

    intercept = lower
    slope = upper - lower

    for j in range(len(data)):
        for m in range(data[0].shape[1]):
            data[j][:, m] = (data[j][:, m]-intercept[m])/slope[m]
    return lower, upper


def sampnames_scattered(comm, datadirs, ext, Nsamp=None, namerule=None):
    """
        Finds all files (or the first Nsamp files)
        with extension 'ext' in directories 'datadirs',
        and splits the names of the files to the workers.

        'namerule' can be specified to select only sample
        names for which namerule(name) is True.
    """
    rank = comm.Get_rank()
    if rank == 0:
        datafiles = []
        for datadir in datadirs:
            datafiles += glob.glob(os.path.join(datadir, '*'+ext))

        sampnames_all = [os.path.basename(datafile).replace(' ', '').
                         replace(ext, '') for datafile in datafiles]
        if not namerule is None:
            sampnames_all = [name for name in sampnames_all if namerule(name)]
        if not Nsamp is None:
            sampnames_all = sampnames_all[:Nsamp]
        send_name = np.array_split(np.array(sampnames_all), comm.Get_size())
    else:
        send_name = None
    names_dat = comm.scatter(send_name, root=0)  # @UndefinedVariable
    return list(names_dat)


def total_number_samples(comm, sampnames):
    J_loc = comm.gather(len(sampnames))
    J = sum(J_loc) if comm.Get_rank() == 0 else None
    J = comm.bcast(J)
    return J


def total_number_events_and_samples(comm, sampnames, data=None, Nevent=None, **kw):
    J = total_number_samples(comm, sampnames)
    if not Nevent is None:
        return J*Nevent, J
    if data is None:
        kw_cp = kw.copy()
        kw_cp['scale'] = None
        kw_cp['rm_extreme'] = False
        data = load_fcdata(comm=comm, sampnames=sampnames, **kw_cp)
    N_loc = sum([dat.shape[0] for dat in data])
    Ns = comm.gather(N_loc)
    N = sum(Ns) if comm.Get_rank() == 0 else None
    N = comm.bcast(N)
    return N, J


class EventInd(object):
    """
        Storing indices for subsampled data.
    """
    def __init__(self, sampname, Nevent, indices, i=0, rm_extreme=True):
        self.sampname = sampname
        self.indices = indices
        self.Nevent = Nevent
        self.i = i
        self.rm_extreme = rm_extreme

    def __str__(self):
        return self.name(self.sampname, self.Nevent, self.i, self.rm_extreme)

    @staticmethod
    def name(sampname, Nevent, i, rm_extreme):
        s = 'eventind_'+sampname
        if not Nevent is None:
            s += '_' + str(Nevent)
        s += '_' + str(i)
        if not rm_extreme:
            s += '_no_rm_extreme'
        return s

    @classmethod
    def load(cls, sampname, datadir, Nevent, i=0, rm_extreme=True):
        fname = os.path.join(os.path.join(datadir, 'eventinds'),
                             cls.name(sampname, Nevent, i, rm_extreme)+'.npy')
        data_fname = glob.glob(os.path.join(datadir, sampname+'.*'))[0]
        if not os.path.exists(fname):
            raise IOError("{} does not exist".format(fname))
        if os.path.getmtime(fname) < os.path.getmtime(data_fname):
            raise OldFileError('Data file was modified after eventind file')
        with open(fname, 'rb') as f:
            indices = np.load(f)
        return cls(sampname, Nevent, indices, rm_extreme)

    def save(self, datadir, overwrite):
        if not datadir[-1] == '/':
            datadir += '/'
        savedir = datadir + 'eventinds/'
        if not os.path.exists(savedir):
            try:
                os.mkdir(savedir)
            except OSError as e:
                if not 'File exists' in str(e):
                    raise
        fpath = savedir+str(self)+'.npy'
        if not overwrite:
            while os.path.exists(fpath):
                print(fpath+' already exists, increasing i')
                self.i += 1
                fpath = savedir+str(self)+'.npy'
        print("Saving new eventind at {}".format(fpath))
        with open(fpath, 'wb') as f:
            np.save(f, self.indices)


class PercentilesMPI(object):
    """
        Class for compouting pooled percentiles of data spread onto
        multiple workers. Computed percentiles are saved by default and
        if wanted again, previously computed percentiles are loaded.
    """
    def __init__(self, datadirs, ext, comm, sampnames, Nevent, i_eventind_load,
                 loaddata_kw=None):
        self.comm = comm
        self.rank = comm.Get_rank()

        self.sampnames = sampnames
        self.sampnames_all = self.comm.gather(self.sampnames)
        if self.rank == 0:
            self.sampnames_all = list(np.hstack(self.sampnames_all))
        self.sampnames_all = self.comm.bcast(self.sampnames_all)

        self.datadirs = datadirs
        self.mtime_data = max([max([os.path.getmtime(os.path.join(datadir, name+ext))
                               for name in self.sampnames_all if name+ext in os.listdir(datadir)]) for datadir in self.datadirs])

        self.Nevent = Nevent
        self.i_eventind_load = i_eventind_load
        self.savedir_ = self.savedir(datadirs[0])
        self.loaddata_kw = pformat(loaddata_kw)
        self.key_file_ = self.savedir_+'scale_keys.pkl'
        self.key_dict_ = self.key_dict()  # This dictionary keeps track
          # of previously computed precentiles.
        self.key_ = self.key()

    @classmethod
    def percentiles_pooled_data_clm(cls, datadirs, ext, comm, q, sampnames, data=None,
                                    Nevent=None, i_eventind_load=0, loaddata_kw=None):
        percMPI = cls(datadirs, ext, comm, sampnames, Nevent, i_eventind_load, loaddata_kw)
        return percMPI.percentiles_pooled_data_method(q, data)

    def percentiles_pooled_data(self, q, data=None, load=True, save=True,
                                loadfilef=None, loaddata_kw=None):
        if loaddata_kw is None:
            loaddata_kw = {}
        if load:
            try:
                return self.load_values(q)
            except (NoDataError, OldFileError):
                pass
        if q > 50:
            if data is None:
                loadfilef = lambda name: -loadfilef(name)
                percentile_values = -self.percentiles_pooled_data(
                    100-q, load=False, save=False, loadfilef=loadfilef, loaddata_kw=loaddata_kw)
            else:
                percentile_values = -self.percentiles_pooled_data(
                    100-q, [-dat for dat in data], load=False, save=False)
            if save:
                self.save(q, percentile_values)
            return percentile_values

        NN, _ = total_number_events_and_samples(
            self.comm, self.sampnames, data, datadirs=self.datadirs,
            loadfilef=loadfilef, **loaddata_kw)
        N = int(np.round(NN*q/100))
        if data is None:
            data_list = []
            for name in self.sampnames:
                dat = load_fcsample(name, datadir=self.datadir, ext=self.ext,
                                    Nevent=self.Nevent, i_eventind_load=self.i_eventind_load,
                                    loadfilef=loadfilef, **loaddata_kw)
                self.partition_all_columns(dat, N-1)
                data_list.append(dat[:N, :])
            data_loc = np.vstack(data_list)
        else:
            data_loc = np.vstack(data)
        self.partition_all_columns(data_loc, N-1)
        data_all = mpiutil.collect_data(
            data_loc[:N, :], data_loc.shape[1], 'd', MPI.DOUBLE, self.comm)
        if self.rank == 0:
            self.partition_all_columns(data_all, N-1)
            percentile_values = data_all[N-1, :]
            if save:
                self.save(q, percentile_values)
        else:
            percentile_values = None

        return self.comm.bcast(percentile_values)

    @staticmethod
    def partition_all_columns(data, N):
        for d in range(data.shape[1]):
            data[:, d] = np.partition(data[:, d], N)

    def key_dict(self):
        try:
            with open(self.key_file_, 'rb') as f:
                return pickle.load(f)
        except IOError:
            pass
        return {}

    def key(self):
        if self.rank == 0:
            curr_dict = self.key_dict_
            #print("curr_dict = {}".format(curr_dict))
            samp_frozen = frozenset(self.sampnames_all)
            key_ = ''
            for j, dat in enumerate([samp_frozen, self.Nevent, self.i_eventind_load,
                                     self.loaddata_kw]):
                try:
                    i, curr_dict = curr_dict[dat]
                except:
                    curr_dict[dat] = (len(curr_dict), {})
                    i, curr_dict = curr_dict[dat]
                key_ += '_%d' % i
            #print("parent_dict = {}".format(parent_dict))
            #with open(self.key_file_, 'w') as f:
            #    pickle.dump(self.key_dict_, f)
        else:
            key_ = None
        return self.comm.bcast(key_)

    @staticmethod
    def name(q, key):
        return 'percentile_'+str(q)+key

    @staticmethod
    def savedir(datadir):
        if not datadir[-1] == '/':
            datadir += '/'
        return datadir+'scale_dat/'

    def load_values(self, q):
        if self.rank == 0:
            fname = self.savedir_+self.name(q, self.key_)+'.txt'
            try:
                values = np.loadtxt(fname)
                data_found = True

                if os.path.getmtime(fname) < self.mtime_data:
                    # saved percentile file older than data.
                    raise OldFileError

            except (IOError, OldFileError) as e_:
                e = e_
                data_found = False
            else:
                e = None
        else:
            values = None
            data_found = False
            e = None
        e = self.comm.bcast(e)
        if isinstance(e, OldFileError):
            raise OldFileError

        data_found = self.comm.bcast(data_found)
        if not data_found:
            raise NoDataError
        return self.comm.bcast(values)

    def save(self, q, values):
        if self.rank == 0:
            print("Saving new percentiles for q = {} in {}: {}".format(q, self.savedir_, values))
            if not os.path.exists(self.savedir_):
                os.mkdir(self.savedir_)
            np.savetxt(self.savedir_+self.name(q, self.key_)+'.txt', values)
            with open(self.savedir_+'scale_keys.pkl', 'wb') as f:
                pickle.dump(self.key_dict_, f, -1)
