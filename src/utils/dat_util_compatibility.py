'''
    Old dat_util functions, available for compatibility.
'''
from mpi4py import MPI
import glob
import numpy as np
import numpy.random as npr

from . import load_and_save as ls
from .dat_util import non_extreme_ind, add_noise, maxminscale, percentilescale, meta_data


def load_fcdata_to_root(ext, loadfilef, startrow, startcol, marker_lab, datadir,
                        verbose=True, comm=MPI.COMM_WORLD, **kw):
    '''
        As load_fcdata below, but only loading data to first rank when MPI is used.
    '''

    rank = comm.Get_rank()

    data = None
    metasamp = {'names': None}
    metadata = {'samp': metasamp, 'marker_lab': None}

    if rank == 0:
        data, metadata = load_fcdata_no_mpi(ext, loadfilef, startrow, startcol, marker_lab, datadir, **kw)

    if rank == 0 and verbose:
        print "sampnames = {}".format(metadata['samp']['names'])
        print "data sizes = {}".format([dat.shape for dat in data])

    return data, metadata


def load_fcdata_no_mpi(ext, loadfilef, startrow, startcol, marker_lab, datadir,
                       Nsamp=None, Nevent=None, scale='percentilescale',
                       rm_extreme=True, perturb_extreme=False, eventind=[],
                       eventind_dic={}, datanames=None, load_eventind=True,
                       i_eventind_load=0):
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
        scale           -   'maxminscale', 'percentilescale' or None. If 'maxminscale' each sample is scaled so that
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
        J = min(J, Nsamp)

    data = []
    sampnames = []
    print "J = {} samples will be loaded".format(J)
    for j in range(J):
        sampnames.append(datafiles[j].replace(datadir, '').replace(' ', '').replace(ext, ''))
        data.append(loadfilef(datafiles[j])[startrow:, startcol:])

    metadata = meta_data(sampnames, marker_lab)

    compute_new_eventind = True
    if load_eventind:
        try:
            eventind_dic = ls.load_eventind(datadir, Nevent, i_eventind_load)
        except:
            eventind_dic = {}

    if all([name in eventind_dic.keys() for name in sampnames]) or len(eventind) > J:
        compute_new_eventind = False

    if perturb_extreme or (rm_extreme and compute_new_eventind):
        if rm_extreme:
            ok_inds = []
        for j, dat in enumerate(data):
            if rm_extreme and (sampnames[j] in eventind_dic.keys()):
                ok_inds.append([])
            else:
                ok = non_extreme_ind(dat)
                if perturb_extreme:
                    data[j][~ok, :] = add_noise(data[j][~ok, :])
                if rm_extreme:
                    ok_inds.append(np.nonzero(ok)[0])

    if not rm_extreme:
        ok_inds = [np.arange(dat.shape[0]) for dat in data]

    for j, dat in enumerate(data):
        if compute_new_eventind and not sampnames[j] in eventind_dic.keys():
            if Nevent is None:
                indices_j = ok_inds[j]
            else:
                indices_j = npr.choice(ok_inds[j], Nevent, replace=False)
            eventind.append(indices_j)
            eventind_dic[sampnames[j]] = indices_j
        else:
            try:
                indices_j = eventind_dic[sampnames[j]]
            except KeyError as e:
                print "Key error({0}): {1}".format(e.errno, e.strerror)
                indices_j = eventind[j]
        data[j] = dat[indices_j, :]

    if compute_new_eventind:
        ls.save_eventind(eventind_dic, datadir, Nevent)

    if scale == 'maxminscale':
        maxminscale(data)
    elif scale == 'percentilescale':
        percentilescale(data)
    elif not scale is None:
        raise ValueError("Scaling {} is unsupported".format(scale))

    return data, metadata

