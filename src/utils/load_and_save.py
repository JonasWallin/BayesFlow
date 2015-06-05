# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 21:23:52 2015

@author: johnsson
"""
import cPickle as pickle
import os
import numpy as np
import json
import jsonutil
import yaml
import re
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def save_object_to_file(obj, filename):
    if rank == 0:
        with open(filename, 'wb+') as output:
            pickle.dump(obj, output, -1)
        
def save_object(obj, savedir, filename=None, objtype=None):
    if rank == 0:
        if not savedir.endswith('/'):
            savedir += '/'
        if not filename is None:
            save_object_to_file(obj,savedir+filename)
            return
        if objtype is None:
            objtype = obj.__class__.__name__

        if objtype == 'eventind':
            save_eventind(obj,savedir)
            return
        if objtype == 'HMlogB':
            save_HMlogB(obj,savedir)
            return
        if objtype == 'HMlog':
            save_HMlog(obj,savedir)
            return
        if objtype == 'HMElog':
            save_HMElog(obj,savedir)
            return

        defaultnames = {'HMlogB':'blog','HMlog':'log_ne','HMElog':'log',
        'BMres':'hmres_','HMres':'hmres_','faillog':'flog','triallog':'tlog',
        'eventind':'eventind'}
        filename = defaultnames[objtype]
        if hasattr(obj,'mergeMeth'):
            filename += obj.mergeMeth
        filename += '.pkl'
        print "saving file {}".format(filename)
        save_object_to_file(obj,savedir+filename)

def load_burnlog(savedir):
    return load_HMlogB(savedir)

def load_prodlog(savedir):
    return load_HMElog(savedir)

def save_HMlogB(obj,savedir):
    json.dump(obj,savedir+'blog.json',cls=jsonutil.HMlogBEncoder)

def load_HMlogB(savedir):
    if not savedir[-1] == '/':
        savedir += '/'
    try:
        with open(savedir+'blog.pkl','rb') as f:
            return pickle.load(f)
    except:
        with open(savedir+'blog.json') as f:
            return json.load(f,object_hook=jsonutil.hmlogb_decoder)

def save_HMlog(obj,savedir,logname=None):
    if not savedir[-1] == '/':
        savedir += '/'
    if logname is None:
        logname = 'log_ne'
    json.dump(obj,savedir+logname+'.json',cls=jsonutil.HMlogEncoder)

    savedir += 'syndata/'
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    for j,name in obj.savesampnames_loc:
        np.savetxt(savedir+name+'_MODEL.txt', obj.Y_sim_loc[j])
    if rank == 0:
        np.savetxt(savedir+'pooled_MODEL.txt', obj.Y_pooled_sim)

def load_HMlog(savedir,name=None):
    if name is None:
        name = 'log_ne'
    if not savedir[-1] == '/':
        savedir += '/'
    try:
        with open(savedir+name+'.pkl','rb') as f:
            return pickle.load(f)
    except:
        with open(savedir+name+'.json','rb') as f:
            hmlog = json.load(f,object_hook=jsonutil.hmlog_decoder)
        hmlog.set_syndata_dir(savedir+'syndata/')
        return hmlog

def save_HMElog(obj,savedir):
    if not savedir[-1] == '/':
        savedir += '/'
    save_HMlog(obj,savedir,logname='log')
    savedir += 'classif_freq/'
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    try:
        for j,name in obj.names_loc:
            np.savetxt(savedir+name+'_CLASSIF_FREQ.txt',obj.classif_freq_loc[j])
    except:
        if rank == 0:
            for j,name in obj.names:
                np.savetxt(savedir+name+'_CLASSIF_FREQ.txt',obj.classif_freq[j])

def load_HMElog(savedir):
    if not savedir[-1] == '/':
        savedir += '/'
    try:
        with open(savedir+'log.pkl','rb') as f:
            return pickle.load(f)
    except:
        with open(savedir+'log.json','rb') as f:
            hmelog = json.load(f,object_hook=jsonutil.hmlog_decoder)
            return hmelog
## TODO! Load also other data



def load_percentilescale(datadir,q,scaleKey):
    q1,q2 = q
    lower_q = load_percentile(datadir,q1,scaleKey)
    upper_q = load_percentile(datadir,q2,scaleKey)
    intercept = lower_q
    slope = upper_q - lower_q
    return intercept,slope

def percentilename(datadir,q,scaleKey):
    if datadir[-1] != '/':
        datadir += '/'
    savedir = datadir + 'scale_dat/'
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    return savedir+'percentile_'+str(q)+scaleKey+'.txt'

def load_percentile(datadir,q,scaleKey):
    filename = percentilename(datadir,q,scaleKey)
    return np.loadtxt(filename)

def save_percentile(percentile,datadir,q,scaleKey):
    if datadir[-1] != '/':
        datadir += '/'
    filename = percentilename(datadir,q,scaleKey)
    np.savetxt(filename,percentile)

def save_eventind(eventind_dict,savedir,Nevent,name=None):
    i = 0
    eventfile = eventfilename(savedir,Nevent,i,name)
    while os.path.exists(eventfile):
        i += 1
        eventfile = eventfilename(savedir,Nevent,i)

    for key in eventind_dict:
        eventind_dict[key] = list(eventind_dict[key])
    #print "eventind_dict = {}".format(eventind_dict)
    #print "list(eventind_dict['sample8']) = {}".format(list(eventind_dict['sample8']))    
    with open(eventfile,'w') as f:
        json.dump(eventind_dict,f)

def load_eventind(savedir,Nevent=None,i=0,name=None,pkl=False,verbose=False):
    if pkl:
        return pickle.load(open(savedir+'eventind.pkl','rb'))
    eventfile = eventfilename(savedir,Nevent,i,name)
    if not os.path.exists(eventfile):
        eventfile = eventfilename(savedir,Nevent,i)
    with open(eventfile, 'r') as f:
        try:
            line1 = f.readline()
            eventind_dic = yaml.load(line1)
        except:
            if verbose:
                print "Loading eventind Matlab style"
            lines = [line1]+f.readlines()
            line = ' '.join(lines)
            line = line.replace('\r','')
            line = line.replace('\t','')
            line = line.replace('\n','')
            line = re.sub('\[([0-9]+)\]','\\1',line)
            eventind_dic = yaml.load(line)
    if verbose:
        print "Events loaded ok"
    for key in eventind_dic:
        eventind_dic[key] = np.array(eventind_dic[key])
    return eventind_dic

def eventfilename(savedir,Nevent,i,name=None):
    if not savedir[-1] == '/':
        savedir += '/'
    eventdir = savedir+'eventinds/'
    if not os.path.exists(eventdir):
        os.mkdir(eventdir)
    eventfile = 'eventind'
    if not name is None:
        eventfile += '_'+name
    if not Nevent is None:
        eventfile += '_%d' % Nevent
    eventfile += '_%d.json' % i
    return eventdir+eventfile


                


    


def load_faillog(savedir):
    return pickle.load(open(savedir+'flog'+'.pkl','rb'))
    
def load_triallog(savedir):
    return pickle.load(open(savedir+'tlog'+'.pkl','rb'))
    
def load_HMres(savedir,mergemeth):
    return pickle.load(open(savedir+'hmres_'+mergemeth+'.pkl','rb'))
    
