# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 21:23:52 2015

@author: johnsson
"""
import cPickle as pickle
import os
import numpy as np
import json
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
        if objtype == 'thetas':
            save_thetas(obj,savedir)
            return
        if objtype == 'eventind':
            save_eventind(obj,savedir)
        defaultnames = {'HMlogB':'blog','HMlog':'log_ne','HMElog':'log',
        'BMres':'hmres_','HMres':'hmres_','faillog':'flog','triallog':'tlog',
        'eventind':'eventind'}
        if objtype is None:
            objtype = obj.__class__.__name__
        filename = defaultnames[objtype]
        if hasattr(obj,'mergeMeth'):
            filename += obj.mergeMeth
        filename += '.pkl'
        save_object_to_file(obj,savedir+filename)

def eventfilename(savedir,Nevent,i):
    if not Nevent is None:
        eventfile = 'eventind_%d_%d.json' % (Nevent,i)
    else:
        eventfile = 'eventind_%d.json' % i
    return savedir+eventfile

def save_eventind(eventind_dict,savedir,Nevent):
    i = 0
    eventfile = eventfilename(savedir,Nevent,i)
    while os.path.exists(eventfile):
        i += 1
        eventfile = eventfilename(savedir,Nevent,i)

    for key in eventind_dict:
        eventind_dict[key] = list(eventind_dict[key])
    #print "eventind_dict = {}".format(eventind_dict)
    #print "list(eventind_dict['sample8']) = {}".format(list(eventind_dict['sample8']))    
    with open(eventfile,'w') as f:
        json.dump(eventind_dict,f)

def load_eventind(savedir,Nevent=None,i=0):
    eventfile = eventfilename(savedir,Nevent,i)
    if os.path.exists(eventfile):
        with open(eventfile, 'r') as f:
            try:
                line1 = f.readline()
                eventind_dic = yaml.load(line1)
            except:
                print "Loading eventind Matlab style"
                lines = [line1]+f.readlines()
                line = ' '.join(lines)
                line = line.replace('\r','')
                line = line.replace('\t','')
                line = line.replace('\n','')
                line = re.sub('\[([0-9]+)\]','\\1',line)
                eventind_dic = yaml.load(line)
        print "Events loaded ok"
        for key in eventind_dic:
            eventind_dic[key] = np.array(eventind_dic[key])
        return eventind_dic
    print "No .json file with events exist, loading .pkl file instead"
    try:
        return pickle.load(open(savedir+'eventind.pkl','rb'))
    except:
        print "Eventinds not saved previously, returning empty dictionary"
    return {}

def load_burnlog(savedir):
    return load_HMlogB(savedir)

def load_prodlog(savedir):
    return load_HMElog(savedir)
                
def load_HMlogB(savedir):
    return pickle.load(open(savedir+'blog'+'.pkl','rb'))
    
def load_HMlog(savedir):
    return pickle.load(open(savedir+'log_ne'+'.pkl','rb'))
    
def load_HMElog(savedir):
    return pickle.load(open(savedir+'log'+'.pkl','rb'))

def load_faillog(savedir):
    return pickle.load(open(savedir+'flog'+'.pkl','rb'))
    
def load_triallog(savedir):
    return pickle.load(open(savedir+'tlog'+'.pkl','rb'))
    
def load_HMres(savedir,mergemeth):
    return pickle.load(open(savedir+'hmres_'+mergemeth+'.pkl','rb'))
    
