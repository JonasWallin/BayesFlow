# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 21:23:52 2015

@author: johnsson
"""
import cPickle as pickle

def save_object_to_file(obj, filename):
	with open(filename, 'wb+') as output:
		pickle.dump(obj, output, -1)
        
def save_object(obj, savedir, filename=None,):
	if not savedir.endswith('/'):
		savedir += '/'
	if not filename is None:
		save_object_to_file(obj,savedir+filename)
		return
	defaultnames = {'HMlogB':'blog','HMlog':'log','HMElog':'log',
	'BMres':'hmres_'}
	filename = defaultnames[obj.__class__.__name__]
	if hasattr(obj,'mergeMeth'):
		filename += obj.mergeMeth
	filename += '.pkl'
	save_object_to_file(obj,savedir+filename)
				
def load_HMlogB(savedir):
	return pickle.load(open(savedir+'blog'+'.pkl','rb'))
	
def load_HMlog(savedir):
	return pickle.load(open(savedir+'log'+'.pkl','rb'))
	
def load_HMElog(savedir):
	return pickle.load(open(savedir+'log'+'.pkl','rb'))
	
def load_HMres(savedir,mergemeth):
	return pickle.load(open(savedir+'hmres_'+mergemeth+'.pkl','rb'))