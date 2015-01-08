# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 18:35:14 2014

@author: johnsson
"""
from __future__ import division
import numpy as np
from mpi4py import MPI
import copy as cp

import BayesFlow.utils.Bhattacharyya as bhat
from BayesFlow.utils import diptest
import BayesFlow.utils.discriminant as discr


def sort_sucolist(sclist):
    newlist = []
    K = np.amax([np.amax(sc) for sc in sclist])
    notfound = np.ones(K+1)
    i = 0
    while any(notfound):
        i = np.nonzero(notfound)[0][0]
        for sc in sclist:
            if sc.count(i) > 0:
                newlist.append(sc)
                sclist.remove(sc)
                notfound[np.array(sc)] = 0
                break
    return newlist

class SuperComponents(object):

    def __init__(self,BME,hGMM = None):
        if MPI.COMM_WORLD.Get_rank() == 0:
            self.J = BME.J
            self.names = BME.names
            self.mupers = BME.mupers_sim_mean
            self.Sigmapers = BME.Sigmapers_sim_mean
            self.mulat = BME.theta_sim_mean
            self.Sigmalat = BME.Sigmaexp_sim_mean
            self.active_komp = BME.active_komp
            self.classif_freq = cp.deepcopy(BME.classif_freq)
            self.sim = sum(self.classif_freq[0][0,:])
            self.K = BME.K# NB! Outlier cluster not included in super components
            self.p = np.copy(BME.prob_sim_mean[:,:self.K])
            self.complist = [[k] for k in range(self.K)]

    def merge(self,method,thr,**mmfArgs):
        if method == 'demp':
            self.hierarchical_merge(self.get_median_overlap,thr,**mmfArgs)
            self.hclean()
        elif method == 'bhat':
            self.greedy_merge(self.get_median_bh_dist,thr,**mmfArgs)
            self.gclean()
        elif method == 'bhat_hier':
            self.hierarchical_merge(self.get_median_bh_dt_dist,thr,**mmfArgs)
            self.hclean()
        elif method == 'bhat_hier_dip':
            data = mmfArgs.pop('data')
            lowthr = mmfArgs.pop('lowthr')
            dipthr = mmfArgs.pop('dipthr')
            self.hierarchical_merge(self.get_median_bh_dt_dist,thr,data=data,**mmfArgs)
            self.hierarchical_merge(self.get_median_bh_dt_dist_dip,thr=lowthr,data=data,bhatthr=lowthr,dipthr=dipthr,**mmfArgs)
            self.hclean()
        else:
            raise ValueError, 'Unknown method for merging'

    def hierarchical_merge(self,mergeMeasureFun,thr,**mmfArgs):
        if (self.p < 0).any():
            raise ValueError, 'Negative p'
        mm = mergeMeasureFun(**mmfArgs)
        if (mm > thr).any():
            self.merge_kl(np.unravel_index(np.argmax(mm),mm.shape))
            self.hierarchical_merge(mergeMeasureFun,thr,**mmfArgs)

    def merge_kl(self,ind):
        for cl in self.classif_freq:
            cl[:,ind[1]] += cl[:,ind[0]]
            cl[:,ind[0]] = 0
        self.complist[ind[1]] = [self.complist[ind[1]],self.complist[ind[0]]]
        self.complist[ind[0]] = []
        self.p[:,ind[1]] += self.p[:,ind[0]]
        self.p[:,ind[0]] = 2 #Avoiding division by 0

    def greedy_merge(self,mergeMeasureFun,thr,**mmfKw):
        mm = mergeMeasureFun(**mmfKw)
        suco_assign = -np.ones(mm.shape[0],dtype='int')
        sucos = []
        while np.amax(mm) > thr:
            nextmerge = np.unravel_index(np.argmax(mm),mm.shape)
            nextsuco = []
            if suco_assign[nextmerge[0]] == -1 or suco_assign[nextmerge[0]] != suco_assign[nextmerge[1]]:
                for nm in nextmerge:
                    if suco_assign[nm] == -1:
                        nextsuco += [nm]
                    else:
                        nextsuco += sucos[suco_assign[nm]]
                #print "nextmerge = {}".format(nextmerge)
                suco_assign[np.array(nextsuco)] = len(sucos)
                sucos.append(nextsuco)
            mm[nextmerge[0],nextmerge[1]] = -np.inf
            mm[nextmerge[1],nextmerge[0]] = -np.inf
        self.complist_flattened = [sucos[s] for s in np.unique(suco_assign[suco_assign > -1])]
        self.complist_flattened += [[k] for k in np.nonzero(suco_assign == -1)[0]]
        self.complist_flattened = sort_sucolist(self.complist_flattened)

    def hclean(self):
        for i in range(self.complist.count([])):
            self.complist.remove([])
        self.complist_flattened = self.flatten_complist()
        self.p[self.p == 2] = 0
        print "self.complist_flattened = {}".format(self.complist_flattened)
        left_comp = np.array([suco[0] for suco in self.complist_flattened])
        print "left_comp = {}".format(left_comp)
        print "[clf.shape for clf in self.classif_freq] = {}".format([clf.shape for clf in self.classif_freq])
        self.classif_freq = [clf[:,left_comp] for clf in self.classif_freq]
        self.p = self.p[:,left_comp]

    def gclean(self):
        newp = np.empty((self.J,len(self.complist_flattened)))
        new_clf = [np.empty((clf.shape[0],len(self.complist_flattened))) for clf in self.classif_freq]
        for k,co in enumerate(self.complist_flattened):
            newp[:,k] = np.sum(self.p[:,np.array(co)],axis=1)
            for j,clf in enumerate(self.classif_freq):
                new_clf[j][:,k] = np.sum(clf[:,np.array(co)],axis=1)
        self.p = newp
        self.classif_freq = new_clf

    def flatten_complist(self):
        fllist  = []
        for suco in self.complist:
            #print "suco = {}".format(suco)
            suco = self.flatten_supercomp(suco)
            #print "sucofl = {}".format(suco)
            fllist.append(suco)
        fllist = sort_sucolist(fllist)
        return fllist

    def flatten_supercomp(self,suco):
        i = 0
        while i < len(suco):
            if type(suco[i]) is list:
                suco  = suco[:i] + suco[i] + suco[(i+1):]
                suco = self.flatten_supercomp(suco)
            i += 1
        return suco

    def get_median_overlap(self,fixvalind=[],fixval=-1):
        overlap = self.get_overlap()
        return self.get_medprop_pers(overlap,fixvalind,fixval)

    def get_median_bh_dist(self,fixvalind=[],fixval=-1):
        bhd = self.get_bh_dist()
        return self.get_medprop_pers(bhd,fixvalind,fixval)

    def get_median_bh_dt_dist(self,data,fixvalind=[],fixval=-1):
        bhd = self.get_bh_dist_data(data)
        return self.get_medprop_pers(bhd,fixvalind,fixval)

    def get_median_bh_dt_dist_dip(self,data,bhatthr,dipthr,diptabdir,fixvalind=[],fixval=-1):
        bhd = self.get_bh_dist_data(data)
        mbhd = self.get_medprop_pers(bhd,fixvalind,fixval)
        while (mbhd > bhatthr).any():
            ind = np.unravel_index(np.argmax(mbhd),mbhd.shape)
            print "Dip test for {} and {}".format(*ind)
            if self.okdiptest(ind,data,dipthr,diptabdir):
                return mbhd
            fixvalind.append(ind)
            mbhd = self.get_medprop_pers(bhd,fixvalind,fixval)
        return mbhd

    def get_medprop_pers(self,prop,fixvalind=[],fixval=-1):
        med_prop = np.empty(prop[0].shape)
        for k in range(med_prop.shape[0]):
            for l in range(med_prop.shape[1]):
                med_prop[k,l] = np.median([pr[k,l] for pr in prop])
        for ind in fixvalind:
            if len(ind) == 1:
                med_prop[ind,:] = fixval
            else:
                med_prop[ind[0],ind[1]] = fixval
                med_prop[ind[1],ind[0]] = fixval
        return med_prop

    def get_overlap(self):
        overlap = [np.zeros((self.K,self.K)) for cl in self.classif_freq]
        for j,cl in enumerate(self.classif_freq):
            for k in range(self.K):
                for l in range(self.K):
                    overlap[j][k,l] += np.mean(np.prod(cl[:,(k,l)],1))
                overlap[j][k,:] /= self.p[j,k]
                overlap[j][k,k] = 0
            overlap[j] /= self.sim**2
        return overlap

    def get_bh_dist(self):
        bhd = [np.empty((self.K,self.K)) for j in range(self.J)]
        for j in range(self.J):
            for k in range(self.K):
                if self.active_komp[j,k] > 0.05:
                    muk = self.mupers[j,k,:]
                    Sigmak = self.Sigmapers[j,k,:,:]
                else:
                    muk = self.mulat[k,:]
                    Sigmak = self.Sigmalat[k,:,:]
                for l in range(self.K):
                    if self.active_komp[j,l] > 0.05:
                        mul = self.mupers[j,l,:]
                        Sigmal = self.Sigmapers[j,l,:,:]
                    else:
                        mul = self.mulat[l,:]
                        Sigmal = self.Sigmalat[l,:,:]
                    bhd[j][k,l] = bhat.bhattacharyya_dist(muk,Sigmak,mul,Sigmal)
                bhd[j][k,k] = 0
        return bhd

    def get_bh_dist_data(self,data):
        d = data[0].shape[1]
        bhd = [-np.ones((self.K,self.K)) for j in range(self.J)]
        for j in range(self.J):
            for k in range(self.K):
                if (self.p[:,k] < 1.5).any():
                    if np.sum(self.classif_freq[j][:,k] > .5) < d:
                        #print "Too low classif_freq for k = {}".format(k)
                        #print "np.sum(self.classif_freq[{}][:,{}]>.5) = {}".format(j,k,np.sum(self.classif_freq[j][:,k]>.5))
                        #print "np.sum(self.classif_freq[j][:,k]) = {}".format(np.sum(self.classif_freq[j][:,k]))
                        bhd[j][k,:] = 0
                    else:
                        muk,Sigmak = discr.population_mu_Sigma(data[j],self.classif_freq[j][:,k])
                        for l in range(self.K):
                            if l != k and (self.p[:,l] < 1.5).any():
                                if 0:
                                    print "l = {}".format(l)
                                if np.sum(self.classif_freq[j][:,l] > .5) < d:
                                    bhd[j][k,l] = 0
                                else:
                                    if 0:
                                        #print "np.sum(self.classif_freq[{}][:,{}] > .5) = {}".format(j,k,np.sum(self.classif_freq[j][:,k] > .5))
                                        #print "np.sum(self.classif_freq[{}][:,{}]) = {}".format(j,l,np.sum(self.classif_freq[j][:,l]))
                                        #print "np.sum(self.classif_freq[{}][:,{}]) = {}".format(j,k,np.sum(self.classif_freq[j][:,k]))
                                    mul,Sigmal = discr.population_mu_Sigma(data[j],self.classif_freq[j][:,l])
                                    bhd[j][k,l] = bhat.bhattacharyya_dist(muk,Sigmak,mul,Sigmal)   
                bhd[j][k,k] = 0             
        return bhd

    def okdiptest(self,ind,data,thr,diptabdir):
        J = len(data)
        maxbelow = np.ceil(J/4) - 1
        for dim in [None]+range(data[0].shape[1]):
            below = 0
            for j in range(J):
                if diptest.get_pdip_discr_jkl(j,ind[0],ind[1],self.classif_freq,self.p,data,diptabdir,dim) < thr:
                    below += 1
                    if below > maxbelow:
                        print "For ind {} and {}, diptest failed for dim: {}".format(ind[0],ind[1],dim)
                        return False
        print "Diptest ok for {} and {}".format(ind[0],ind[1])
        return True


