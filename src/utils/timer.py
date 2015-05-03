# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 10:35:12 2015

@author: johnsson
"""
import time
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

class Timer:
    def __init__(self):
        self.curr_time = time.time()
        self.t = {}
        
    def timepoint(self,name):
        if rank == 0:
            old_time = self.curr_time
            self.curr_time = time.time()
            self.t[name] = self.curr_time - old_time
        
    def get_timepoint(self,name):
        if rank == 0:
            return self.t[name]
    
    def print_timepoints(self):
        if rank == 0:
            print self.t