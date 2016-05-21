# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 10:35:12 2015

@author: johnsson
"""
from __future__ import division
import time
from mpi4py import MPI


class Timer:
    def __init__(self, comm=MPI.COMM_WORLD):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.curr_time = time.time()
        self.t = {}

    def timepoint(self, name):
        if self.rank == 0:
            old_time = self.curr_time
            self.curr_time = time.time()
            try:
                self.t[name] += self.curr_time - old_time
            except KeyError:
                self.t[name] = self.curr_time - old_time

    def get_timepoint(self, name):
        if self.rank == 0:
            return self.t[name]

    def print_timepoints(self, iter=None):
        if self.rank == 0:
            if iter is None:
                for tp in self.t:
                    print("{}: {} s\n".format(tp, self.t[tp]))
            else:
                for tp in self.t:
                    print("{}: {} s per iteration".format(tp, self.t[tp]/iter))
