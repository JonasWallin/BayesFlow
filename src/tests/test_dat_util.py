import unittest
import tempfile
import shutil
import numpy as np
from mpi4py import MPI
import os

from BayesFlow.utils.dat_util import sampnames_scattered, total_number_samples, total_number_events_and_samples
from BayesFlow.utils.dat_util import EventInd, load_fcdata
from BayesFlow.tests.pipeline import SynSample
from BayesFlow.exceptions import OldFileError


class TestDatUtil(unittest.TestCase):

    def setUp(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        if self.rank == 0:
            self.tmpdirs = [tempfile.mkdtemp(), tempfile.mkdtemp()]
        else:
            self.tmpdirs = None
        self.tmpdirs = self.comm.bcast(self.tmpdirs)
        self.ext = '.txt'
        self.loadfilef = lambda filename: np.loadtxt(filename)

    def startup(self):
        if self.rank == 0:
            Js = [3, 5]
            self.J = sum(Js)
            self.N = 0
            self.synsamps = []
            self.names = []
            for i, (J, dir_) in enumerate(zip(Js, self.tmpdirs)):
                ns = 10*np.arange(1, J+1)
                self.synsamps.append([SynSample(j=i*self.J+j, n_obs=n, d=2, C=5) for j, n in zip(range(J), ns)])
                for synsamp in self.synsamps[-1]:
                    synsamp.generate_data(savedir=dir_)
                    self.names.append(synsamp.name)
                self.N += np.sum(ns)
        else:
            self.J = None
            self.N = None
        self.J = self.comm.bcast(self.J)
        self.N = self.comm.bcast(self.N)

    def test_sampnames_scattered(self):
        self.startup()
        names = sampnames_scattered(self.comm, self.tmpdirs, self.ext)
        self.assertTrue(len(names) <= self.J/self.comm.Get_size()+1)
        allnames = self.comm.gather(names)
        if self.rank == 0:
            allnames = [name for names_ in allnames for name in names_]
            self.assertEqual(set(allnames), set(self.names))

    def test_total_number_samples(self):
        self.startup()
        names = sampnames_scattered(self.comm, self.tmpdirs, self.ext)
        self.assertEqual(total_number_samples(self.comm, names), self.J)

    def test_total_numbers_events_and_samples(self):
        self.startup()
        names = sampnames_scattered(self.comm, self.tmpdirs, self.ext)
        N, J = total_number_events_and_samples(
            self.comm, names, datadirs=self.tmpdirs, ext=self.ext,
            loadfilef=self.loadfilef)
        self.assertEqual(N, self.N)
        self.assertEqual(J, self.J)

    def test_eventind(self):
        self.startup()
        if self.rank == 0:
            name = self.names[0]
            datadir = self.tmpdirs[0]
            Nevent = 5
            i = 3
            rm_extreme = False
            eventind = EventInd(name, Nevent=Nevent, indices=np.arange(5), i=i, rm_extreme=rm_extreme)
            eventind.save(datadir, overwrite=False)
            EventInd.load(name, datadir, Nevent, i=i, rm_extreme=rm_extreme)
        self.startup()
        if self.rank == 0:
            with self.assertRaises(OldFileError):
                EventInd.load(name, datadir, Nevent, i=i, rm_extreme=rm_extreme)

    def test_load_fcdata(self):
        self.startup()
        data = load_fcdata(self.tmpdirs, self.ext, self.loadfilef, comm=self.comm)
        data2 = load_fcdata(self.tmpdirs, self.ext, self.loadfilef, comm=self.comm)
        for dat, dat2 in zip(data, data2):
            np.testing.assert_array_almost_equal(dat, dat2)

    def tearDown(self):
        if self.rank == 0:
            for tmp in self.tmpdirs:
                shutil.rmtree(tmp)

if __name__ == '__main__':
    unittest.main()
    #suite = unittest.TestLoader().loadTestsFromTestCase(TestDatUtil)
    #suite.debug()