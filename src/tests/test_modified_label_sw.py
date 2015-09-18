import os
from mpi4py import MPI

from test_pipeline import Pipeline, SynSample, BadQualityError
from test_balanced import SaveDict


class TrackDict(SaveDict):

    def add_to(self, key, value=1):
        self.load()
        try:
            self._dic[key] += value
        except KeyError:
            self._dic[key] = value
        self.save()

    def append_to(self, key, value):
        self.load()
        try:
            self._dic[key].append(value)
        except KeyError:
            self._dic[key] = [value]
        self.save()

    def print_table(self):
        self.load()
        dict_ = self._dic
        width = min(max([len(str(key)) for key in dict_.keys()]), 15)
        print "\n".join("{:<{width}s}  {}".format(
            str(key), dict_[key], width=width) for key in sorted(dict_.keys()))

if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        fname = 'src/tests/cache/modified_label_switch.pkl'
        if not os.path.exists('src/tests/cache'):
            os.mkdir('src/tests/cache')
        quality_res = TrackDict(fname)

    par_normal = 'src/tests/param/0.py'
    par_modsw = 'src/tests/param/0_modsw.py'

    for i in range(10):

        for name, par in [('normal', par_normal), ('modsw', par_modsw)]:
            pipeline = Pipeline(J=5, K=15, N=1000, d=2, C=4,
                                data_class=SynSample, ver='B',
                                par_file=par)
            pipeline.run()
            if rank == 0:
                quality_res.append_to('number of mod_sw switches',
                                      pipeline.number_label_switches)
                try:
                    pipeline.quality_check()
                except BadQualityError:
                    quality_res.add_to(name+' fail')
                else:
                    quality_res.add_to(name+' pass')
        if rank == 0:
            quality_res.print_table()
