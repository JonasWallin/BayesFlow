# -*- coding: utf-8 -*-
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from numpy import get_include

setup(name='BayesFlow',
      version='0.1',
      author='Jonas Wallin',
      url='',
      author_email='jonas.wallin81@gmail.com',
      requires=['numpy (>=1.3.0)',
                'cython (>=0.17)',
                'matplotlib',
                'mpi4py'
                'yaml',
                'json'],
      cmdclass = {'build_ext': build_ext},
      packages=['BayesFlow','BayesFlow.PurePython',
                'BayesFlow.PurePython.distribution','BayesFlow.distribution',
                'BayesFlow.utils','BayesFlow.mixture_util',
                'BayesFlow.data'],
      package_dir={'BayesFlow': 'src/'},
      ext_modules = [Extension("BayesFlow.mixture_util.GMM_util",
                               sources=["src/mixture_util/GMM_util.pyx",
                                        "src/mixture_util/c/draw_x.c",
                                        "src/distribution/c/distribution_c.c"],
                               extra_compile_args = ['-m64'],
                               #extra_link_args = ['-Wl, --no-as-needed'],       
                               libraries=['gfortran','m'],
                               language='c'),
                     Extension("BayesFlow.distribution.distribution_cython",
                               sources=["src/distribution/distribution_cython.pyx",
                                        "src/distribution/c/distribution_c.c"],
                               extra_compile_args = ['-m64'],
                               #extra_link_args = ['-Wl, --no-as-needed'],     
                               include_dirs = [get_include()],
                               libraries=['gfortran','m'],
                               language='c')],
      )
