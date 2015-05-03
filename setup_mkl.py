# -*- coding: utf-8 -*-
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from numpy import get_include
import os

setup(name='BayesFlow',
      version='0.1',
	  author='Jonas Wallin',
	  url='',
   author_email='jonas.wallin81@gmail.com',
	  requires=['numpy (>=1.3.0)',
                'cython (>=0.17)',
                'matplotlib',
				'mpi4py'],
      cmdclass = {'build_ext': build_ext},
        packages=['BayesFlow','BayesFlow.PurePython',
				'BayesFlow.PurePython.distribution','BayesFlow.distribution','BayesFlow.utils','BayesFlow.mixture_util','BayesFlow.data'],
      package_dir={'BayesFlow': 'src/'},
      ext_modules = [Extension("BayesFlow.mixture_util.GMM_util",sources=["src/mixture_util/GMM_util.pyx","src/mixture_util/c/draw_x.c","src/distribution/c/distribution_c.c"],
                               library_dirs = [os.environ['MKLROOT']+'/lib/intel64'],
                               libraries=['mkl_intel_lp64', 'mkl_core', 'mkl_sequential', 'pthread', 'm'],
                               include_dirs = [get_include(),os.environ['MKLROOT']+'/include'],
                               extra_compile_args=['-m64','-Wl,--no-as-needed','-DMKL'],
                               extra_link_args = ['-Wl,--no-as-needed'],
                               language='c')
                ,Extension("BayesFlow.distribution.distribution_cython",sources=["src/distribution/distribution_cython.pyx","src/distribution/c/distribution_c.c"],
                               library_dirs = [os.environ['MKLROOT']+'/lib/intel64'],
                               libraries=['mkl_intel_lp64', 'mkl_core', 'mkl_sequential', 'pthread', 'm'],
                               include_dirs = [get_include(),os.environ['MKLROOT']+'/include'],
                               extra_compile_args=['-m64','-DMKL'],
                               extra_link_args = ['-Wl,--no-as-needed'],
                               language='c')],
      )



