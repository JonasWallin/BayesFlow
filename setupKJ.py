# -*- coding: utf-8 -*-
# routines for lapack:http://www.physics.orst.edu/~rubin/nacphy/lapack/linear.html
#https://developer.apple.com/library/prerelease/ios/documentation/Accelerate/Reference/BLAS_Ref/index.html#//apple_ref/doc/uid/TP30000414-SW9
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from numpy import get_include

setup(name='bayesianmixture',
      version='0.1',
	  author='Jonas Wallin',
	  url='',
   author_email='jonas.wallin81@gmail.com',
	  requires=['numpy (>=1.3.0)',
                'cython (>=0.17)',
                'matplotlib',
				'mpi4py'],
      cmdclass = {'build_ext': build_ext},
        packages=['bayesianmixture','bayesianmixture.PurePython',
				'bayesianmixture.PurePython.distribution','bayesianmixture.distribution','bayesianmixture.utils','bayesianmixture.mixture_util'],
      package_dir={'bayesianmixture': 'src/'},
	  package_data={'': ['data/*.dat']},
      ext_modules = [Extension("bayesianmixture.mixture_util.GMM_util",sources=["src/mixture_util/GMM_util.pyx","src/mixture_util/c/draw_x.c","src/distribution/c/distribution_c.c"],
                               include_dirs = [get_include(),
                                               '/usr/include', '/usr/local/include',
                                               '/usr/local/atlas/include'],
                               library_dirs = ['/usr/lib', '/usr/local/lib', '/usr/local/atlas/lib'],
                               libraries=['gfortran','m','atlas'],
                               language='c')
                ,Extension("bayesianmixture.distribution.distribution_cython",sources=["src/distribution/distribution_cython.pyx","src/distribution/c/distribution_c.c"],
                               include_dirs = [get_include(),
                                               '/usr/include', '/usr/local/include',
                                               '/usr/local/atlas/include'],
                               library_dirs = ['/usr/lib', '/usr/local/atlas/lib','/usr/local/lib'],

                               libraries=['gfortran','m','atlas'],
                               language='c')],
      )

