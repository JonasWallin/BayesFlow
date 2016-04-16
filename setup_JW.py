# -*- coding: utf-8 -*-
# routines for lapack:http://www.physics.orst.edu/~rubin/nacphy/lapack/linear.html
#https://developer.apple.com/library/prerelease/ios/documentation/Accelerate/Reference/BLAS_Ref/index.html#//apple_ref/doc/uid/TP30000414-SW9
from distutils.core import setup
from distutils.extension import Extension
from distutils.command.clean import clean
from distutils.command.install import install
from Cython.Distutils import build_ext
from numpy import get_include



class MyInstall(install):

    # Calls the default run command, then deletes the build area
    # (equivalent to "setup clean --all").
    def run(self):
        install.run(self)
        c = clean(self.distribution)
        c.all = True
        c.finalize_options()
        c.run()

include_dirs = [get_include(),
               #  '/usr/include', '/usr/local/include',
                '/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/Headers/']
setup(name='BayesFlow',
      version='0.1',
	  author='Jonas Wallin',
	  url='',
   author_email='jonas.wallin81@gmail.com',
	  requires=['numpy (>=1.3.0)',
                'cython (>=0.17)',
                'matplotlib',
				'mpi4py',
				'logisticnormal'],
      cmdclass = {'build_ext': build_ext},
				 #'install': MyInstall},
        packages=['BayesFlow.utils.initialization','BayesFlow','BayesFlow.PurePython',
				'BayesFlow.PurePython.distribution','BayesFlow.distribution','BayesFlow.utils','BayesFlow.mixture_util','BayesFlow.tests'],
      package_dir={'BayesFlow': 'src/'},
	  package_data={'': ['data/*.dat']},
      ext_modules = [Extension("BayesFlow.mixture_util.GMM_util",sources=["src/mixture_util/GMM_util.pyx","src/mixture_util/c/draw_x.c","src/distribution/c/distribution_c.c"],
                               include_dirs = include_dirs,
                                libraries=['gfortran','m','cblas','clapack'],
                               language='c')
                ,Extension("BayesFlow.distribution.distribution_cython",sources=["src/distribution/distribution_cython.pyx","src/distribution/c/distribution_c.c"],
                               include_dirs = include_dirs,
                               libraries=['gfortran','m','cblas','clapack'],
                               language='c'),
				Extension("BayesFlow.distribution.logisticNormal",sources=["src/distribution/logisticNormal.pyx","src/distribution/c/distribution_c.c"],
                               	include_dirs = include_dirs,
							    libraries=['gfortran','m','cblas','clapack'],
                               language='c')],
      )
