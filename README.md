# BayesFlow
version 0.1

## Installation
- Ensure that you have obtained the dependencies listed below.
- Update the configuration file setup.cfg: 
  - Specify which libraries you are using for linear algebra (e.g. library = cblas, clapack)
  - Specify library directories and header file directories for the listed libraries.
  - If needed, specify additional arguments that should be passed to the compiler (e.g. if you use MKL)
NB! If you use MKL, in addition to the usual arguments to the compiler, you also need to pass the argument -DMKL.
- Go to the main directory (BayesFlow/) and run `python setup.py install`.

## Examples
- demo_HF.py: Demonstration of how to use BayesFlow for healthyFlowData dataset. 

## Dependencies

The package has the following dependencies:
- Python, including packages numpy, cython, matplotlib, mpi4py, yaml, json, (rpy2)
- OpenMPI
- Libraries for linear algebra computations with c interfaces, e.g. CBLAS and CLAPACK or MKL. 

The dependence on rpy2 is only needed for computing dip test and loading data from R package healhtyFlowData.
If installation without this dependency is wanted, files

```
src/__init__   
src/utils/__init__  
src/data/__init__
```

should be replaced by
```
src/__init_mindep__  
src/utils/__init_mindep__  
src/data/__init_mindep__
```
for example do `mv src/__init_mindep__ src/__init__` et.c. 

## How to obtain dependencies

### Ubuntu

Python, OpenMPI and the required Python packages can be installed by:
```
sudo apt-get install python, openmpi  
sudo pip install numpy, cython, matplotlib, mpi4py, pyyaml, json, rpy2
```
For linear algebra, you can for example use ATLAS, which includes BLAS and the 
routines from LAPACK that we need. ATLAS can be installed from http://math-atlas.sourceforge.net/.

Then put the libraries you will use in the configuration file setup.cfg
along with directories for headers (include_dirs) and for the libraries (library_dirs),
for example:
```
[build_ext]
include_dirs = 	/usr/include:/local/include:/opt/local/include:/usr/include/atlas
library_dirs = 	/opt/local/lib:/usr/lib:/usr/local/lib
libraries = blas, lapack_atlas
```

If you instead use MKL, your setup.cfg-file might look like this:
```
[build_ext]
include_dirs =      ${MKLROOT}/include
library_dirs =      ${MKLROOT}/lib/intel64
libraries =         mkl_intel_lp64, mkl_core, mkl_sequential, pthread
extra_compile_args = -m64 -Wl --no-as-needed -DMKL
extra_link_args =   -Wl --no-as-needed
```
Note that you need to pass the extra argument `-DMKL` to the compiler.

### Mac

Python and OpenMPI can be installed using homebrew.
First install homebrew following the instructions at http://brew.sh.
Then install pyhon and OpenMPI:
```
brew install python, openmpi
```
Then python packages can be installed using
```
pip install numpy, cython, matplotlib, mpi4py, json, pyyaml, rpy2
```
For linear algebra c libraries, you can use the BLAS and LAPACK distributions provided in the Accelerate veclib framework.
Then put
```
[build_ext]
include_dirs = /usr/include:/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/Headers/
libraries = cblas, clapack
```
in your configuration file setup.cfg.





