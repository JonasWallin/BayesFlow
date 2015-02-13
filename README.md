# BayesFlow

version 0.1

## Dependencies

The package has the following dependencies:
- Python, including packages numpy, cython, matplotlib, mpi4py and rpy2
- openmpi
- C libraries for linear algebra computations, e.g. cblas and clapack or mkl. 
- gfortran

The dependence on rpy2 is not needed for core functionality, and if installation without this dependency is wanted, files

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
for example do `mv src/__init_mindep__ src/__init__` et.c. rpy2 is needed for loading the healthyFlowData data set from R, and it is also needed for diptest computations.

The dependecies can be obtained by following the instructions below.

### Ubuntu
```
sudo apt-get install gcc, python, openmpi  
sudo pip install numpy, cython, matplotlib, mpi4py, rpy2
```
### Mac

Python can be installed either using homebrew (http://brew.sh) or MacPorts (https://www.macports.org/). If you do not already have installed any of them, we recommend installing homebrew.

For linear algebra c libraries, you can use the BLAS and LAPACK distributions provided in the Accelerate veclib framework.

#### Install Python and openmpi

##### Using homebrew
```
brew install python, openmpi
```
##### Using MacPorts

To be written.

#### Install Python packages
```
pip install numpy, cython, matplotlib, mpi4py, rpy2
```


TODO
=====
1. Add files
2. Add documentaion 





