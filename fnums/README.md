# NUMS fortran library

`cmake` would be the way to compile the project, but alternatively we can do it as in the following example:

`mkdir build`  
`cd build`  
`gfortran -c path_to/fnums_constants.f90`  
`gfortran -c path_to/numpy.f90`  
`gfortran -c path_to/fdms/fdm1.f90`  
`...`  
`gfortran main.f90 numpy.o fdm1.o fdm2.o ...`

To clean the tree, simply remove the build directory.