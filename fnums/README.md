# NUMS fortran library

`cmake` would be the way to compile the project, but alternatively we can do it as in the following example:

`mkdir build`  
`cd build`  
`gfortran -c ../fnums_constants.f90`  
`gfortran -c ../numpy.f90`  
`gfortran -c ../fdms/fdm1.f90`  
`...`  
`gfortran main.f90 numpy.o fdm1.o fdm2.o ...`
