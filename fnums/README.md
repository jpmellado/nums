# NUMS fortran library

`cmake` would be the way to compile the project, but we can do it as follows if you do not have/want `cmake`:

`mkdir build`  
`cd build`  
`gfortran -c ../fnums_constants.f90`  
`gfortran -c ../fdms/fdm1.f90`  
`...`

and then compile and link your main program in `build` as well.