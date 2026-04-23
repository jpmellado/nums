program derivatives
    use fnums_constants, only: wp
    use fdm1
    use numpy
    implicit none

    integer n
    real(wp), allocatable :: x(:)
    real(wp), allocatable :: f(:), fp(:), fp_a(:)
    real(wp) :: one_ov_step

    !########################################################################
    ! create grid of points of the independent variable
    n = 20
    call linspace(-1.0_wp, 1.0_wp, n, result=x)
    one_ov_step = real(n - 1, wp)/(x(n) - x(1))

    ! create memory space for the function and its derivative
    allocate (f, mold=x)
    allocate (fp, mold=f)
    allocate (fp_a, mold=f)

    f(:) = exp(x(:))                                    ! function
    fp_a(:) = exp(x(:))                                 ! analytical derivative
    call fdm1_e121(f, fp)                               ! numerical derivative (unnormalized)
    fp(:) = fp(:)*one_ov_step                           ! normalized with grid step
    print *, 'Error in exponential function ', maxval(abs(fp(:) - fp_a(:)))     ! L_\infty error

    f(:) = sin(x(:))                                    ! function
    fp_a(:) = cos(x(:))                                 ! analytical derivative
    call fdm1_e121(f, fp)                               ! numerical derivative (unnormalized)
    fp(:) = fp(:)*one_ov_step                           ! normalized with grid step
    print *, 'Error sin function ', maxval(abs(fp(:) - fp_a(:)))     ! L_\infty error
    stop
end program
