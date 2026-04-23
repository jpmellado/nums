program derivatives
    use fnums_constants, only: wp
    use fdm1
    implicit none

    integer, parameter :: n = 20
    real(wp), allocatable :: x(:)
    real(wp), allocatable :: f(:), fp(:), fp_a(:)
    real(wp) :: one_ov_step

    !########################################################################
    ! create grid of points of the independent variable
    call linspace(-1.0_wp, 1.0_wp, n, result=x)
    one_ov_step = real(n - 1, wp)/(x(n) - x(1))

    ! create memory space for the function and its derivative
    allocate (f, mold=x)
    allocate (fp_a, mold=x)

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

contains
    !########################################################################
    !########################################################################
    subroutine linspace(start, end, num_points, result)
        real(wp), intent(in) :: start, end
        integer, intent(in) :: num_points
        real(wp), allocatable, intent(out) :: result(:)

        real(wp) step
        integer i

        allocate (result(num_points))

        step = (end - start)/real(num_points - 1, wp)
        do i = 1, num_points
            result(i) = step*real(i - 1, wp)
        end do

        return
    end subroutine

end program
