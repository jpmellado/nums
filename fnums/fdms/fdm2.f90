! Finite-difference approximations to 2. order derivative

module fdm1
    use fnums_constants, only: wp
    implicit none
    private

    public :: fdm2_e121
    public :: fdm2_e2p

    integer i, n

contains
    subroutine fdm2_e121(f, d)
! Calculates the FD approximation to the second-order derivative in uniform grids
! using a explicit formulation with biased formulas at the boundaries.
! It is second-order in the interior points, first order at the boundaries.
! Still need to divide by the grid spacing h^2.
! Input arguments:
!    - f: values of the function at the grid points
! Output arguments:
!    - d: approximations to the derivative of the function at the grid points
        real(wp), intent(in) :: f(:)
        real(wp), intent(out) :: d(:)

        n = size(f)

        i = 1
        d(i) = f(i + 2) - 2.0_wp*f(i + 1) + f(i)
        do i = 2, n - 1
            d(i) = f(i + 1) - 2.0_wp*f(i) + f(i - 1)
        end do
        i = n
        d(i) = f(i) - 2.0_wp*f(i - 1) + f(i - 2)

        return
    end subroutine

    subroutine fdm2_e2p(f, d)
! Calculates the FD approximation to the second-order derivative in uniform grids
! using a explicit formulation with periodic boundary conditions.
! It is second-order.
! Still need to divide by the grid spacing h^2.
! Input arguments:
!    - f: values of the function at the grid points
! Output arguments:
!    - d: approximations to the derivative of the function at the grid points
        real(wp), intent(in) :: f(:)

        n = size(f)
        allocate (d, mold=f)

        i = 1
        d(i) = f(i + 1) - 2.0_wp*f(i) + f(n)
        do i = 2, n - 1
            d(i) = f(i + 1) - 2.0_wp*f(i) + f(i - 1)
        end do
        i = n
        d(i) = f(1) - 2.0_wp*f(i) + f(i - 1)

        return
    end subroutine

end module
