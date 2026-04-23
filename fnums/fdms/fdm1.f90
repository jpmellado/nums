! Finite-difference approximations to 1. order derivatives

module fdm1
    use fnums_constants, only: wp
    implicit none
    private

    public :: fdm1_e121
    public :: fdm1_e2p
    public :: fdm1_e11
    public :: fdm1_e1p

    integer i, n

contains
    subroutine fdm1_e121(f, d)
! Calculates the FD approximation to the first-order derivative in uniform grids
! using a explicit formulation with biased formulas at the boundaries.
! It is second-order in the interior points, first order at the boundaries.
! Still need to divide by the grid spacing h.
! Input arguments:
!    - f: values of the function at the grid points
! Output arguments:
!    - d: approximations to the derivative of the function at the grid points
        real(wp), intent(in) :: f(:)
        real(wp), allocatable, intent(out) :: d(:)

        n = size(f)
        allocate (d, mold=f)

        i = 1
        d(i) = f(i + 1) - f(i)
        do i = 2, n - 1
            d(i) = 0.5_wp*(f(i + 1) - f(i - 1))
        end do
        i = n
        d(i) = f(i) - f(i - 1)

        return
    end subroutine

    subroutine fdm1_e2p(f, d)
! Calculates the FD approximation to the first-order derivative in uniform grids
! using a explicit formulation with periodic boundary conditions.
! The last point in the array is one before the end of the periodic interval.
! It is second-order.
! Still need to divide by the grid spacing h.
! Input arguments:
!    - f: values of the function at the grid points
! Output arguments:
!    - d: approximations to the derivative of the function at the grid points
        real(wp), intent(in) :: f(:)
        real(wp), allocatable, intent(out) :: d(:)

        n = size(f)
        allocate (d, mold=f)

        i = 1
        d(i) = 0.5_wp*(f(i + 1) - f(n))
        do i = 2, n - 1
            d(i) = 0.5_wp*(f(i + 1) - f(i - 1))
        end do
        i = n
        d(i) = 0.5_wp*(f(1) - f(i - 1))

        return
    end subroutine

    subroutine fdm1_e11(f, d)
! Calculates the FD approximation to the first-order derivative in uniform grids
! using a explicit formulation with forward biased formulas
! Still need to divide by the grid spacing h.
! Input arguments:
!    - f: values of the function at the grid points
! Output arguments:
!    - d: approximations to the derivative of the function at the grid points
        real(wp), intent(in) :: f(:)
        real(wp), allocatable, intent(out) :: d(:)

        n = size(f)
        allocate (d, mold=f)

        do i = 1, n - 1
            d(i) = f(i + 1) - f(i)
        end do
        i = n
        d(i) = f(i) - f(i - 1)

        return
    end subroutine

    subroutine fdm1_e1p(f, d)
! Same, but periodic boundary conditions
! The last point in the array is one before the end of the periodic interval.
! Input arguments:
!    - f: values of the function at the grid points
! Output arguments:
!    - d: approximations to the derivative of the function at the grid points
        real(wp), intent(in) :: f(:)
        real(wp), allocatable, intent(out) :: d(:)

        n = size(f)
        allocate (d, mold=f)

        do i = 1, n - 1
            d(i) = f(i + 1) - f(i)
        end do
        i = n
        d(i) = f(1) - f(i)

        return
    end subroutine

end module
