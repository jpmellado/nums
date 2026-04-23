! Mimic python numpy module

module numpy
    use fnums_constants, only: wp
    implicit none
    private

    public :: linspace
    public :: meshgrid

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

    !########################################################################
    !########################################################################
    subroutine meshgrid(x, y, x2d, y2d)
        real(wp), intent(in) :: x(:), y(:)
        real(wp), allocatable, intent(out) :: x2d(:, :), y2d(:, :)

        integer i, j

        allocate (x2d(size(x), size(y)))
        allocate (y2d(size(x), size(y)))

        do j = 1, size(y)
            x2d(:, j) = x(:)
        end do

        do i = 1, size(x)
            y2d(i, :) = y(:)
        end do

        return
    end subroutine

end module
