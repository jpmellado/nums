! Mimic python numpy module

module numpy
    use fnums_constants, only: wp
    implicit none
    private

    public :: linspace

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

end module
