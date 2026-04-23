module partials
    use fnums_constants, only: wp
    use fdm1
    implicit none
    private

    public :: partial_dt    ! derived type

    ! -----------------------------------------------------------------------
    type :: partial_dt
        real(wp) :: step_x
        real(wp) :: step_y
        procedure(fdm_scheme_ice), pointer, nopass :: scheme
    contains
        procedure :: initialize => initialize_partial_dt
        procedure :: dx => compute_partial_dx
        procedure :: dy => compute_partial_dy
    end type
    abstract interface
        subroutine fdm_scheme_ice(f, d)
            import wp
            real(wp), intent(in) :: f(:)
            real(wp), intent(out) :: d(:)
        end subroutine fdm_scheme_ice
    end interface

contains
    !########################################################################
    !########################################################################
    subroutine initialize_partial_dt(self, scheme, step_x, step_y)
        class(partial_dt), intent(out) :: self
        procedure(fdm_scheme_ice) :: scheme
        real(wp), optional :: step_x, step_y

        self%scheme => scheme

        if (present(step_x)) then
            self%step_x = step_x
        else
            self%step_x = 1.0_wp
        end if

        if (present(step_y)) then
            self%step_y = step_y
        else
            self%step_y = 1.0_wp
        end if

        return
    end subroutine

    subroutine compute_partial_dx(self, f, result)
        class(partial_dt), intent(in) :: self
        real(wp), intent(in) :: f(:, :)
        real(wp), allocatable, intent(out) :: result(:, :)

        integer j

        allocate (result, mold=f)
        do j = 1, size(f, 2)
            call self%scheme(f(:, j), result(:, j))
        end do
        result(:, :) = result(:, :)/self%step_x

        return
    end subroutine

    subroutine compute_partial_dy(self, f, result)
        class(partial_dt), intent(in) :: self
        real(wp), intent(in) :: f(:, :)
        real(wp), allocatable, intent(out) :: result(:, :)

        integer i

        allocate (result, mold=f)
        do i = 1, size(f, 1)
            call self%scheme(f(i, :), result(i, :))
        end do
        result(:, :) = result(:, :)/self%step_y

        return
    end subroutine

end module
