program velocity
    use fnums_constants, only: wp
    use fdm1
    use partials
    use numpy
    implicit none

    integer nx
    real(wp), allocatable :: x(:), xc(:, :)
    real(wp) :: hx
    integer ny
    real(wp), allocatable :: y(:), yc(:, :)
    real(wp) :: hy
    type(partial_dt) :: partial_1, partial_2

    ! Create grid
    nx = 50
    call linspace(-1.0_wp, 1.0_wp, nx, result=x)
    hx = (x(nx) - x(1))/real(nx - 1, wp)

    ny = 20
    call linspace(-1.0_wp, 1.0_wp, ny, result=y)
    hy = (y(ny) - y(1))/real(ny - 1, wp)

    ! call meshgrid(x, y, xc, yc)

    ! Create object for the partial derivatives
    call partial_1%initialize(scheme=fdm1_e121, step_x=hx, step_y=hy)
    call partial_2%initialize(scheme=fdm1_e121, step_x=hx*hx, step_y=hy*hy)

    ! to be continued

    stop
end program
