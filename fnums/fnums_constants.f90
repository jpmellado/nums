module fnums_constants
    implicit none

    ! from https://fortran-lang.org/en/learn/best_practices/floating_point/
    integer, parameter :: sp = kind(1.0)
    integer, parameter :: dp = kind(1.0d0)
    ! Single precision real numbers, 6 digits, range 10⁻³⁷ to 10³⁷-1; 32 bits
    ! integer, parameter :: sp = selected_real_kind(6, 37)
    ! Double precision real numbers, 15 digits, range 10⁻³⁰⁷ to 10³⁰⁷-1; 64 bits
    ! integer, parameter :: dp = selected_real_kind(15, 307)
    integer, parameter :: wp = dp             ! working precision

    real(wp), parameter :: pi_wp = 3.14159265358979323846_wp

end module
