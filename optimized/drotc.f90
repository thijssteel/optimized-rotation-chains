subroutine drotc(side, dir, startup, shutdown, m, n, k,&
    A, lda, C, ldc, S, lds)
    use iso_c_binding, only: c_char, c_bool
    implicit none
!    .. Scalar Arguments ..
    integer, intent(in) :: m, n, k, lda, ldc, lds
    character, intent(in) :: dir, side
    logical, intent(in) :: startup, shutdown
!    .. Array Arguments ..
    double precision, intent(inout) :: A(lda,*)
    double precision, intent(in) :: C(ldc,*), S(lds,*)

    interface
        subroutine drotc_c(side, dir, startup, shutdown, m, n, k,&
            A, lda, C, ldc, S, lds) bind(c, name="drotc")
            use iso_c_binding, only: c_char, c_double, c_int, c_bool
            implicit none
            integer(c_int), intent(in) :: m, n, k, lda, ldc, lds
            character(c_char), intent(in) :: dir, side
            logical(c_bool), intent(in) :: startup, shutdown
            real(c_double), intent(inout) :: A(lda,*)
            real(c_double), intent(in) :: C(ldc,*), S(lds,*)


        end subroutine drotc_c
    end interface

    character(c_char) :: side_to_c, dir_to_c
    logical(c_bool) :: startup_to_c, shutdown_to_c

    ! Convert characters to C characters
    side_to_c = side
    dir_to_c = dir
    startup_to_c = startup
    shutdown_to_c = shutdown

    ! Call C function
    call drotc_c(side_to_c, dir_to_c, startup_to_c,&
     shutdown_to_c, m, n, k, A, lda, C, ldc, S, lds)

end subroutine drotc
