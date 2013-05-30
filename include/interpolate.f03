interface
  type(C_PTR) function plan_interpolate_3d(n0,n1,n2,in,out,flags) bind(C, name='plan_interpolate_3d')
    import
    integer(C_INT), value :: n0
    integer(C_INT), value :: n1
    integer(C_INT), value :: n2
    complex(C_DOUBLE_COMPLEX), dimension(*), intent(in) :: in
    complex(C_DOUBLE_COMPLEX), dimension(*), intent(out) :: out
    integer(C_INT), value :: flags
  end function plan_interpolate_3d

  subroutine interpolate_execute(plan,in,out) bind(C, name='interpolate_execute')
    import
    type(C_PTR), value :: plan
    complex(C_DOUBLE_COMPLEX), dimension(*), intent(in) :: in
    complex(C_DOUBLE_COMPLEX), dimension(*), intent(out) :: out
  end subroutine interpolate_execute

  type(C_PTR) function plan_interpolate_3d_split(n0,n1,n2,flags) bind(C, name='plan_interpolate_3d_split')
    import
    integer(C_INT), value :: n0
    integer(C_INT), value :: n1
    integer(C_INT), value :: n2
    integer(C_INT), value :: flags
  end function plan_interpolate_3d_split

  subroutine interpolate_execute_split(plan,rin,iin,rout,iout) bind(C, name='interpolate_execute_split')
    import
    type(C_PTR), value :: plan
    real(C_DOUBLE), dimension(*), intent(in) :: rin
    real(C_DOUBLE), dimension(*), intent(in) :: iin
    real(C_DOUBLE), dimension(*), intent(out) :: rout
    real(C_DOUBLE), dimension(*), intent(out) :: iout
  end subroutine interpolate_execute_split

  type(C_PTR) function plan_interpolate_3d_split_product(n0,n1,n2,flags) bind(C, name='plan_interpolate_3d_split_product')
    import
    integer(C_INT), value :: n0
    integer(C_INT), value :: n1
    integer(C_INT), value :: n2
    integer(C_INT), value :: flags
  end function plan_interpolate_3d_split_product

  subroutine interpolate_execute_split_product(plan,rin,iin,out) bind(C, name='interpolate_execute_split_product')
    import
    type(C_PTR), value :: plan
    real(C_DOUBLE), dimension(*), intent(in) :: rin
    real(C_DOUBLE), dimension(*), intent(in) :: iin
    real(C_DOUBLE), dimension(*), intent(out) :: out
  end subroutine interpolate_execute_split_product

  subroutine interpolate_print_timings(plan) bind(C, name='interpolate_print_timings')
    import
    type(C_PTR), value :: plan
  end subroutine interpolate_print_timings

  subroutine interpolate_destroy_plan(plan) bind(C, name='interpolate_destroy_plan')
    import
    type(C_PTR), value :: plan
  end subroutine
end interface
