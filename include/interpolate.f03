interface
  type(C_PTR) function plan_interpolate_3d(n0,n1,n2,in,out) bind(C, name='plan_interpolate_3d')
    import
    integer(C_INT), value :: n0
    integer(C_INT), value :: n1
    integer(C_INT), value :: n2
    complex(C_DOUBLE_COMPLEX), dimension(*), intent(in) :: in
    complex(C_DOUBLE_COMPLEX), dimension(*), intent(out) :: out
  end function plan_interpolate_3d

  subroutine interpolate_execute(plan,in,out) bind(C, name='interpolate_execute')
    import
    type(C_PTR), value :: plan
    complex(C_DOUBLE_COMPLEX), dimension(*), intent(in) :: in
    complex(C_DOUBLE_COMPLEX), dimension(*), intent(out) :: out
  end subroutine interpolate_execute

  subroutine interpolate_print_timings(plan) bind(C, name='interpolate_print_timings')
    import
    type(C_PTR), value :: plan
  end subroutine interpolate_print_timings

  subroutine interpolate_destroy_plan(plan) bind(C, name='interpolate_destroy_plan')
    import
    type(C_PTR), value :: plan
  end subroutine
end interface
