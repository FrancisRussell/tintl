#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.68])
AC_INIT([libresample], [0.1], [fpr02@doc.ic.ac.uk])
AC_CONFIG_SRCDIR([lib/interpolate.c])
AC_CONFIG_HEADERS([include/interpolate_config.h])

# Checks for programs.
AC_PROG_CC
AC_PROG_CC_STDC
AC_PROG_FC

# Checks for FFTW's cycle counter header
AC_C_INLINE
AC_HEADER_TIME
AC_CHECK_HEADERS([sys/time.h c_asm.h intrinsics.h mach/mach_time.h])

AC_SUBST(OPENMP_CFLAGS, "${OPENMP_CFLAGS}")

AC_CHECK_TYPE([hrtime_t],[AC_DEFINE(HAVE_HRTIME_T, 1, [Define to 1 if hrtime_t is defined in <sys/time.h>])],,[#if HAVE_SYS_TIME_H
#include <sys/time.h>
#endif])

AC_CHECK_FUNCS([gethrtime read_real_time time_base_to_time clock_gettime mach_absolute_time])

dnl Cray UNICOS _rtc() (real-time clock) intrinsic
AC_MSG_CHECKING([for _rtc intrinsic])
rtc_ok=yes
AC_TRY_LINK([#ifdef HAVE_INTRINSICS_H
#include <intrinsics.h>
#endif], [_rtc()], [AC_DEFINE(HAVE__RTC,1,[Define if you have the UNICOS _rtc() intrinsic.])], [rtc_ok=no])
AC_MSG_RESULT($rtc_ok)

#OpenMP Support
AX_OPENMP(AC_SUBST(HAVE_OPENMP, 1), AC_SUBST(HAVE_OPENMP, 0))

#FFTW
AC_ARG_WITH([fftw],
            AC_HELP_STRING([--with-fftw=PFX], [Use the FFTW3 installation in PFX]),
            [
             FFTW3_CPPFLAGS="-I$withval/include"
             FFTW3_OMP_CPPFLAGS="$FFTW3_CPPFLAGS"
             FFTW3_LDFLAGS="-L$withval/lib"
             FFTW3_OMP_LDFLAGS="$FFTW3_LDFLAGS"
            ]
            )

#IMKL
AC_ARG_WITH([mkl],
            AC_HELP_STRING([--with-mkl=PFX], [Use the Intel MKL installation in PFX]),
            [
             MKL_CPPFLAGS="-I$withval/include/fftw"
             MKL_LDFLAGS="-L$withval/lib/intel64"
            ]
            )

AX_CHECK_LIBRARY([FFTW3], [fftw3.h], [fftw3], 
                 AC_MSG_NOTICE([Found FFTW3 (single-threaded)]), 
                 [AC_MSG_NOTICE([Failed to find FFTW3 (single-threaded)]); unset AS_TR_SH(ac_cv_header_fftw3.h)])

AX_CHECK_LIBRARY([FFTW3_OMP], [fftw3.h], [fftw3_omp], 
                 AC_MSG_NOTICE([Found FFTW3 (OpenMP)]), 
                 [AC_MSG_NOTICE([Unable to find FFTW3 (OpenMP)]); unset AS_TR_SH(ac_cv_header_fftw3.h)])

AX_CHECK_LIBRARY([MKL], [fftw3.h], [mkl_rt], 
                 AC_MSG_NOTICE([Found Intel IMKL]), 
                 [AC_MSG_NOTICE([Unable to find Intel MKL]); unset AS_TR_SH(ac_cv_header_fftw3.h)])

if test "$ax_cv_have_MKL" == "yes"; then
  AC_MSG_NOTICE([Using Intel MKL])
  AC_SUBST(FFTW_INCLUDES, "$MKL_CPPFLAGS")
  AC_SUBST(FFTW_LDLIBS, "-lmkl_rt")
  AC_SUBST(FFTW_LDFLAGS, "$MKL_LDFLAGS")
elif test "$ax_cv_have_FFTW3_OMP" == "yes"; then
  AC_MSG_NOTICE([Using FFTW3 with OpenMP support.])
  AC_SUBST(FFTW_INCLUDES, "$FFTW3_OMP_CPPFLAGS")
  AC_SUBST(FFTW_LDLIBS, "-lfftw3_omp -lfftw3")
  AC_SUBST(FFTW_LDFLAGS, "$FFTW3_OMP_LDFLAGS")
elif test "$ax_cv_have_FFTW3" == "yes"; then
  AC_MSG_NOTICE([Using FFTW3 without threading support.])
  AC_SUBST(FFTW_INCLUDES, "$FFTW3_CPPFLAGS")
  AC_SUBST(FFTW_LDLIBS, "-lfftw3")
  AC_SUBST(FFTW_LDFLAGS, "$FFTW3_LDFLAGS")
else
  AC_MSG_ERROR([Unable to determine appropriate FFT library])
fi

AC_CONFIG_FILES([Makefile])

AC_OUTPUT
