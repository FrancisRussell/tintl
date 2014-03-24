# SYNOPSIS
#
#   AX_WITH_LIBRARY(NAME, HEADER, LIBNAME, ["optional"|"required"])
#
# DESCRIPTION 
#
#   Checks for the supplied library name and header and sets environment
#   variables and calls AC_SUBST for $NAME_LDFLAGS, $NAME_LDLIBS and
#   $NAME_CFLAGS if found.
#
# LICENSE
#
#   Copyright (c) 2013 Francis Russell
#   <francis@unchartedbackwaters.co.uk>
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful, but
#   WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
#   General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program. If not, see <http://www.gnu.org/licenses/>.
#
#   As a special exception, the respective Autoconf Macro's copyright
#   owner gives unlimited permission to copy, distribute and modify the
#   configure scripts that are the output of Autoconf when processing
#   the Macro. You need not follow the terms of the GNU General Public
#   License when using or distributing such scripts, even though
#   portions of the text of the Macro appear in them. The GNU General
#   Public License (GPL) does govern all other use of the material that
#   constitutes the Autoconf Macro.
#
#   This special exception to the GPL applies to versions of the
#   Autoconf Macro released by the Autoconf Archive. When you make and
#   distribute a modified version of the Autoconf Macro, you may extend
#   this special exception to the GPL to apply to your modified version
#   as well.


AC_DEFUN([AX_WITH_LIBRARY], 
[
  AC_ARG_WITH([$1], [AS_HELP_STRING([--with-$1=PFX], [Use installation of $1 in PFX])], [location="$withval"])
  saved_CFLAGS="$CFLAGS"
  saved_CPPFLAGS="$CPPFLAGS"
  saved_LDFLAGS="$LDFLAGS"

  AS_IF([test "$4" = "optional"], [optional="yes"],
        [test "$4" = "required"], [optional=""],
        [test -z "$4"],           [optional=""],
        [AC_MSG_ERROR([Unrecognised value ["]$4["], should be "optional", "required" or unspecified.])])

  include_directives="-I/usr/include/$1 -I/usr/local/include/$1"

  AS_IF([test -n "$location"],[
    include_directives="$include_directives -I$location/include"
    LDFLAGS="$LDFLAGS -L$location/lib"
    ])

  CFLAGS="$CFLAGS $include_directives"
  CPPFLAGS="$CPPFLAGS $include_directives"

  [HAVE_]AS_TR_SH(AS_TR_CPP($1))=0

  AC_CHECK_HEADER($2, [
    AC_CHECK_LIB($3, [main], [
      AS_IF([test -n "$location"], [
        AS_TR_SH(AS_TR_CPP($1))_LDFLAGS="-L$location/lib"
        AS_TR_SH(AS_TR_CPP($1))_LDLIBS="-l$3"
      ], [
        AS_TR_SH(AS_TR_CPP($1))_LDFLAGS=""
        AS_TR_SH(AS_TR_CPP($1))_LDLIBS="-l$3"
      ])
      AS_TR_SH(AS_TR_CPP($1))_CFLAGS="$include_directives"
      AC_SUBST(AS_TR_SH(AS_TR_CPP($1))_CFLAGS)
      AC_SUBST(AS_TR_SH(AS_TR_CPP($1))_LDFLAGS)
      AC_SUBST(AS_TR_SH(AS_TR_CPP($1))_LDLIBS)
      [HAVE_]AS_TR_SH(AS_TR_CPP($1))=1
      AC_DEFINE([HAVE_]AS_TR_SH(AS_TR_CPP($1)), 1, [Defined if $1 is available.])
      AC_MSG_NOTICE([Found $1.])
      ],
      AS_IF([test -z "$optional"], 
       AC_MSG_ERROR([Unable to find library $3.]), 
       AC_MSG_NOTICE([Unable to find library $3 so compiling without $1 support.])
       ))
  ],
    AS_IF([test -z "$optional"], 
       AC_MSG_ERROR([Unable to find header $2.]), 
       AC_MSG_NOTICE([Unable to find header $2 so compiling without $1 support.])
       )
  )

  AC_SUBST([HAVE_]AS_TR_SH(AS_TR_CPP($1)))

  CFLAGS="$saved_CFLAGS"
  CPPFLAGS="$saved_CPPFLAGS"
  LDFLAGS="$saved_LDFLAGS"
])
