/////////////////////////////////////////////////////////////////////
// = NMatrix
//
// A linear algebra library for scientific computation in Ruby.
// NMatrix is part of SciRuby.
//
// NMatrix was originally inspired by and derived from NArray, by
// Masahiro Tanaka: http://narray.rubyforge.org
//
// == Copyright Information
//
// SciRuby is Copyright (c) 2010 - 2013, Ruby Science Foundation
// NMatrix is Copyright (c) 2013, Ruby Science Foundation
//
// Please see LICENSE.txt for additional copyright notices.
//
// == Contributing
//
// By contributing source code to SciRuby, you agree to be bound by
// our Contributor Agreement:
//
// * https://github.com/SciRuby/sciruby/wiki/Contributor-Agreement
//
// == inc.h
//
// Includes needed for LAPACK, CLAPACK, and CBLAS functions.
//

#ifndef INC_H
# define INC_H


extern "C" { // These need to be in an extern "C" block or you'll get all kinds of undefined symbol errors.
  #include <cblas.h>

  #ifdef HAVE_CLAPACK_H
    #include <clapack.h>
  #endif
}

#endif // INC_H