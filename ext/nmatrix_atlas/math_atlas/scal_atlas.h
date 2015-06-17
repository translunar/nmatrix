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
// SciRuby is Copyright (c) 2010 - 2014, Ruby Science Foundation
// NMatrix is Copyright (c) 2012 - 2014, John Woods and the Ruby Science Foundation
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
// == scal.h
//
// BLAS scal function.
//

#ifndef SCAL_ATLAS_H
#define SCAL_ATLAS_H

#include "math/scal.h"

namespace nm { namespace math { namespace atlas {

template <typename DType>
inline void scal(const int n, const DType scalar, DType* x, const int incx) {
  //call internal implementation if no specialization below
  nm::math::scal(n, scalar, x, incx);
}

template <>
inline void scal(const int n, const float scalar, float* x, const int incx) {
  cblas_sscal(n, scalar, x, incx);
}

template <>
inline void scal(const int n, const double scalar, double* x, const int incx) {
  cblas_dscal(n, scalar, x, incx);
}

template <>
inline void scal(const int n, const Complex64 scalar, Complex64* x, const int incx) {
  cblas_cscal(n, &scalar, x, incx);
}

template <>
inline void scal(const int n, const Complex128 scalar, Complex128* x, const int incx) {
  cblas_zscal(n, &scalar, x, incx);
}

/*
 * Function signature conversion for LAPACK's scal function.
 */
template <typename DType>
inline void cblas_scal(const int n, const void* scalar, void* x, const int incx) {
  scal<DType>(n, *static_cast<const DType*>(scalar), static_cast<DType*>(x), incx);
}

}}} // end of nm::math::atlas

#endif
