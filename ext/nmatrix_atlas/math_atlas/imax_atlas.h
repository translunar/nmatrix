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
// == imax.h
//
// BLAS level 1 function imax.
//

#ifndef IMAX_ATLAS_H
#define IMAX_ATLAS_H

#include "math/imax.h"

namespace nm { namespace math { namespace atlas {

template<typename DType>
inline int imax(const int n, const DType *x, const int incx) {
  //call internal implementation if no specialization below
  return nm::math::imax(n, x, incx);
}

template<>
inline int imax(const int n, const float* x, const int incx) {
  return cblas_isamax(n, x, incx);
}

template<>
inline int imax(const int n, const double* x, const int incx) {
  return cblas_idamax(n, x, incx);
}

template<>
inline int imax(const int n, const Complex64* x, const int incx) {
  return cblas_icamax(n, x, incx);
}

template <>
inline int imax(const int n, const Complex128* x, const int incx) {
  return cblas_izamax(n, x, incx);
}

template<typename DType>
inline int cblas_imax(const int n, const void* x, const int incx) {
  return imax<DType>(n, static_cast<const DType*>(x), incx);
}

}}} // end of namespace nm::math::atlas

#endif /* IMAX_ATLAS_H */
