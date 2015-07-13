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
// == getrs.h
//
// getrs function in native C++.
//

#ifndef POTRS_H
#define POTRS_H

namespace nm { namespace math {

template <typename DType>
int potrs(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const int N, const int NRHS, const DType* A,
           const int lda, DType* B, const int ldb)
{
  rb_raise(rb_eNotImpError, "only CLAPACK version implemented thus far");
  return 0;
}

/*
* Function signature conversion for calling LAPACK's potrs functions as directly as possible.
*
* For documentation: http://www.netlib.org/lapack/double/dpotrs.f
*
* This function should normally go in math.cpp, but we need it to be available to nmatrix.cpp.
*/
template <typename DType>
inline int clapack_potrs(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const int n, const int nrhs,
                         const void* a, const int lda, void* b, const int ldb) {
  return potrs<DType>(order, uplo, n, nrhs, reinterpret_cast<const DType*>(a), lda, reinterpret_cast<DType*>(b), ldb);
}


} } // end nm::math

#endif // POTRS_H
