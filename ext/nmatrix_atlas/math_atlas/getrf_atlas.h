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
// == getrf.h
//

/*
 *             Automatically Tuned Linear Algebra Software v3.8.4
 *                    (C) Copyright 1999 R. Clint Whaley
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *   1. Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *   2. Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions, and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *   3. The name of the ATLAS group or the names of its contributers may
 *      not be used to endorse or promote products derived from this
 *      software without specific written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE ATLAS GROUP OR ITS CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef GETRF_ATLAS_H
#define GETRF_ATLAS_H

#include "math/getrf.h"


namespace nm { namespace math { namespace atlas {

template <typename DType>
inline int getrf(const enum CBLAS_ORDER order, const int m, const int n, DType* a, const int lda, int* ipiv) {
  //unless overridden below, use the internal implementation of getrf
  //from ext/nmatrix/math/getrf.h
  return nm::math::getrf<DType>(order, m, n, a, lda, ipiv);
}

#if defined (HAVE_CLAPACK_H) || defined (HAVE_ATLAS_CLAPACK_H)
template <>
inline int getrf(const enum CBLAS_ORDER order, const int m, const int n, float* a, const int lda, int* ipiv) {
  return clapack_sgetrf(order, m, n, a, lda, ipiv);
}

template <>
inline int getrf(const enum CBLAS_ORDER order, const int m, const int n, double* a, const int lda, int* ipiv) {
  return clapack_dgetrf(order, m, n, a, lda, ipiv);
}

template <>
inline int getrf(const enum CBLAS_ORDER order, const int m, const int n, Complex64* a, const int lda, int* ipiv) {
  return clapack_cgetrf(order, m, n, reinterpret_cast<void*>(a), lda, ipiv);
}

template <>
inline int getrf(const enum CBLAS_ORDER order, const int m, const int n, Complex128* a, const int lda, int* ipiv) {
  return clapack_zgetrf(order, m, n, reinterpret_cast<void*>(a), lda, ipiv);
}
#endif

/*
* Function signature conversion for calling LAPACK's getrf functions as directly as possible.
*
* For documentation: http://www.netlib.org/lapack/double/dgetrf.f
*
* This function should normally go in math.cpp, but we need it to be available to nmatrix.cpp.
*/
template <typename DType>
inline int clapack_getrf(const enum CBLAS_ORDER order, const int m, const int n, void* a, const int lda, int* ipiv) {
  return getrf<DType>(order, m, n, reinterpret_cast<DType*>(a), lda, ipiv);
}

} } } // end nm::math::atlas

#endif // GETRF_ATLAS_H
