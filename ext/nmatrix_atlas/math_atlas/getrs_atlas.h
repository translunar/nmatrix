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

#ifndef GETRS_ATLAS_H
#define GETRS_ATLAS_H

#include "math_atlas/inc.h"

#include "math/getrs.h"

namespace nm { namespace math { namespace atlas {

/*
 * Solves a system of linear equations A*X = B with a general NxN matrix A using the LU factorization computed by GETRF.
 *
 * From ATLAS 3.8.0.
 */
template <typename DType>
inline int getrs(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE Trans, const int N, const int NRHS, const DType* A,
           const int lda, const int* ipiv, DType* B, const int ldb)
{
  //use internal implementation unless overridden below
  return nm::math::getrs<DType>(Order, Trans, N, NRHS, A, lda, ipiv, B, ldb);
}

#if defined (HAVE_CLAPACK_H) || defined (HAVE_ATLAS_CLAPACK_H)
template <>
inline int getrs(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE Trans, const int N, const int NRHS, const float* A,
           const int lda, const int* ipiv, float* B, const int ldb)
{
  return clapack_sgetrs(Order, Trans, N, NRHS, A, lda, ipiv, B, ldb);
}

template <>
inline int getrs(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE Trans, const int N, const int NRHS, const double* A,
           const int lda, const int* ipiv, double* B, const int ldb)
{
  return clapack_dgetrs(Order, Trans, N, NRHS, A, lda, ipiv, B, ldb);
}

template <>
inline int getrs(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE Trans, const int N, const int NRHS, const Complex64* A,
           const int lda, const int* ipiv, Complex64* B, const int ldb)
{
  return clapack_cgetrs(Order, Trans, N, NRHS, reinterpret_cast<const void*>(A), lda, ipiv, reinterpret_cast<void*>(B), ldb);
}

template <>
inline int getrs(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE Trans, const int N, const int NRHS, const Complex128* A,
           const int lda, const int* ipiv, Complex128* B, const int ldb)
{
  return clapack_zgetrs(Order, Trans, N, NRHS, reinterpret_cast<const void*>(A), lda, ipiv, reinterpret_cast<void*>(B), ldb);
}
#endif

/*
* Function signature conversion for calling LAPACK's getrs functions as directly as possible.
*
* For documentation: http://www.netlib.org/lapack/double/dgetrs.f
*
* This function should normally go in math.cpp, but we need it to be available to nmatrix.cpp.
*/
template <typename DType>
inline int clapack_getrs(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE trans, const int n, const int nrhs,
                         const void* a, const int lda, const int* ipiv, void* b, const int ldb) {
  return getrs<DType>(order, trans, n, nrhs, reinterpret_cast<const DType*>(a), lda, ipiv, reinterpret_cast<DType*>(b), ldb);
}


} } } // end nm::math::atlas

#endif // GETRS_ATLAS_H
