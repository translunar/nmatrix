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
// == trsm.h
//
// trsm function in native C++.
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

#ifndef TRSM_ATLAS_H
#define TRSM_ATLAS_H

#include "math/trsm.h"

namespace nm { namespace math { namespace atlas {

template <typename DType, typename = typename std::enable_if<!std::is_integral<DType>::value>::type>
inline void trsm(const enum CBLAS_ORDER order,
                 const enum CBLAS_SIDE side, const enum CBLAS_UPLO uplo,
                 const enum CBLAS_TRANSPOSE trans_a, const enum CBLAS_DIAG diag,
                 const int m, const int n, const DType alpha, const DType* a,
                 const int lda, DType* b, const int ldb)
{
  //call the internal implementation if no specialized version is available
  nm::math::trsm(order, side, uplo, trans_a, diag, m, n, alpha, a, lda, b, ldb);
}

template <>
inline void trsm(const enum CBLAS_ORDER order, const enum CBLAS_SIDE side, const enum CBLAS_UPLO uplo,
                 const enum CBLAS_TRANSPOSE trans_a, const enum CBLAS_DIAG diag,
                 const int m, const int n, const float alpha, const float* a,
                 const int lda, float* b, const int ldb)
{
  cblas_strsm(order, side, uplo, trans_a, diag, m, n, alpha, a, lda, b, ldb);
}

template <>
inline void trsm(const enum CBLAS_ORDER order, const enum CBLAS_SIDE side, const enum CBLAS_UPLO uplo,
                 const enum CBLAS_TRANSPOSE trans_a, const enum CBLAS_DIAG diag,
                 const int m, const int n, const double alpha, const double* a,
                 const int lda, double* b, const int ldb)
{
  cblas_dtrsm(order, side, uplo, trans_a, diag, m, n, alpha, a, lda, b, ldb);
}


template <>
inline void trsm(const enum CBLAS_ORDER order, const enum CBLAS_SIDE side, const enum CBLAS_UPLO uplo,
                 const enum CBLAS_TRANSPOSE trans_a, const enum CBLAS_DIAG diag,
                 const int m, const int n, const Complex64 alpha, const Complex64* a,
                 const int lda, Complex64* b, const int ldb)
{
  cblas_ctrsm(order, side, uplo, trans_a, diag, m, n, (const void*)(&alpha), (const void*)(a), lda, (void*)(b), ldb);
}

template <>
inline void trsm(const enum CBLAS_ORDER order, const enum CBLAS_SIDE side, const enum CBLAS_UPLO uplo,
                 const enum CBLAS_TRANSPOSE trans_a, const enum CBLAS_DIAG diag,
                 const int m, const int n, const Complex128 alpha, const Complex128* a,
                 const int lda, Complex128* b, const int ldb)
{
  cblas_ztrsm(order, side, uplo, trans_a, diag, m, n, (const void*)(&alpha), (const void*)(a), lda, (void*)(b), ldb);
}


} } }  // namespace nm::math
#endif // TRSM_ATLAS_H
