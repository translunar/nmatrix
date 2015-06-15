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
// == math.h
//
// Header file for math functions, interfacing with BLAS, etc.
//
// For instructions on adding CBLAS and CLAPACK functions, see the
// beginning of math.cpp.
//
// Some of these functions are from ATLAS. Here is the license for
// ATLAS:
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

#ifndef MATH_ATLAS_H
#define MATH_ATLAS_H

/*
 * Standard Includes
 */

#include "math_atlas/inc.h"
#include "math/math.h"

namespace nm {
  namespace math {
    namespace atlas {

/*
 * Types
 */


/*
 * Functions
 */


template <typename DType>
inline void syrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans, const int N,
                 const int K, const DType* alpha, const DType* A, const int lda, const DType* beta, DType* C, const int ldc) {
  rb_raise(rb_eNotImpError, "syrk not yet implemented for non-BLAS dtypes");
}

template <typename DType>
inline void herk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans, const int N,
                 const int K, const DType* alpha, const DType* A, const int lda, const DType* beta, DType* C, const int ldc) {
  rb_raise(rb_eNotImpError, "herk not yet implemented for non-BLAS dtypes");
}

template <>
inline void syrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans, const int N,
                 const int K, const float* alpha, const float* A, const int lda, const float* beta, float* C, const int ldc) {
  cblas_ssyrk(Order, Uplo, Trans, N, K, *alpha, A, lda, *beta, C, ldc);
}

template <>
inline void syrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans, const int N,
                 const int K, const double* alpha, const double* A, const int lda, const double* beta, double* C, const int ldc) {
  cblas_dsyrk(Order, Uplo, Trans, N, K, *alpha, A, lda, *beta, C, ldc);
}

template <>
inline void syrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans, const int N,
                 const int K, const Complex64* alpha, const Complex64* A, const int lda, const Complex64* beta, Complex64* C, const int ldc) {
  cblas_csyrk(Order, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc);
}

template <>
inline void syrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans, const int N,
                 const int K, const Complex128* alpha, const Complex128* A, const int lda, const Complex128* beta, Complex128* C, const int ldc) {
  cblas_zsyrk(Order, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc);
}


template <>
inline void herk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans, const int N,
                 const int K, const Complex64* alpha, const Complex64* A, const int lda, const Complex64* beta, Complex64* C, const int ldc) {
  cblas_cherk(Order, Uplo, Trans, N, K, alpha->r, A, lda, beta->r, C, ldc);
}

template <>
inline void herk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans, const int N,
                 const int K, const Complex128* alpha, const Complex128* A, const int lda, const Complex128* beta, Complex128* C, const int ldc) {
  cblas_zherk(Order, Uplo, Trans, N, K, alpha->r, A, lda, beta->r, C, ldc);
}


template <typename DType>
inline void trmm(const enum CBLAS_ORDER order, const enum CBLAS_SIDE side, const enum CBLAS_UPLO uplo,
                 const enum CBLAS_TRANSPOSE ta, const enum CBLAS_DIAG diag, const int m, const int n, const DType* alpha,
                 const DType* A, const int lda, DType* B, const int ldb) {
  rb_raise(rb_eNotImpError, "trmm not yet implemented for non-BLAS dtypes");
}

template <>
inline void trmm(const enum CBLAS_ORDER order, const enum CBLAS_SIDE side, const enum CBLAS_UPLO uplo,
                 const enum CBLAS_TRANSPOSE ta, const enum CBLAS_DIAG diag, const int m, const int n, const float* alpha,
                 const float* A, const int lda, float* B, const int ldb) {
  cblas_strmm(order, side, uplo, ta, diag, m, n, *alpha, A, lda, B, ldb);
}

template <>
inline void trmm(const enum CBLAS_ORDER order, const enum CBLAS_SIDE side, const enum CBLAS_UPLO uplo,
                 const enum CBLAS_TRANSPOSE ta, const enum CBLAS_DIAG diag, const int m, const int n, const double* alpha,
                 const double* A, const int lda, double* B, const int ldb) {
  cblas_dtrmm(order, side, uplo, ta, diag, m, n, *alpha, A, lda, B, ldb);
}

template <>
inline void trmm(const enum CBLAS_ORDER order, const enum CBLAS_SIDE side, const enum CBLAS_UPLO uplo,
                 const enum CBLAS_TRANSPOSE ta, const enum CBLAS_DIAG diag, const int m, const int n, const Complex64* alpha,
                 const Complex64* A, const int lda, Complex64* B, const int ldb) {
  cblas_ctrmm(order, side, uplo, ta, diag, m, n, alpha, A, lda, B, ldb);
}

template <>
inline void trmm(const enum CBLAS_ORDER order, const enum CBLAS_SIDE side, const enum CBLAS_UPLO uplo,
                 const enum CBLAS_TRANSPOSE ta, const enum CBLAS_DIAG diag, const int m, const int n, const Complex128* alpha,
                 const Complex128* A, const int lda, Complex128* B, const int ldb) {
  cblas_ztrmm(order, side, uplo, ta, diag, m, n, alpha, A, lda, B, ldb);
}

/*
 * From ATLAS 3.8.0:
 *
 * Computes one of two LU factorizations based on the setting of the Order
 * parameter, as follows:
 * ----------------------------------------------------------------------------
 *                       Order == CblasColMajor
 * Column-major factorization of form
 *   A = P * L * U
 * where P is a row-permutation matrix, L is lower triangular with unit
 * diagonal elements (lower trapazoidal if M > N), and U is upper triangular
 * (upper trapazoidal if M < N).
 *
 * ----------------------------------------------------------------------------
 *                       Order == CblasRowMajor
 * Row-major factorization of form
 *   A = P * L * U
 * where P is a column-permutation matrix, L is lower triangular (lower
 * trapazoidal if M > N), and U is upper triangular with unit diagonals (upper
 * trapazoidal if M < N).
 *
 * ============================================================================
 * Let IERR be the return value of the function:
 *    If IERR == 0, successful exit.
 *    If (IERR < 0) the -IERR argument had an illegal value
 *    If (IERR > 0 && Order == CblasColMajor)
 *       U(i-1,i-1) is exactly zero.  The factorization has been completed,
 *       but the factor U is exactly singular, and division by zero will
 *       occur if it is used to solve a system of equations.
 *    If (IERR > 0 && Order == CblasRowMajor)
 *       L(i-1,i-1) is exactly zero.  The factorization has been completed,
 *       but the factor L is exactly singular, and division by zero will
 *       occur if it is used to solve a system of equations.
 */
template <typename DType>
inline int potrf(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const int N, DType* A, const int lda) {
#if defined HAVE_CLAPACK_H || defined HAVE_ATLAS_CLAPACK_H
  rb_raise(rb_eNotImpError, "not yet implemented for non-BLAS dtypes");
#else
  rb_raise(rb_eNotImpError, "only CLAPACK version implemented thus far");
#endif
  return 0;
}

#if defined HAVE_CLAPACK_H || defined HAVE_ATLAS_CLAPACK_H
template <>
inline int potrf(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const int N, float* A, const int lda) {
  return clapack_spotrf(order, uplo, N, A, lda);
}

template <>
inline int potrf(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const int N, double* A, const int lda) {
  return clapack_dpotrf(order, uplo, N, A, lda);
}

template <>
inline int potrf(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const int N, Complex64* A, const int lda) {
  return clapack_cpotrf(order, uplo, N, reinterpret_cast<void*>(A), lda);
}

template <>
inline int potrf(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const int N, Complex128* A, const int lda) {
  return clapack_zpotrf(order, uplo, N, reinterpret_cast<void*>(A), lda);
}
#endif

template <bool is_complex, typename DType>
inline void lauum(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const int N, DType* A, const int lda) {

  int Nleft, Nright;
  const DType ONE = 1;
  DType *G, *U0 = A, *U1;

  if (N > 1) {
    Nleft = N >> 1;
    #ifdef NB
      if (Nleft > NB) Nleft = ATL_MulByNB(ATL_DivByNB(Nleft));
    #endif

    Nright = N - Nleft;

    // FIXME: There's a simpler way to write this next block, but I'm way too tired to work it out right now.
    if (uplo == CblasUpper) {
      if (order == CblasRowMajor) {
        G = A + Nleft;
        U1 = G + Nleft * lda;
      } else {
        G = A + Nleft * lda;
        U1 = G + Nleft;
      }
    } else {
      if (order == CblasRowMajor) {
        G = A + Nleft * lda;
        U1 = G + Nleft;
      } else {
        G = A + Nleft;
        U1 = G + Nleft * lda;
      }
    }

    lauum<is_complex, DType>(order, uplo, Nleft, U0, lda);

    if (is_complex) {

      nm::math::herk<DType>(order, uplo,
                            uplo == CblasLower ? CblasConjTrans : CblasNoTrans,
                            Nleft, Nright, &ONE, G, lda, &ONE, U0, lda);

      nm::math::trmm<DType>(order, CblasLeft, uplo, CblasConjTrans, CblasNonUnit, Nright, Nleft, &ONE, U1, lda, G, lda);
    } else {
      nm::math::syrk<DType>(order, uplo,
                            uplo == CblasLower ? CblasTrans : CblasNoTrans,
                            Nleft, Nright, &ONE, G, lda, &ONE, U0, lda);

      nm::math::trmm<DType>(order, CblasLeft, uplo, CblasTrans, CblasNonUnit, Nright, Nleft, &ONE, U1, lda, G, lda);
    }
    lauum<is_complex, DType>(order, uplo, Nright, U1, lda);

  } else {
    *A = *A * *A;
  }
}


#if defined HAVE_CLAPACK_H || defined HAVE_ATLAS_CLAPACK_H
template <bool is_complex>
inline void lauum(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const int N, float* A, const int lda) {
  clapack_slauum(order, uplo, N, A, lda);
}

template <bool is_complex>
inline void lauum(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const int N, double* A, const int lda) {
  clapack_dlauum(order, uplo, N, A, lda);
}

template <bool is_complex>
inline void lauum(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const int N, Complex64* A, const int lda) {
  clapack_clauum(order, uplo, N, A, lda);
}

template <bool is_complex>
inline void lauum(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const int N, Complex128* A, const int lda) {
  clapack_zlauum(order, uplo, N, A, lda);
}
#endif


/*
* Function signature conversion for calling LAPACK's lauum functions as directly as possible.
*
* For documentation: http://www.netlib.org/lapack/double/dlauum.f
*
* This function should normally go in math.cpp, but we need it to be available to nmatrix.cpp.
*/
template <bool is_complex, typename DType>
inline int clapack_lauum(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const int n, void* a, const int lda) {
  if (n < 0)              rb_raise(rb_eArgError, "n cannot be less than zero, is set to %d", n);
  if (lda < n || lda < 1) rb_raise(rb_eArgError, "lda must be >= max(n,1); lda=%d, n=%d\n", lda, n);

  if (uplo == CblasUpper) lauum<is_complex, DType>(order, uplo, n, reinterpret_cast<DType*>(a), lda);
  else                    lauum<is_complex, DType>(order, uplo, n, reinterpret_cast<DType*>(a), lda);

  return 0;
}




/*
 * Macro for declaring LAPACK specializations of the getrf function.
 *
 * type is the DType; call is the specific function to call; cast_as is what the DType* should be
 * cast to in order to pass it to LAPACK.
 */
#define LAPACK_GETRF(type, call, cast_as)                                     \
template <>                                                                   \
inline int getrf(const enum CBLAS_ORDER Order, const int M, const int N, type * A, const int lda, int* ipiv) { \
  int info = call(Order, M, N, reinterpret_cast<cast_as *>(A), lda, ipiv);    \
  if (!info) return info;                                                     \
  else {                                                                      \
    rb_raise(rb_eArgError, "getrf: problem with argument %d\n", info);        \
    return info;                                                              \
  }                                                                           \
}

/* Specialize for ATLAS types */
/*LAPACK_GETRF(float,      clapack_sgetrf, float)
LAPACK_GETRF(double,     clapack_dgetrf, double)
LAPACK_GETRF(Complex64,  clapack_cgetrf, void)
LAPACK_GETRF(Complex128, clapack_zgetrf, void)
*/



/*
* Function signature conversion for calling LAPACK's potrf functions as directly as possible.
*
* For documentation: http://www.netlib.org/lapack/double/dpotrf.f
*
* This function should normally go in math.cpp, but we need it to be available to nmatrix.cpp.
*/
template <typename DType>
inline int clapack_potrf(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const int n, void* a, const int lda) {
  return potrf<DType>(order, uplo, n, reinterpret_cast<DType*>(a), lda);
}



template <typename DType>
inline int potri(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const int n, DType* a, const int lda) {
  rb_raise(rb_eNotImpError, "potri not yet implemented for non-BLAS dtypes");
  return 0;
}


#if defined HAVE_CLAPACK_H || defined HAVE_ATLAS_CLAPACK_H
template <>
inline int potri(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const int n, float* a, const int lda) {
  return clapack_spotri(order, uplo, n, a, lda);
}

template <>
inline int potri(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const int n, double* a, const int lda) {
  return clapack_dpotri(order, uplo, n, a, lda);
}

template <>
inline int potri(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const int n, Complex64* a, const int lda) {
  return clapack_cpotri(order, uplo, n, reinterpret_cast<void*>(a), lda);
}

template <>
inline int potri(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const int n, Complex128* a, const int lda) {
  return clapack_zpotri(order, uplo, n, reinterpret_cast<void*>(a), lda);
}
#endif


/*
 * Function signature conversion for calling LAPACK's potri functions as directly as possible.
 *
 * For documentation: http://www.netlib.org/lapack/double/dpotri.f
 *
 * This function should normally go in math.cpp, but we need it to be available to nmatrix.cpp.
 */
template <typename DType>
inline int clapack_potri(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const int n, void* a, const int lda) {
  return potri<DType>(order, uplo, n, reinterpret_cast<DType*>(a), lda);
}


}}} // end namespace nm::math::atlas

#endif // MATH_ATLAS_H
