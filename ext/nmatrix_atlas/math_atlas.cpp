//Following structure of ext/nmatrix/math.cpp
/*
 * Project Includes
 */

#include "data/data.h"

#include "math_atlas/inc.h"

#include "math/util.h"

//BLAS
#include "math_atlas/asum_atlas.h"
#include "math_atlas/trsm_atlas.h"
#include "math/long_dtype.h" //this should really be in gemv.h, gemm.h
#include "math_atlas/gemv_atlas.h"
#include "math_atlas/gemm_atlas.h"
#include "math_atlas/imax_atlas.h"
#include "math_atlas/nrm2_atlas.h"
#include "math_atlas/rot_atlas.h"
#include "math_atlas/rotg_atlas.h"
#include "math_atlas/scal_atlas.h"

//LAPACK
#include "math/laswp.h"
#include "math_atlas/getrf_atlas.h"
#include "math_atlas/getri_atlas.h"
#include "math_atlas/getrs_atlas.h"
#include "math_atlas/potrs_atlas.h"

#include "math_atlas/gesvd.h"
#include "math_atlas/gesdd.h"
#include "math_atlas/geev.h"

#include "math_atlas/math_atlas.h"

/*
 * Forward Declarations
 */

extern "C" {
  /* BLAS Level 3. */
  static VALUE nm_cblas_trsm(VALUE self, VALUE order, VALUE side, VALUE uplo, VALUE trans_a, VALUE diag, VALUE m, VALUE n,
                             VALUE vAlpha, VALUE a, VALUE lda, VALUE b, VALUE ldb);
  static VALUE nm_cblas_trmm(VALUE self, VALUE order, VALUE side, VALUE uplo, VALUE trans_a, VALUE diag, VALUE m, VALUE n,
                             VALUE alpha, VALUE a, VALUE lda, VALUE b, VALUE ldb);
  static VALUE nm_cblas_herk(VALUE self, VALUE order, VALUE uplo, VALUE trans, VALUE n, VALUE k, VALUE alpha, VALUE a,
                             VALUE lda, VALUE beta, VALUE c, VALUE ldc);
  static VALUE nm_cblas_syrk(VALUE self, VALUE order, VALUE uplo, VALUE trans, VALUE n, VALUE k, VALUE alpha, VALUE a,
                             VALUE lda, VALUE beta, VALUE c, VALUE ldc);

  /* LAPACK. */
  static VALUE nm_atlas_clapack_getrf(VALUE self, VALUE order, VALUE m, VALUE n, VALUE a, VALUE lda);
  static VALUE nm_clapack_potrf(VALUE self, VALUE order, VALUE uplo, VALUE n, VALUE a, VALUE lda);
  static VALUE nm_clapack_getrs(VALUE self, VALUE order, VALUE trans, VALUE n, VALUE nrhs, VALUE a, VALUE lda, VALUE ipiv, VALUE b, VALUE ldb);
  static VALUE nm_clapack_potrs(VALUE self, VALUE order, VALUE uplo, VALUE n, VALUE nrhs, VALUE a, VALUE lda, VALUE b, VALUE ldb);
  static VALUE nm_atlas_clapack_getri(VALUE self, VALUE order, VALUE n, VALUE a, VALUE lda, VALUE ipiv);
  static VALUE nm_clapack_potri(VALUE self, VALUE order, VALUE uplo, VALUE n, VALUE a, VALUE lda);
  static VALUE nm_clapack_laswp(VALUE self, VALUE n, VALUE a, VALUE lda, VALUE k1, VALUE k2, VALUE ipiv, VALUE incx);
  static VALUE nm_clapack_lauum(VALUE self, VALUE order, VALUE uplo, VALUE n, VALUE a, VALUE lda);

  static VALUE nm_lapack_gesvd(VALUE self, VALUE jobu, VALUE jobvt, VALUE m, VALUE n, VALUE a, VALUE lda, VALUE s, VALUE u, VALUE ldu, VALUE vt, VALUE ldvt, VALUE lworkspace_size);
  static VALUE nm_lapack_gesdd(VALUE self, VALUE jobz, VALUE m, VALUE n, VALUE a, VALUE lda, VALUE s, VALUE u, VALUE ldu, VALUE vt, VALUE ldvt, VALUE lworkspace_size);
  static VALUE nm_lapack_geev(VALUE self, VALUE compute_left, VALUE compute_right, VALUE n, VALUE a, VALUE lda, VALUE w, VALUE wi, VALUE vl, VALUE ldvl, VALUE vr, VALUE ldvr, VALUE lwork);
}

////////////////////
// Math Functions //
////////////////////

namespace nm { 
  namespace math {

    /*
     * Function signature conversion for calling CBLAS' trsm functions as directly as possible.
     *
     * For documentation: http://www.netlib.org/blas/dtrsm.f
     */
    template <typename DType>
    inline static void cblas_trsm(const enum CBLAS_ORDER order, const enum CBLAS_SIDE side, const enum CBLAS_UPLO uplo,
                                   const enum CBLAS_TRANSPOSE trans_a, const enum CBLAS_DIAG diag,
                                   const int m, const int n, const void* alpha, const void* a,
                                   const int lda, void* b, const int ldb)
    {
      trsm<DType>(order, side, uplo, trans_a, diag, m, n, *reinterpret_cast<const DType*>(alpha),
                  reinterpret_cast<const DType*>(a), lda, reinterpret_cast<DType*>(b), ldb);
    }


    /*
     * Function signature conversion for calling CBLAS' trmm functions as directly as possible.
     *
     * For documentation: http://www.netlib.org/blas/dtrmm.f
     */
    template <typename DType>
    inline static void cblas_trmm(const enum CBLAS_ORDER order, const enum CBLAS_SIDE side, const enum CBLAS_UPLO uplo,
                                  const enum CBLAS_TRANSPOSE ta, const enum CBLAS_DIAG diag, const int m, const int n, const void* alpha,
                                  const void* A, const int lda, void* B, const int ldb)
    {
      trmm<DType>(order, side, uplo, ta, diag, m, n, reinterpret_cast<const DType*>(alpha),
                  reinterpret_cast<const DType*>(A), lda, reinterpret_cast<DType*>(B), ldb);
    }


    /*
     * Function signature conversion for calling CBLAS' syrk functions as directly as possible.
     *
     * For documentation: http://www.netlib.org/blas/dsyrk.f
     */
    template <typename DType>
    inline static void cblas_syrk(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans,
                                  const int n, const int k, const void* alpha,
                                  const void* A, const int lda, const void* beta, void* C, const int ldc)
    {
      syrk<DType>(order, uplo, trans, n, k, reinterpret_cast<const DType*>(alpha),
                  reinterpret_cast<const DType*>(A), lda, reinterpret_cast<const DType*>(beta), reinterpret_cast<DType*>(C), ldc);
    }

    /*
     * Function signature conversion for calling CBLAS' gesvd functions as directly as possible.
     */
    template <typename DType, typename CType>
    inline static int lapack_gesvd(char jobu, char jobvt, int m, int n, void* a, int lda, void* s, void* u, int ldu, void* vt, int ldvt, void* work, int lwork, void* rwork) {
      return gesvd<DType,CType>(jobu, jobvt, m, n, reinterpret_cast<DType*>(a), lda, reinterpret_cast<DType*>(s), reinterpret_cast<DType*>(u), ldu, reinterpret_cast<DType*>(vt), ldvt, reinterpret_cast<DType*>(work), lwork, reinterpret_cast<CType*>(rwork));
    }

    /*
     * Function signature conversion for calling CBLAS' gesdd functions as directly as possible.
     */
    template <typename DType, typename CType>
    inline static int lapack_gesdd(char jobz, int m, int n, void* a, int lda, void* s, void* u, int ldu, void* vt, int ldvt, void* work, int lwork, int* iwork, void* rwork) {
      return gesdd<DType,CType>(jobz, m, n, reinterpret_cast<DType*>(a), lda, reinterpret_cast<DType*>(s), reinterpret_cast<DType*>(u), ldu, reinterpret_cast<DType*>(vt), ldvt, reinterpret_cast<DType*>(work), lwork, iwork, reinterpret_cast<CType*>(rwork));
    }

  }
}

extern "C" {

///////////////////
// Ruby Bindings //
///////////////////

void nm_math_init_atlas() {

  /* ATLAS-CLAPACK Functions */
  rb_define_singleton_method(cNMatrix_LAPACK, "clapack_getrf", (METHOD)nm_atlas_clapack_getrf, 5);
//  rb_define_singleton_method(cNMatrix_LAPACK, "clapack_potrf", (METHOD)nm_clapack_potrf, 5);
//  rb_define_singleton_method(cNMatrix_LAPACK, "clapack_getrs", (METHOD)nm_clapack_getrs, 9);
//  rb_define_singleton_method(cNMatrix_LAPACK, "clapack_potrs", (METHOD)nm_clapack_potrs, 8);
  rb_define_singleton_method(cNMatrix_LAPACK, "clapack_getri", (METHOD)nm_atlas_clapack_getri, 5);
//  rb_define_singleton_method(cNMatrix_LAPACK, "clapack_potri", (METHOD)nm_clapack_potri, 5);
//  rb_define_singleton_method(cNMatrix_LAPACK, "clapack_laswp", (METHOD)nm_clapack_laswp, 7);
//  rb_define_singleton_method(cNMatrix_LAPACK, "clapack_lauum", (METHOD)nm_clapack_lauum, 5);
//
//  /* Non-ATLAS regular LAPACK Functions called via Fortran interface */
//  rb_define_singleton_method(cNMatrix_LAPACK, "lapack_gesvd", (METHOD)nm_lapack_gesvd, 12);
//  rb_define_singleton_method(cNMatrix_LAPACK, "lapack_gesdd", (METHOD)nm_lapack_gesdd, 11);
//  rb_define_singleton_method(cNMatrix_LAPACK, "lapack_geev",  (METHOD)nm_lapack_geev,  12);
//
//  rb_define_singleton_method(cNMatrix_BLAS, "cblas_trsm", (METHOD)nm_cblas_trsm, 12);
//  rb_define_singleton_method(cNMatrix_BLAS, "cblas_trmm", (METHOD)nm_cblas_trmm, 12);
//  rb_define_singleton_method(cNMatrix_BLAS, "cblas_syrk", (METHOD)nm_cblas_syrk, 11);
//  rb_define_singleton_method(cNMatrix_BLAS, "cblas_herk", (METHOD)nm_cblas_herk, 11);

}

static VALUE nm_cblas_trsm(VALUE self,
                           VALUE order,
                           VALUE side, VALUE uplo,
                           VALUE trans_a, VALUE diag,
                           VALUE m, VALUE n,
                           VALUE alpha,
                           VALUE a, VALUE lda,
                           VALUE b, VALUE ldb)
{
  static void (*ttable[nm::NUM_DTYPES])(const enum CBLAS_ORDER, const enum CBLAS_SIDE, const enum CBLAS_UPLO,
                                        const enum CBLAS_TRANSPOSE, const enum CBLAS_DIAG,
                                        const int m, const int n, const void* alpha, const void* a,
                                        const int lda, void* b, const int ldb) = {
      NULL, NULL, NULL, NULL, NULL, // integers not allowed due to division
      nm::math::cblas_trsm<float>,
      nm::math::cblas_trsm<double>,
      cblas_ctrsm, cblas_ztrsm, // call directly, same function signature!
      nm::math::cblas_trsm<nm::Rational32>,
      nm::math::cblas_trsm<nm::Rational64>,
      nm::math::cblas_trsm<nm::Rational128>,
      nm::math::cblas_trsm<nm::RubyObject>
  };

  nm::dtype_t dtype = NM_DTYPE(a);

  if (!ttable[dtype]) {
    rb_raise(nm_eDataTypeError, "this matrix operation undefined for integer matrices");
  } else {
    void *pAlpha = NM_ALLOCA_N(char, DTYPE_SIZES[dtype]);
    rubyval_to_cval(alpha, dtype, pAlpha);

    ttable[dtype](blas_order_sym(order), blas_side_sym(side), blas_uplo_sym(uplo), blas_transpose_sym(trans_a), blas_diag_sym(diag), FIX2INT(m), FIX2INT(n), pAlpha, NM_STORAGE_DENSE(a)->elements, FIX2INT(lda), NM_STORAGE_DENSE(b)->elements, FIX2INT(ldb));
  }

  return Qtrue;
}

static VALUE nm_cblas_trmm(VALUE self,
                           VALUE order,
                           VALUE side, VALUE uplo,
                           VALUE trans_a, VALUE diag,
                           VALUE m, VALUE n,
                           VALUE alpha,
                           VALUE a, VALUE lda,
                           VALUE b, VALUE ldb)
{
  static void (*ttable[nm::NUM_DTYPES])(const enum CBLAS_ORDER,
                                        const enum CBLAS_SIDE, const enum CBLAS_UPLO,
                                        const enum CBLAS_TRANSPOSE, const enum CBLAS_DIAG,
                                        const int m, const int n, const void* alpha, const void* a,
                                        const int lda, void* b, const int ldb) = {
      NULL, NULL, NULL, NULL, NULL, // integers not allowed due to division
      nm::math::cblas_trmm<float>,
      nm::math::cblas_trmm<double>,
      cblas_ctrmm, cblas_ztrmm, // call directly, same function signature!
      NULL, NULL, NULL, NULL
      /*
      nm::math::cblas_trmm<nm::Rational32>,
      nm::math::cblas_trmm<nm::Rational64>,
      nm::math::cblas_trmm<nm::Rational128>,
      nm::math::cblas_trmm<nm::RubyObject>*/
  };

  nm::dtype_t dtype = NM_DTYPE(a);

  if (!ttable[dtype]) {
    rb_raise(nm_eDataTypeError, "this matrix operation not yet defined for non-BLAS dtypes");
  } else {
    void *pAlpha = NM_ALLOCA_N(char, DTYPE_SIZES[dtype]);
    rubyval_to_cval(alpha, dtype, pAlpha);

    ttable[dtype](blas_order_sym(order), blas_side_sym(side), blas_uplo_sym(uplo), blas_transpose_sym(trans_a), blas_diag_sym(diag), FIX2INT(m), FIX2INT(n), pAlpha, NM_STORAGE_DENSE(a)->elements, FIX2INT(lda), NM_STORAGE_DENSE(b)->elements, FIX2INT(ldb));
  }

  return b;
}

static VALUE nm_cblas_syrk(VALUE self,
                           VALUE order,
                           VALUE uplo,
                           VALUE trans,
                           VALUE n, VALUE k,
                           VALUE alpha,
                           VALUE a, VALUE lda,
                           VALUE beta,
                           VALUE c, VALUE ldc)
{
  static void (*ttable[nm::NUM_DTYPES])(const enum CBLAS_ORDER, const enum CBLAS_UPLO, const enum CBLAS_TRANSPOSE,
                                        const int n, const int k, const void* alpha, const void* a,
                                        const int lda, const void* beta, void* c, const int ldc) = {
      NULL, NULL, NULL, NULL, NULL, // integers not allowed due to division
      nm::math::cblas_syrk<float>,
      nm::math::cblas_syrk<double>,
      cblas_csyrk, cblas_zsyrk, // call directly, same function signature!
      NULL, NULL, NULL, NULL
      /*nm::math::cblas_trsm<nm::Rational32>,
      nm::math::cblas_trsm<nm::Rational64>,
      nm::math::cblas_trsm<nm::Rational128>,
      nm::math::cblas_trsm<nm::RubyObject>*/
  };

  nm::dtype_t dtype = NM_DTYPE(a);

  if (!ttable[dtype]) {
    rb_raise(nm_eDataTypeError, "this matrix operation undefined for integer matrices");
  } else {
    void *pAlpha = NM_ALLOCA_N(char, DTYPE_SIZES[dtype]),
         *pBeta = NM_ALLOCA_N(char, DTYPE_SIZES[dtype]);
    rubyval_to_cval(alpha, dtype, pAlpha);
    rubyval_to_cval(beta, dtype, pBeta);

    ttable[dtype](blas_order_sym(order), blas_uplo_sym(uplo), blas_transpose_sym(trans), FIX2INT(n), FIX2INT(k), pAlpha, NM_STORAGE_DENSE(a)->elements, FIX2INT(lda), pBeta, NM_STORAGE_DENSE(c)->elements, FIX2INT(ldc));
  }

  return Qtrue;
}

static VALUE nm_cblas_herk(VALUE self,
                           VALUE order,
                           VALUE uplo,
                           VALUE trans,
                           VALUE n, VALUE k,
                           VALUE alpha,
                           VALUE a, VALUE lda,
                           VALUE beta,
                           VALUE c, VALUE ldc)
{

  nm::dtype_t dtype = NM_DTYPE(a);

  if (dtype == nm::COMPLEX64) {
    cblas_cherk(blas_order_sym(order), blas_uplo_sym(uplo), blas_transpose_sym(trans), FIX2INT(n), FIX2INT(k), NUM2DBL(alpha), NM_STORAGE_DENSE(a)->elements, FIX2INT(lda), NUM2DBL(beta), NM_STORAGE_DENSE(c)->elements, FIX2INT(ldc));
  } else if (dtype == nm::COMPLEX128) {
    cblas_zherk(blas_order_sym(order), blas_uplo_sym(uplo), blas_transpose_sym(trans), FIX2INT(n), FIX2INT(k), NUM2DBL(alpha), NM_STORAGE_DENSE(a)->elements, FIX2INT(lda), NUM2DBL(beta), NM_STORAGE_DENSE(c)->elements, FIX2INT(ldc));
  } else
    rb_raise(rb_eNotImpError, "this matrix operation undefined for non-complex dtypes");
  return Qtrue;
}

/*
 * Function signature conversion for calling CBLAS' gesvd functions as directly as possible.
 *
 * xGESVD computes the singular value decomposition (SVD) of a real
 * M-by-N matrix A, optionally computing the left and/or right singular
 * vectors. The SVD is written
 *
 *      A = U * SIGMA * transpose(V)
 *
 * where SIGMA is an M-by-N matrix which is zero except for its
 * min(m,n) diagonal elements, U is an M-by-M orthogonal matrix, and
 * V is an N-by-N orthogonal matrix.  The diagonal elements of SIGMA
 * are the singular values of A; they are real and non-negative, and
 * are returned in descending order.  The first min(m,n) columns of
 * U and V are the left and right singular vectors of A.
 *
 * Note that the routine returns V**T, not V.
 */

static VALUE nm_lapack_gesvd(VALUE self, VALUE jobu, VALUE jobvt, VALUE m, VALUE n, VALUE a, VALUE lda, VALUE s, VALUE u, VALUE ldu, VALUE vt, VALUE ldvt, VALUE lwork) {
  static int (*gesvd_table[nm::NUM_DTYPES])(char, char, int, int, void* a, int, void* s, void* u, int, void* vt, int, void* work, int, void* rwork) = {
    NULL, NULL, NULL, NULL, NULL, // no integer ops
    nm::math::lapack_gesvd<float,float>,
    nm::math::lapack_gesvd<double,double>,
    nm::math::lapack_gesvd<nm::Complex64,float>,
    nm::math::lapack_gesvd<nm::Complex128,double>,
    NULL, NULL, NULL, NULL // no rationals or Ruby objects
  };

  nm::dtype_t dtype = NM_DTYPE(a);


  if (!gesvd_table[dtype]) {
    rb_raise(rb_eNotImpError, "this operation not yet implemented for non-BLAS dtypes");
    return Qfalse;
  } else {
    int M = FIX2INT(m),
        N = FIX2INT(n);

    int min_mn  = NM_MIN(M,N);
    int max_mn  = NM_MAX(M,N);

    char JOBU = lapack_svd_job_sym(jobu),
         JOBVT = lapack_svd_job_sym(jobvt);

    // only need rwork for complex matrices
    int rwork_size  = (dtype == nm::COMPLEX64 || dtype == nm::COMPLEX128) ? 5 * min_mn : 0;
    void* rwork     = rwork_size > 0 ? NM_ALLOCA_N(char, DTYPE_SIZES[dtype] * rwork_size) : NULL;
    int work_size   = FIX2INT(lwork);

    // ignore user argument for lwork if it's too small.
    work_size       = NM_MAX((dtype == nm::COMPLEX64 || dtype == nm::COMPLEX128 ? 2 * min_mn + max_mn : NM_MAX(3*min_mn + max_mn, 5*min_mn)), work_size);
    void* work      = NM_ALLOCA_N(char, DTYPE_SIZES[dtype] * work_size);

    int info = gesvd_table[dtype](JOBU, JOBVT, M, N, NM_STORAGE_DENSE(a)->elements, FIX2INT(lda),
      NM_STORAGE_DENSE(s)->elements, NM_STORAGE_DENSE(u)->elements, FIX2INT(ldu), NM_STORAGE_DENSE(vt)->elements, FIX2INT(ldvt),
      work, work_size, rwork);
    return INT2FIX(info);
  }
}

/*
 * Function signature conversion for calling CBLAS' gesdd functions as directly as possible.
 *
 * xGESDD uses a divide-and-conquer strategy to compute the singular value decomposition (SVD) of a real
 * M-by-N matrix A, optionally computing the left and/or right singular
 * vectors. The SVD is written
 *
 *      A = U * SIGMA * transpose(V)
 *
 * where SIGMA is an M-by-N matrix which is zero except for its
 * min(m,n) diagonal elements, U is an M-by-M orthogonal matrix, and
 * V is an N-by-N orthogonal matrix.  The diagonal elements of SIGMA
 * are the singular values of A; they are real and non-negative, and
 * are returned in descending order.  The first min(m,n) columns of
 * U and V are the left and right singular vectors of A.
 *
 * Note that the routine returns V**T, not V.
 */
static VALUE nm_lapack_gesdd(VALUE self, VALUE jobz, VALUE m, VALUE n, VALUE a, VALUE lda, VALUE s, VALUE u, VALUE ldu, VALUE vt, VALUE ldvt, VALUE lwork) {
  static int (*gesdd_table[nm::NUM_DTYPES])(char, int, int, void* a, int, void* s, void* u, int, void* vt, int, void* work, int, int* iwork, void* rwork) = {
    NULL, NULL, NULL, NULL, NULL, // no integer ops
    nm::math::lapack_gesdd<float,float>,
    nm::math::lapack_gesdd<double,double>,
    nm::math::lapack_gesdd<nm::Complex64,float>,
    nm::math::lapack_gesdd<nm::Complex128,double>,
    NULL, NULL, NULL, NULL // no rationals or Ruby objects
  };

  nm::dtype_t dtype = NM_DTYPE(a);

  if (!gesdd_table[dtype]) {
    rb_raise(rb_eNotImpError, "this operation not yet implemented for non-BLAS dtypes");
    return Qfalse;
  } else {
    int M = FIX2INT(m),
        N = FIX2INT(n);

    int min_mn  = NM_MIN(M,N);
    int max_mn  = NM_MAX(M,N);

    char JOBZ = lapack_svd_job_sym(jobz);

    // only need rwork for complex matrices
    void* rwork = NULL;

    int work_size = FIX2INT(lwork); // Make sure we allocate enough work, regardless of the user request.
    if (dtype == nm::COMPLEX64 || dtype == nm::COMPLEX128) {
      int rwork_size = min_mn * (JOBZ == 'N' ? 5 : NM_MAX(5*min_mn + 7, 2*max_mn + 2*min_mn + 1));
      rwork = NM_ALLOCA_N(char, DTYPE_SIZES[dtype] * rwork_size);

      if (JOBZ == 'N')      work_size = NM_MAX(work_size, 3*min_mn + NM_MAX(max_mn, 6*min_mn));
      else if (JOBZ == 'O') work_size = NM_MAX(work_size, 3*min_mn*min_mn + NM_MAX(max_mn, 5*min_mn*min_mn + 4*min_mn));
      else                  work_size = NM_MAX(work_size, 3*min_mn*min_mn + NM_MAX(max_mn, 4*min_mn*min_mn + 4*min_mn));
    } else {
      if (JOBZ == 'N')      work_size = NM_MAX(work_size, 2*min_mn + max_mn);
      else if (JOBZ == 'O') work_size = NM_MAX(work_size, 2*min_mn*min_mn + max_mn + 2*min_mn);
      else                  work_size = NM_MAX(work_size, min_mn*min_mn + max_mn + 2*min_mn);
    }
    void* work  = NM_ALLOCA_N(char, DTYPE_SIZES[dtype] * work_size);
    int* iwork  = NM_ALLOCA_N(int, 8*min_mn);

    int info = gesdd_table[dtype](JOBZ, M, N, NM_STORAGE_DENSE(a)->elements, FIX2INT(lda),
      NM_STORAGE_DENSE(s)->elements, NM_STORAGE_DENSE(u)->elements, FIX2INT(ldu), NM_STORAGE_DENSE(vt)->elements, FIX2INT(ldvt),
      work, work_size, iwork, rwork);
    return INT2FIX(info);
  }
}

/*
 * Function signature conversion for calling CBLAS' geev functions as directly as possible.
 *
 * GEEV computes for an N-by-N real nonsymmetric matrix A, the
 * eigenvalues and, optionally, the left and/or right eigenvectors.
 *
 * The right eigenvector v(j) of A satisfies
 *                    A * v(j) = lambda(j) * v(j)
 * where lambda(j) is its eigenvalue.
 *
 * The left eigenvector u(j) of A satisfies
 *                 u(j)**H * A = lambda(j) * u(j)**H
 * where u(j)**H denotes the conjugate transpose of u(j).
 *
 * The computed eigenvectors are normalized to have Euclidean norm
 * equal to 1 and largest component real.
 */
static VALUE nm_lapack_geev(VALUE self, VALUE compute_left, VALUE compute_right, VALUE n, VALUE a, VALUE lda, VALUE w, VALUE wi, VALUE vl, VALUE ldvl, VALUE vr, VALUE ldvr, VALUE lwork) {
  static int (*geev_table[nm::NUM_DTYPES])(char, char, int, void* a, int, void* w, void* wi, void* vl, int, void* vr, int, void* work, int, void* rwork) = {
    NULL, NULL, NULL, NULL, NULL, // no integer ops
    nm::math::lapack_geev<float,float>,
    nm::math::lapack_geev<double,double>,
    nm::math::lapack_geev<nm::Complex64,float>,
    nm::math::lapack_geev<nm::Complex128,double>,
    NULL, NULL, NULL, NULL // no rationals or Ruby objects
  };

  nm::dtype_t dtype = NM_DTYPE(a);


  if (!geev_table[dtype]) {
    rb_raise(rb_eNotImpError, "this operation not yet implemented for non-BLAS dtypes");
    return Qfalse;
  } else {
    int N = FIX2INT(n);

    char JOBVL = lapack_evd_job_sym(compute_left),
         JOBVR = lapack_evd_job_sym(compute_right);

    void* A  = NM_STORAGE_DENSE(a)->elements;
    void* WR = NM_STORAGE_DENSE(w)->elements;
    void* WI = wi == Qnil ? NULL : NM_STORAGE_DENSE(wi)->elements;
    void* VL = NM_STORAGE_DENSE(vl)->elements;
    void* VR = NM_STORAGE_DENSE(vr)->elements;

    // only need rwork for complex matrices (wi == Qnil for complex)
    int rwork_size  = dtype == nm::COMPLEX64 || dtype == nm::COMPLEX128 ? N * DTYPE_SIZES[dtype] : 0; // 2*N*floattype for complex only, otherwise 0
    void* rwork     = rwork_size > 0 ? NM_ALLOCA_N(char, rwork_size) : NULL;
    int work_size   = FIX2INT(lwork);
    void* work;

    int info;

    // if work size is 0 or -1, query.
    if (work_size <= 0) {
      work_size = -1;
      work = NM_ALLOC_N(char, DTYPE_SIZES[dtype]); //2*N * DTYPE_SIZES[dtype]);
      info = geev_table[dtype](JOBVL, JOBVR, N, A, FIX2INT(lda), WR, WI, VL, FIX2INT(ldvl), VR, FIX2INT(ldvr), work, work_size, rwork);
      work_size = (int)(dtype == nm::COMPLEX64 || dtype == nm::FLOAT32 ? reinterpret_cast<float*>(work)[0] : reinterpret_cast<double*>(work)[0]);
      // line above is basically: work_size = (int)(work[0]); // now have new work_size
      NM_FREE(work);
      if (info == 0)
        rb_warn("geev: calculated optimal lwork of %d; to eliminate this message, use a positive value for lwork (at least 2*shape[i])", work_size);
      else return INT2FIX(info); // error of some kind on query!
    }

    // if work size is < 2*N, just set it to 2*N
    if (work_size < 2*N) work_size = 2*N;
    if (work_size < 3*N && (dtype == nm::FLOAT32 || dtype == nm::FLOAT64)) {
      work_size = JOBVL == 'V' || JOBVR == 'V' ? 4*N : 3*N;
    }

    // Allocate work array for actual run
    work = NM_ALLOCA_N(char, work_size * DTYPE_SIZES[dtype]);

    // Perform the actual calculation.
    info = geev_table[dtype](JOBVL, JOBVR, N, A, FIX2INT(lda), WR, WI, VL, FIX2INT(ldvl), VR, FIX2INT(ldvr), work, work_size, rwork);

    return INT2FIX(info);
  }
}

static VALUE nm_clapack_lauum(VALUE self, VALUE order, VALUE uplo, VALUE n, VALUE a, VALUE lda) {
  static int (*ttable[nm::NUM_DTYPES])(const enum CBLAS_ORDER, const enum CBLAS_UPLO, const int n, void* a, const int lda) = {
      /*nm::math::clapack_lauum<uint8_t, false>,
      nm::math::clapack_lauum<int8_t, false>,
      nm::math::clapack_lauum<int16_t, false>,
      nm::math::clapack_lauum<uint32_t, false>,
      nm::math::clapack_lauum<uint64_t, false>,*/
      NULL, NULL, NULL, NULL, NULL,
      nm::math::clapack_lauum<false, float>,
      nm::math::clapack_lauum<false, double>,
#if defined (HAVE_CLAPACK_H) || defined (HAVE_ATLAS_CLAPACK_H)
      clapack_clauum, clapack_zlauum, // call directly, same function signature!
#else // Especially important for Mac OS, which doesn't seem to include the ATLAS clapack interface.
      nm::math::clapack_lauum<true, nm::Complex64>,
      nm::math::clapack_lauum<true, nm::Complex128>,
#endif
/*
      nm::math::clapack_lauum<nm::Rational32, false>,
      nm::math::clapack_lauum<nm::Rational64, false>,
      nm::math::clapack_lauum<nm::Rational128, false>,
      nm::math::clapack_lauum<nm::RubyObject, false>

*/
  };

  if (!ttable[NM_DTYPE(a)]) {
    rb_raise(rb_eNotImpError, "does not yet work for non-BLAS dtypes (needs herk, syrk, trmm)");
  } else {
    // Call either our version of lauum or the LAPACK version.
    ttable[NM_DTYPE(a)](blas_order_sym(order), blas_uplo_sym(uplo), FIX2INT(n), NM_STORAGE_DENSE(a)->elements, FIX2INT(lda));
  }

  return a;
}


/* Call any of the clapack_xgetrf functions as directly as possible.
 *
 * The clapack_getrf functions (dgetrf, sgetrf, cgetrf, and zgetrf) compute an LU factorization of a general M-by-N
 * matrix A using partial pivoting with row interchanges.
 *
 * The factorization has the form:
 *    A = P * L * U
 * where P is a permutation matrix, L is lower triangular with unit diagonal elements (lower trapezoidal if m > n),
 * and U is upper triangular (upper trapezoidal if m < n).
 *
 * This is the right-looking level 3 BLAS version of the algorithm.
 *
 * == Arguments
 * See: http://www.netlib.org/lapack/double/dgetrf.f
 * (You don't need argument 5; this is the value returned by this function.)
 *
 * You probably don't want to call this function. Instead, why don't you try clapack_getrf, which is more flexible
 * with its arguments?
 *
 * This function does almost no type checking. Seriously, be really careful when you call it! There's no exception
 * handling, so you can easily crash Ruby!
 *
 * Returns an array giving the pivot indices (normally these are argument #5).
 */
static VALUE nm_atlas_clapack_getrf(VALUE self, VALUE order, VALUE m, VALUE n, VALUE a, VALUE lda) {
  static int (*ttable[nm::NUM_DTYPES])(const enum CBLAS_ORDER, const int m, const int n, void* a, const int lda, int* ipiv) = {
      NULL, NULL, NULL, NULL, NULL, // integers not allowed due to division
      nm::math::atlas::clapack_getrf<float>,
      nm::math::atlas::clapack_getrf<double>,
#if defined (HAVE_CLAPACK_H) || defined (HAVE_ATLAS_CLAPACK_H)
      clapack_cgetrf, clapack_zgetrf, // call directly, same function signature!
#else // Especially important for Mac OS, which doesn't seem to include the ATLAS clapack interface.
      nm::math::atlas::clapack_getrf<nm::Complex64>,
      nm::math::atlas::clapack_getrf<nm::Complex128>,
#endif
      nm::math::atlas::clapack_getrf<nm::Rational32>,
      nm::math::atlas::clapack_getrf<nm::Rational64>,
      nm::math::atlas::clapack_getrf<nm::Rational128>,
      nm::math::atlas::clapack_getrf<nm::RubyObject>
  };

  int M = FIX2INT(m),
      N = FIX2INT(n);

  // Allocate the pivot index array, which is of size MIN(M, N).
  size_t ipiv_size = std::min(M,N);
  int* ipiv = NM_ALLOCA_N(int, ipiv_size);

  if (!ttable[NM_DTYPE(a)]) {
    rb_raise(nm_eDataTypeError, "this matrix operation undefined for integer matrices");
  } else {
    // Call either our version of getrf or the LAPACK version.
    ttable[NM_DTYPE(a)](blas_order_sym(order), M, N, NM_STORAGE_DENSE(a)->elements, FIX2INT(lda), ipiv);
  }

  // Result will be stored in a. We return ipiv as an array.
  VALUE ipiv_array = rb_ary_new2(ipiv_size);
  for (size_t i = 0; i < ipiv_size; ++i) {
    rb_ary_store(ipiv_array, i, INT2FIX(ipiv[i]));
  }

  return ipiv_array;
}


/* Call any of the clapack_xpotrf functions as directly as possible.
 *
 * You probably don't want to call this function. Instead, why don't you try clapack_potrf, which is more flexible
 * with its arguments?
 *
 * This function does almost no type checking. Seriously, be really careful when you call it! There's no exception
 * handling, so you can easily crash Ruby!
 *
 * Returns an array giving the pivot indices (normally these are argument #5).
 */
static VALUE nm_clapack_potrf(VALUE self, VALUE order, VALUE uplo, VALUE n, VALUE a, VALUE lda) {
#if !defined(HAVE_CLAPACK_H) && !defined(HAVE_ATLAS_CLAPACK_H)
  rb_raise(rb_eNotImpError, "potrf currently requires CLAPACK");
#endif

  static int (*ttable[nm::NUM_DTYPES])(const enum CBLAS_ORDER, const enum CBLAS_UPLO, const int n, void* a, const int lda) = {
      NULL, NULL, NULL, NULL, NULL, // integers not allowed due to division
      nm::math::clapack_potrf<float>,
      nm::math::clapack_potrf<double>,
#if defined (HAVE_CLAPACK_H) || defined (HAVE_ATLAS_CLAPACK_H)
      clapack_cpotrf, clapack_zpotrf, // call directly, same function signature!
#else // Especially important for Mac OS, which doesn't seem to include the ATLAS clapack interface.
      nm::math::clapack_potrf<nm::Complex64>,
      nm::math::clapack_potrf<nm::Complex128>,
#endif
      NULL, NULL, NULL, NULL /*
      nm::math::clapack_potrf<nm::Rational32>,
      nm::math::clapack_potrf<nm::Rational64>,
      nm::math::clapack_potrf<nm::Rational128>,
      nm::math::clapack_potrf<nm::RubyObject> */
  };

  if (!ttable[NM_DTYPE(a)]) {
    rb_raise(rb_eNotImpError, "this operation not yet implemented for non-BLAS dtypes");
    // FIXME: Once BLAS dtypes are implemented, replace error above with the error below.
    //rb_raise(nm_eDataTypeError, "this matrix operation undefined for integer matrices");
  } else {
    // Call either our version of potrf or the LAPACK version.
    ttable[NM_DTYPE(a)](blas_order_sym(order), blas_uplo_sym(uplo), FIX2INT(n), NM_STORAGE_DENSE(a)->elements, FIX2INT(lda));
  }

  return a;
}


/*
 * Call any of the clapack_xgetrs functions as directly as possible.
 */
static VALUE nm_clapack_getrs(VALUE self, VALUE order, VALUE trans, VALUE n, VALUE nrhs, VALUE a, VALUE lda, VALUE ipiv, VALUE b, VALUE ldb) {
  static int (*ttable[nm::NUM_DTYPES])(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE Trans, const int N,
                                       const int NRHS, const void* A, const int lda, const int* ipiv, void* B,
                                       const int ldb) = {
      NULL, NULL, NULL, NULL, NULL, // integers not allowed due to division
      nm::math::clapack_getrs<float>,
      nm::math::clapack_getrs<double>,
#if defined (HAVE_CLAPACK_H) || defined (HAVE_ATLAS_CLAPACK_H)
      clapack_cgetrs, clapack_zgetrs, // call directly, same function signature!
#else // Especially important for Mac OS, which doesn't seem to include the ATLAS clapack interface.
      nm::math::clapack_getrs<nm::Complex64>,
      nm::math::clapack_getrs<nm::Complex128>,
#endif
      nm::math::clapack_getrs<nm::Rational32>,
      nm::math::clapack_getrs<nm::Rational64>,
      nm::math::clapack_getrs<nm::Rational128>,
      nm::math::clapack_getrs<nm::RubyObject>
  };

  // Allocate the C version of the pivot index array
  int* ipiv_;
  if (TYPE(ipiv) != T_ARRAY) {
    rb_raise(rb_eArgError, "ipiv must be of type Array");
  } else {
    ipiv_ = NM_ALLOCA_N(int, RARRAY_LEN(ipiv));
    for (int index = 0; index < RARRAY_LEN(ipiv); ++index) {
      ipiv_[index] = FIX2INT( RARRAY_PTR(ipiv)[index] );
    }
  }

  if (!ttable[NM_DTYPE(a)]) {
    rb_raise(nm_eDataTypeError, "this matrix operation undefined for integer matrices");
  } else {

    // Call either our version of getrs or the LAPACK version.
    ttable[NM_DTYPE(a)](blas_order_sym(order), blas_transpose_sym(trans), FIX2INT(n), FIX2INT(nrhs), NM_STORAGE_DENSE(a)->elements, FIX2INT(lda),
                        ipiv_, NM_STORAGE_DENSE(b)->elements, FIX2INT(ldb));
  }

  // b is both returned and modified directly in the argument list.
  return b;
}


/*
 * Call any of the clapack_xpotrs functions as directly as possible.
 */
static VALUE nm_clapack_potrs(VALUE self, VALUE order, VALUE uplo, VALUE n, VALUE nrhs, VALUE a, VALUE lda, VALUE b, VALUE ldb) {
  static int (*ttable[nm::NUM_DTYPES])(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const int N,
                                       const int NRHS, const void* A, const int lda, void* B, const int ldb) = {
      NULL, NULL, NULL, NULL, NULL, // integers not allowed due to division
      nm::math::clapack_potrs<float,false>,
      nm::math::clapack_potrs<double,false>,
#if defined (HAVE_CLAPACK_H) || defined (HAVE_ATLAS_CLAPACK_H)
      clapack_cpotrs, clapack_zpotrs, // call directly, same function signature!
#else // Especially important for Mac OS, which doesn't seem to include the ATLAS clapack interface.
      nm::math::clapack_potrs<nm::Complex64,true>,
      nm::math::clapack_potrs<nm::Complex128,true>,
#endif
      nm::math::clapack_potrs<nm::Rational32,false>,
      nm::math::clapack_potrs<nm::Rational64,false>,
      nm::math::clapack_potrs<nm::Rational128,false>,
      nm::math::clapack_potrs<nm::RubyObject,false>
  };


  if (!ttable[NM_DTYPE(a)]) {
    rb_raise(nm_eDataTypeError, "this matrix operation undefined for integer matrices");
  } else {

    // Call either our version of potrs or the LAPACK version.
    ttable[NM_DTYPE(a)](blas_order_sym(order), blas_uplo_sym(uplo), FIX2INT(n), FIX2INT(nrhs), NM_STORAGE_DENSE(a)->elements, FIX2INT(lda),
                        NM_STORAGE_DENSE(b)->elements, FIX2INT(ldb));
  }

  // b is both returned and modified directly in the argument list.
  return b;
}

/* Call any of the clapack_xgetri functions as directly as possible.
 *
 * You probably don't want to call this function. Instead, why don't you try clapack_getri, which is more flexible
 * with its arguments?
 *
 * This function does almost no type checking. Seriously, be really careful when you call it! There's no exception
 * handling, so you can easily crash Ruby!
 *
 * Returns an array giving the pivot indices (normally these are argument #5).
 */
static VALUE nm_atlas_clapack_getri(VALUE self, VALUE order, VALUE n, VALUE a, VALUE lda, VALUE ipiv) {
#if !defined (HAVE_CLAPACK_H) && !defined (HAVE_ATLAS_CLAPACK_H)
  rb_raise(rb_eNotImpError, "getri currently requires CLAPACK");
#endif

  static int (*ttable[nm::NUM_DTYPES])(const enum CBLAS_ORDER, const int n, void* a, const int lda, const int* ipiv) = {
      NULL, NULL, NULL, NULL, NULL, // integers not allowed due to division
      nm::math::atlas::clapack_getri<float>,
      nm::math::atlas::clapack_getri<double>,
#if defined (HAVE_CLAPACK_H) || defined (HAVE_ATLAS_CLAPACK_H)
      clapack_cgetri, clapack_zgetri, // call directly, same function signature!
#else // Especially important for Mac OS, which doesn't seem to include the ATLAS clapack interface.
      nm::math::atlas::clapack_getri<nm::Complex64>,
      nm::math::atlas::clapack_getri<nm::Complex128>,
#endif
      NULL, NULL, NULL, NULL /*
      nm::math::clapack_getri<nm::Rational32>,
      nm::math::clapack_getri<nm::Rational64>,
      nm::math::clapack_getri<nm::Rational128>,
      nm::math::clapack_getri<nm::RubyObject> */
  };

  // Allocate the C version of the pivot index array
  int* ipiv_;
  if (TYPE(ipiv) != T_ARRAY) {
    rb_raise(rb_eArgError, "ipiv must be of type Array");
  } else {
    ipiv_ = NM_ALLOCA_N(int, RARRAY_LEN(ipiv));
    for (int index = 0; index < RARRAY_LEN(ipiv); ++index) {
      ipiv_[index] = FIX2INT( RARRAY_PTR(ipiv)[index] );
    }
  }

  if (!ttable[NM_DTYPE(a)]) {
    rb_raise(rb_eNotImpError, "this operation not yet implemented for non-BLAS dtypes");
    // FIXME: Once non-BLAS dtypes are implemented, replace error above with the error below.
    //rb_raise(nm_eDataTypeError, "this matrix operation undefined for integer matrices");
  } else {
    // Call either our version of getri or the LAPACK version.
    ttable[NM_DTYPE(a)](blas_order_sym(order), FIX2INT(n), NM_STORAGE_DENSE(a)->elements, FIX2INT(lda), ipiv_);
  }

  return a;
}


/* Call any of the clapack_xpotri functions as directly as possible.
 *
 * You probably don't want to call this function. Instead, why don't you try clapack_potri, which is more flexible
 * with its arguments?
 *
 * This function does almost no type checking. Seriously, be really careful when you call it! There's no exception
 * handling, so you can easily crash Ruby!
 *
 * Returns an array giving the pivot indices (normally these are argument #5).
 */
static VALUE nm_clapack_potri(VALUE self, VALUE order, VALUE uplo, VALUE n, VALUE a, VALUE lda) {
#if !defined (HAVE_CLAPACK_H) && !defined (HAVE_ATLAS_CLAPACK_H)
  rb_raise(rb_eNotImpError, "getri currently requires CLAPACK");
#endif

  static int (*ttable[nm::NUM_DTYPES])(const enum CBLAS_ORDER, const enum CBLAS_UPLO, const int n, void* a, const int lda) = {
      NULL, NULL, NULL, NULL, NULL, // integers not allowed due to division
      nm::math::clapack_potri<float>,
      nm::math::clapack_potri<double>,
#if defined (HAVE_CLAPACK_H) || defined (HAVE_ATLAS_CLAPACK_H)
      clapack_cpotri, clapack_zpotri, // call directly, same function signature!
#else // Especially important for Mac OS, which doesn't seem to include the ATLAS clapack interface.
      nm::math::clapack_potri<nm::Complex64>,
      nm::math::clapack_potri<nm::Complex128>,
#endif
      NULL, NULL, NULL, NULL /*
      nm::math::clapack_getri<nm::Rational32>,
      nm::math::clapack_getri<nm::Rational64>,
      nm::math::clapack_getri<nm::Rational128>,
      nm::math::clapack_getri<nm::RubyObject> */
  };

  if (!ttable[NM_DTYPE(a)]) {
    rb_raise(rb_eNotImpError, "this operation not yet implemented for non-BLAS dtypes");
    // FIXME: Once BLAS dtypes are implemented, replace error above with the error below.
    //rb_raise(nm_eDataTypeError, "this matrix operation undefined for integer matrices");
  } else {
    // Call either our version of getri or the LAPACK version.
    ttable[NM_DTYPE(a)](blas_order_sym(order), blas_uplo_sym(uplo), FIX2INT(n), NM_STORAGE_DENSE(a)->elements, FIX2INT(lda));
  }

  return a;
}


/*
 * Call any of the clapack_xlaswp functions as directly as possible.
 *
 * Note that LAPACK's xlaswp functions accept a column-order matrix, but NMatrix uses row-order. Thus, n should be the
 * number of rows and lda should be the number of columns, no matter what it says in the documentation for dlaswp.f.
 */
static VALUE nm_clapack_laswp(VALUE self, VALUE n, VALUE a, VALUE lda, VALUE k1, VALUE k2, VALUE ipiv, VALUE incx) {
  static void (*ttable[nm::NUM_DTYPES])(const int n, void* a, const int lda, const int k1, const int k2, const int* ipiv, const int incx) = {
      nm::math::clapack_laswp<uint8_t>,
      nm::math::clapack_laswp<int8_t>,
      nm::math::clapack_laswp<int16_t>,
      nm::math::clapack_laswp<int32_t>,
      nm::math::clapack_laswp<int64_t>,
      nm::math::clapack_laswp<float>,
      nm::math::clapack_laswp<double>,
//#ifdef HAVE_CLAPACK_H // laswp doesn't actually exist in clapack.h!
//      clapack_claswp, clapack_zlaswp, // call directly, same function signature!
//#else // Especially important for Mac OS, which doesn't seem to include the ATLAS clapack interface.
      nm::math::clapack_laswp<nm::Complex64>,
      nm::math::clapack_laswp<nm::Complex128>,
//#endif
      nm::math::clapack_laswp<nm::Rational32>,
      nm::math::clapack_laswp<nm::Rational64>,
      nm::math::clapack_laswp<nm::Rational128>,
      nm::math::clapack_laswp<nm::RubyObject>
  };

  // Allocate the C version of the pivot index array
  int* ipiv_;
  if (TYPE(ipiv) != T_ARRAY) {
    rb_raise(rb_eArgError, "ipiv must be of type Array");
  } else {
    ipiv_ = NM_ALLOCA_N(int, RARRAY_LEN(ipiv));
    for (int index = 0; index < RARRAY_LEN(ipiv); ++index) {
      ipiv_[index] = FIX2INT( RARRAY_PTR(ipiv)[index] );
    }
  }

  // Call either our version of laswp or the LAPACK version.
  ttable[NM_DTYPE(a)](FIX2INT(n), NM_STORAGE_DENSE(a)->elements, FIX2INT(lda), FIX2INT(k1), FIX2INT(k2), ipiv_, FIX2INT(incx));

  // a is both returned and modified directly in the argument list.
  return a;
}


}
