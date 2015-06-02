//Following structure of ext/nmatrix/math.cpp
/*
 * Project Includes
 */

#include "math/gesvd.h"
#include "math/gesdd.h"
#include "math/geev.h"

/*
 * Forward Declarations
 */

extern "C" {
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

void nm_math_init_something() {
	cNMatrix_LAPACK = rb_define_module_under(cNMatrix, "LAPACK");

  /* Non-ATLAS regular LAPACK Functions called via Fortran interface */
  rb_define_singleton_method(cNMatrix_LAPACK, "lapack_gesvd", (METHOD)nm_lapack_gesvd, 12);
  rb_define_singleton_method(cNMatrix_LAPACK, "lapack_gesdd", (METHOD)nm_lapack_gesdd, 11);
  rb_define_singleton_method(cNMatrix_LAPACK, "lapack_geev",  (METHOD)nm_lapack_geev,  12);

}

/*
 * Interprets lapack jobu and jobvt arguments, for which LAPACK needs character values A, S, O, or N.
 *
 * Called by lapack_gesvd -- basically inline. svd stands for singular value decomposition.
 */
static inline char lapack_svd_job_sym(VALUE op) {
  if (rb_to_id(op) == rb_intern("all") || rb_to_id(op) == rb_intern("a")) return 'A';
  else if (rb_to_id(op) == rb_intern("return") || rb_to_id(op) == rb_intern("s")) return 'S';
  else if (rb_to_id(op) == rb_intern("overwrite") || rb_to_id(op) == rb_intern("o")) return 'O';
  else if (rb_to_id(op) == rb_intern("none") || rb_to_id(op) == rb_intern("n")) return 'N';
  else rb_raise(rb_eArgError, "Expected :all, :return, :overwrite, :none (or :a, :s, :o, :n, respectively)");
  return 'a';
}

/*
 * Interprets lapack jobvl and jobvr arguments, for which LAPACK needs character values N or V.
 *
 * Called by lapack_geev -- basically inline. evd stands for eigenvalue decomposition.
 */
static inline char lapack_evd_job_sym(VALUE op) {
  if (op == Qfalse || op == Qnil || rb_to_id(op) == rb_intern("n")) return 'N';
  else return 'V';
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

}
