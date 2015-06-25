#include "data/data.h"

#include "lapacke.h"

#include "math/cblas_enums.h"
#include "math/util.h"

namespace nm { namespace math { namespace lapack {

template <typename DType>
inline int getri(const enum CBLAS_ORDER order, const int n, DType* a, const int lda, const int* ipiv) {
  rb_raise(rb_eNotImpError, "getri not yet implemented for non-BLAS dtypes");
  return 0;
}

template <>
inline int getri(const enum CBLAS_ORDER order, const int n, float* a, const int lda, const int* ipiv) {
  return LAPACKE_sgetri(order, n, a, lda, ipiv);
}

template <>
inline int getri(const enum CBLAS_ORDER order, const int n, double* a, const int lda, const int* ipiv) {
  return LAPACKE_dgetri(order, n, a, lda, ipiv);
}

template <>
inline int getri(const enum CBLAS_ORDER order, const int n, Complex64* a, const int lda, const int* ipiv) {
  return LAPACKE_cgetri(order, n, a, lda, ipiv);
}

template <>
inline int getri(const enum CBLAS_ORDER order, const int n, Complex128* a, const int lda, const int* ipiv) {
  return LAPACKE_zgetri(order, n, a, lda, ipiv);
}

/*
 * Function signature conversion for calling LAPACK's getri functions as directly as possible.
 *
 * For documentation: http://www.netlib.org/lapack/double/dgetri.f
 *
 * This function should normally go in math.cpp, but we need it to be available to nmatrix.cpp.
 */
template <typename DType>
inline int clapack_getri(const enum CBLAS_ORDER order, const int n, void* a, const int lda, const int* ipiv) {
  return getri<DType>(order, n, static_cast<DType*>(a), lda, ipiv);
}


} } } // end nm::math::lapack

/*
 * Forward Declarations
 */

extern "C" {
  /* LAPACK. */
  //static VALUE nm_lapack_clapack_getrf(VALUE self, VALUE order, VALUE m, VALUE n, VALUE a, VALUE lda);
  //static VALUE nm_lapack_clapack_getrs(VALUE self, VALUE order, VALUE trans, VALUE n, VALUE nrhs, VALUE a, VALUE lda, VALUE ipiv, VALUE b, VALUE ldb);
  static VALUE nm_lapack_clapack_getri(VALUE self, VALUE order, VALUE n, VALUE a, VALUE lda, VALUE ipiv);
}

extern "C" {

///////////////////
// Ruby Bindings //
///////////////////

void nm_math_init_lapack() {

  /* LAPACK Functions */
  //rb_define_singleton_method(cNMatrix_LAPACK, "clapack_getrf", (METHOD)nm_lapack_clapack_getrf, 5);
  //rb_define_singleton_method(cNMatrix_LAPACK, "clapack_getrs", (METHOD)nm_lapack_clapack_getrs, 9);
  rb_define_singleton_method(cNMatrix_LAPACK, "clapack_getri", (METHOD)nm_lapack_clapack_getri, 5);
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
static VALUE nm_lapack_clapack_getri(VALUE self, VALUE order, VALUE n, VALUE a, VALUE lda, VALUE ipiv) {
  std::cout << "nm_lapack_clapack_getri" << std::endl;

  static int (*ttable[nm::NUM_DTYPES])(const enum CBLAS_ORDER, const int n, void* a, const int lda, const int* ipiv) = {
      NULL, NULL, NULL, NULL, NULL, // integers not allowed due to division
      nm::math::lapack::clapack_getri<float>,
      nm::math::lapack::clapack_getri<double>,
      nm::math::lapack::clapack_getri<nm::Complex64>,
      nm::math::lapack::clapack_getri<nm::Complex128>,
      NULL, NULL, NULL, NULL
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
  } else {
    // Call either our version of getri or the LAPACK version.
    ttable[NM_DTYPE(a)](blas_order_sym(order), FIX2INT(n), NM_STORAGE_DENSE(a)->elements, FIX2INT(lda), ipiv_);
  }

  return a;
}

}
