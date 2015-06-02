#include <ruby.h>

#include "nmatrix.h"

extern "C" {
#if defined HAVE_CBLAS_H
  #include <cblas.h>
#elif defined HAVE_ATLAS_CBLAS_H
  #include <atlas/cblas.h>
#endif

#if defined HAVE_CLAPACK_H
  #include <clapack.h>
#elif defined HAVE_ATLAS_CLAPACK_H
  #include <atlas/clapack.h>
#endif
}

#include "data/data.h"

#include "getri.h"

VALUE cNMatrix;
VALUE cNMatrix_LAPACK;


static inline enum CBLAS_ORDER blas_order_sym(VALUE op) {
  if (rb_to_id(op) == rb_intern("row") || rb_to_id(op) == rb_intern("row_major")) return CblasRowMajor;
  else if (rb_to_id(op) == rb_intern("col") || rb_to_id(op) == rb_intern("col_major") ||
           rb_to_id(op) == rb_intern("column") || rb_to_id(op) == rb_intern("column_major")) return CblasColMajor;
  rb_raise(rb_eArgError, "Expected :row or :col for order argument");
  return CblasRowMajor;
}

extern "C" {

static VALUE nm_clapack_getri(VALUE self, VALUE order, VALUE n, VALUE a, VALUE lda, VALUE ipiv) {
  //nm::NUM_DTYPES is in data/data.h, but theres a NM_NUM_DTYPES in nmatrix.h
  static int (*ttable[NM_NUM_DTYPES])(const enum CBLAS_ORDER, const int n, void* a, const int lda, const int* ipiv) = {
      NULL, NULL, NULL, NULL, NULL, // integers not allowed due to division
      nm::math::clapack_getri<float>,
      nm::math::clapack_getri<double>,
      clapack_cgetri, clapack_zgetri, // call directly, same function signature!
      //nm::math::clapack_getri<nm::Complex64>,
      //nm::math::clapack_getri<nm::Complex128>,
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
  //NM_ALLOCA_N in nm_memory.h
    ipiv_ = NM_ALLOCA_N(int, RARRAY_LEN(ipiv));
    for (int index = 0; index < RARRAY_LEN(ipiv); ++index) {
      ipiv_[index] = FIX2INT( RARRAY_PTR(ipiv)[index] );
    }
  }

  //nmatrix.h
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

static VALUE nm_test(VALUE self) {
  return INT2NUM(2);
}

void Init_nmatrix_atlas() {
  cNMatrix = rb_define_class("NMatrix", rb_cObject);
  cNMatrix_LAPACK = rb_define_module_under(cNMatrix, "LAPACK");

  rb_define_singleton_method(cNMatrix_LAPACK, "clapack_getri_test", (METHOD)nm_clapack_getri, 5);

  rb_define_method(cNMatrix, "test_c_ext_return_2", (METHOD)nm_test, 0);
}

}
