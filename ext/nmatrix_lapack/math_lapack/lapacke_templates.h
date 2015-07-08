#ifndef LAPACKE_TEMPLATES_H
#define LAPACKE_TEMPLATES_H

namespace nm { namespace math { namespace lapack {

//getrf
template <typename DType>
inline int getrf(const enum CBLAS_ORDER order, const int m, const int n, DType* a, const int lda, int* ipiv) {
  //We don't want to call the internal implementation since the the CLAPACK interface is slightly different than the LAPACKE.
  rb_raise(rb_eNotImpError, "lapacke_getrf not implemented for non_BLAS dtypes. Try clapack_getrf instead.");
  return 0;
}

template <>
inline int getrf(const enum CBLAS_ORDER order, const int m, const int n, float* a, const int lda, int* ipiv) {
  return LAPACKE_sgetrf(order, m, n, a, lda, ipiv);
}

template <>
inline int getrf(const enum CBLAS_ORDER order, const int m, const int n, double* a, const int lda, int* ipiv) {
  return LAPACKE_dgetrf(order, m, n, a, lda, ipiv);
}

template <>
inline int getrf(const enum CBLAS_ORDER order, const int m, const int n, Complex64* a, const int lda, int* ipiv) {
  return LAPACKE_cgetrf(order, m, n, a, lda, ipiv);
}

template <>
inline int getrf(const enum CBLAS_ORDER order, const int m, const int n, Complex128* a, const int lda, int* ipiv) {
  return LAPACKE_zgetrf(order, m, n, a, lda, ipiv);
}

template <typename DType>
inline int lapacke_getrf(const enum CBLAS_ORDER order, const int m, const int n, void* a, const int lda, int* ipiv) {
  return getrf<DType>(order, m, n, static_cast<DType*>(a), lda, ipiv);
}

//getri
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

template <typename DType>
inline int lapacke_getri(const enum CBLAS_ORDER order, const int n, void* a, const int lda, const int* ipiv) {
  return getri<DType>(order, n, static_cast<DType*>(a), lda, ipiv);
}

}}}

#endif
