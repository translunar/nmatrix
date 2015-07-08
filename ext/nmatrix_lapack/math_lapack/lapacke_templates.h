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

//getrs
template <typename DType>
inline int getrs(const enum CBLAS_ORDER Order, char Trans, const int N, const int NRHS, const DType* A,
           const int lda, const int* ipiv, DType* B, const int ldb)
{
  rb_raise(rb_eNotImpError, "lapacke_getrs not implemented for non_BLAS dtypes. Try clapack_getrs instead.");
  return 0;
}

template <>
inline int getrs(const enum CBLAS_ORDER Order, char Trans, const int N, const int NRHS, const float* A,
           const int lda, const int* ipiv, float* B, const int ldb)
{
  return LAPACKE_sgetrs(Order, Trans, N, NRHS, A, lda, ipiv, B, ldb);
}

template <>
inline int getrs(const enum CBLAS_ORDER Order, char Trans, const int N, const int NRHS, const double* A,
           const int lda, const int* ipiv, double* B, const int ldb)
{
  return LAPACKE_dgetrs(Order, Trans, N, NRHS, A, lda, ipiv, B, ldb);
}

template <>
inline int getrs(const enum CBLAS_ORDER Order, char Trans, const int N, const int NRHS, const Complex64* A,
           const int lda, const int* ipiv, Complex64* B, const int ldb)
{
  return LAPACKE_cgetrs(Order, Trans, N, NRHS, A, lda, ipiv, B, ldb);
}

template <>
inline int getrs(const enum CBLAS_ORDER Order, char Trans, const int N, const int NRHS, const Complex128* A,
           const int lda, const int* ipiv, Complex128* B, const int ldb)
{
  return LAPACKE_zgetrs(Order, Trans, N, NRHS, A, lda, ipiv, B, ldb);
}

template <typename DType>
inline int lapacke_getrs(const enum CBLAS_ORDER order, char trans, const int n, const int nrhs,
                         const void* a, const int lda, const int* ipiv, void* b, const int ldb) {
  return getrs<DType>(order, trans, n, nrhs, static_cast<const DType*>(a), lda, ipiv, static_cast<DType*>(b), ldb);
}

}}}

#endif
