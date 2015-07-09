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

//potrf
template <typename DType>
inline int potrf(const enum CBLAS_ORDER order, char uplo, const int N, DType* A, const int lda) {
  rb_raise(rb_eNotImpError, "not implemented for non-BLAS dtypes");
  return 0;
}

template <>
inline int potrf(const enum CBLAS_ORDER order, char uplo, const int N, float* A, const int lda) {
  return LAPACKE_spotrf(order, uplo, N, A, lda);
}

template <>
inline int potrf(const enum CBLAS_ORDER order, char uplo, const int N, double* A, const int lda) {
  return LAPACKE_dpotrf(order, uplo, N, A, lda);
}

template <>
inline int potrf(const enum CBLAS_ORDER order, char uplo, const int N, Complex64* A, const int lda) {
  return LAPACKE_cpotrf(order, uplo, N, A, lda);
}

template <>
inline int potrf(const enum CBLAS_ORDER order, char uplo, const int N, Complex128* A, const int lda) {
  return LAPACKE_zpotrf(order, uplo, N, A, lda);
}

template <typename DType>
inline int lapacke_potrf(const enum CBLAS_ORDER order, char uplo, const int n, void* a, const int lda) {
  return potrf<DType>(order, uplo, n, static_cast<DType*>(a), lda);
}

//potrs
template <typename DType>
inline int potrs(const enum CBLAS_ORDER Order, char Uplo, const int N, const int NRHS, const DType* A,
           const int lda, DType* B, const int ldb)
{
  rb_raise(rb_eNotImpError, "not implemented for non-BLAS dtypes");
  return 0;
}

template <>
inline int potrs<float> (const enum CBLAS_ORDER Order, char Uplo, const int N, const int NRHS, const float* A,
           const int lda, float* B, const int ldb)
{
  return LAPACKE_spotrs(Order, Uplo, N, NRHS, A, lda, B, ldb);
}

template <>
inline int potrs<double>(const enum CBLAS_ORDER Order, char Uplo, const int N, const int NRHS, const double* A,
           const int lda, double* B, const int ldb)
{
  return LAPACKE_dpotrs(Order, Uplo, N, NRHS, A, lda, B, ldb);
}

template <>
inline int potrs<Complex64>(const enum CBLAS_ORDER Order, char Uplo, const int N, const int NRHS, const Complex64* A,
           const int lda, Complex64* B, const int ldb)
{
  return LAPACKE_cpotrs(Order, Uplo, N, NRHS, A, lda, B, ldb);
}

template <>
inline int potrs<Complex128>(const enum CBLAS_ORDER Order, char Uplo, const int N, const int NRHS, const Complex128* A,
           const int lda, Complex128* B, const int ldb)
{
  return LAPACKE_zpotrs(Order, Uplo, N, NRHS, A, lda, B, ldb);
}

template <typename DType>
inline int lapacke_potrs(const enum CBLAS_ORDER order, char uplo, const int n, const int nrhs,
                         const void* a, const int lda, void* b, const int ldb) {
  return potrs<DType>(order, uplo, n, nrhs, static_cast<const DType*>(a), lda, static_cast<DType*>(b), ldb);
}

//potri
template <typename DType>
inline int potri(const enum CBLAS_ORDER order, char uplo, const int n, DType* a, const int lda) {
  rb_raise(rb_eNotImpError, "potri not yet implemented for non-BLAS dtypes");
  return 0;
}

template <>
inline int potri(const enum CBLAS_ORDER order, char uplo, const int n, float* a, const int lda) {
  return LAPACKE_spotri(order, uplo, n, a, lda);
}

template <>
inline int potri(const enum CBLAS_ORDER order, char uplo, const int n, double* a, const int lda) {
  return LAPACKE_dpotri(order, uplo, n, a, lda);
}

template <>
inline int potri(const enum CBLAS_ORDER order, char uplo, const int n, Complex64* a, const int lda) {
  return LAPACKE_cpotri(order, uplo, n, a, lda);
}

template <>
inline int potri(const enum CBLAS_ORDER order, char uplo, const int n, Complex128* a, const int lda) {
  return LAPACKE_zpotri(order, uplo, n, a, lda);
}

template <typename DType>
inline int lapacke_potri(const enum CBLAS_ORDER order, char uplo, const int n, void* a, const int lda) {
  return potri<DType>(order, uplo, n, static_cast<DType*>(a), lda);
}


}}}

#endif
