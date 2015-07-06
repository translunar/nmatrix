#ifndef CBLAS_TEMPLATES_ATLAS_H
#define CBLAS_TEMPLATES_ATLAS_H

//includes so we have access to internal implementations
#include "math/rotg.h"
#include "math/rot.h"
#include "math/asum.h"
#include "math/nrm2.h"
#include "math/imax.h"
#include "math/scal.h"
#include "math/gemv.h"
#include "math/gemm.h"
#include "math/trsm.h"

namespace nm { namespace math { namespace atlas {

//Add cblas templates in the correct namespace
#include "math/cblas_templates_core.h"

//Add complex specializations for rot and rotg. These cblas functions are not
//part of the the standard CBLAS and so need to be in an nmatrix-atlas header.
template <>
inline void rotg(Complex64* a, Complex64* b, Complex64* c, Complex64* s) {
  cblas_crotg(a, b, c, s);
}

template <>
inline void rotg(Complex128* a, Complex128* b, Complex128* c, Complex128* s) {
  cblas_zrotg(a, b, c, s);
}
template <>
inline void rot(const int N, Complex64* X, const int incX, Complex64* Y, const int incY, const float c, const float s) {
  cblas_csrot(N, X, incX, Y, incY, c, s);
}

template <>
inline void rot(const int N, Complex128* X, const int incX, Complex128* Y, const int incY, const double c, const double s) {
  cblas_zdrot(N, X, incX, Y, incY, c, s);
}

}}} //nm::math::atlas

#endif
