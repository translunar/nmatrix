#ifndef CBLAS_TEMPLATES_LAPACK_H
#define CBLAS_TEMPLATES_LAPACK_H

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

namespace nm { namespace math { namespace lapack {
 
//Add cblas templates in the correct namespace
#include "math/cblas_templates_core.h"

}}}

#endif
