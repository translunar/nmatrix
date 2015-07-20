#include <ruby.h>

#include "nmatrix.h"

#include "math_atlas/inc.h"

#include "data/data.h"

extern "C" {
void nm_math_init_atlas(); 

void Init_nmatrix_atlas() {
  nm_math_init_atlas();
}

}
