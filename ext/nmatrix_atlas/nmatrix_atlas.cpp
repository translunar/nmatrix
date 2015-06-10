#include <ruby.h>

#include "nmatrix.h"

#include "math_atlas/inc.h"

#include "data/data.h"

extern "C" {
void nm_math_init_atlas(); 

static VALUE nm_test(VALUE self) {
  return INT2NUM(2);
}

void Init_nmatrix_atlas() {
  rb_define_method(cNMatrix, "test_c_ext_return_2", (METHOD)nm_test, 0);

  nm_math_init_atlas();
}

}
