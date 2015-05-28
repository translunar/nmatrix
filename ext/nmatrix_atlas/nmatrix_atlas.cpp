#include <ruby.h>

VALUE cNMatrix;

extern "C" {

static VALUE nm_test(VALUE self) {
  return INT2NUM(2);
}

void Init_nmatrix_atlas() {
  cNMatrix = rb_define_class("NMatrix", rb_cObject);

  //the cast should be to METHOD once we add the nmatrix headers
  rb_define_method(cNMatrix, "test_c_ext_return_2", (VALUE (*)(...))nm_test, 0);
}

}
