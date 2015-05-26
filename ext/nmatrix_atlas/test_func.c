#include <ruby.h>

VALUE cNMatrix;

static VALUE nm_test(VALUE self) {
  return INT2NUM(2);
}

void Init_nmatrix_atlas() {
  cNMatrix = rb_define_class("NMatrix", rb_cObject);

  rb_define_method(cNMatrix, "test_c_ext_return_2", nm_test, 0);
}
