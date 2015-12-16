/////////////////////////////////////////////////////////////////////
// = NMatrix
//
// A linear algebra library for scientific computation in Ruby.
// NMatrix is part of SciRuby.
//
// NMatrix was originally inspired by and derived from NArray, by
// Masahiro Tanaka: http://narray.rubyforge.org
//
// == Copyright Information
//
// SciRuby is Copyright (c) 2010 - 2014, Ruby Science Foundation
// NMatrix is Copyright (c) 2012 - 2014, John Woods and the Ruby Science Foundation
//
// Please see LICENSE.txt for additional copyright notices.
//
// == Contributing
//
// By contributing source code to SciRuby, you agree to be bound by
// our Contributor Agreement:
//
// * https://github.com/SciRuby/sciruby/wiki/Contributor-Agreement
//
// == nmatrix_fftw.cpp
//
// Main file for nmatrix_fftw extension
//

#include <ruby.h>
#include <fftw3.h>
#include "nmatrix.h"
// #include "data/data.h"

static const int
fftw_size(VALUE self, VALUE nmatrix)
{

  return NUM2INT(rb_funcall(nmatrix, rb_intern("size"), 0));
}

VALUE fftw_complex_to_nm_complex(VALUE self, fftw_complex* in)
{
    return rb_funcall(rb_define_module("Kernel"),
                      rb_intern("Complex"),
                      2,
                      rb_float_new(((double (*)) in)[0]),
                      rb_float_new(((double (*)) in)[1]));
}

/**
  fftw_r2c
  @param self
  @param nmatrix
  @return nmatrix
  With FFTW_ESTIMATE as a flag in the plan,
  the input and and output are not overwritten at runtime
  The plan will use a heuristic approach to picking plans
  rather than take measurements
*/
static VALUE
fftw_r2c_one(VALUE self, VALUE in_nmatrix, VALUE out_nmatrix)
{

  fftw_plan plan;

  const int in_size = NUM2INT(rb_funcall(in_nmatrix, rb_intern("size"), 0));

  double* in = ALLOC_N(double, in_size);
  fftw_complex* out = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * in_size * in_size);

  for (int i = 0; i < in_size; i++)
  {
    in[i] = NUM2DBL(rb_funcall(in_nmatrix, rb_intern("[]"), 1, INT2FIX(i)));
  }

  plan = fftw_plan_dft_r2c(1, &in_size, in, out, FFTW_ESTIMATE);
  fftw_execute(plan);

  // Assign the output to the proper locations in the output nmatrix
  for (int i = 0; i < fftw_size(self, out_nmatrix); i++)
  {
    rb_funcall(out_nmatrix, rb_intern("[]="), 2, INT2FIX(i), fftw_complex_to_nm_complex(self, &out[i]));
  }

 // INFO: http://www.fftw.org/doc/New_002darray-Execute-Functions.html#New_002darray-Execute-Functions
  fftw_destroy_plan(plan);
  xfree(in);
  fftw_free(out);
  return out_nmatrix;
}

extern "C" {
  void Init_nmatrix_fftw() 
  {
    VALUE cNMatrix = rb_define_class("NMatrix", rb_cObject);
    VALUE cNMatrix_FFTW = rb_define_module_under(cNMatrix, "FFTW");

    rb_define_singleton_method(
      cNMatrix_FFTW, "r2c_one", (METHOD)fftw_r2c_one, 2);
  }
}
