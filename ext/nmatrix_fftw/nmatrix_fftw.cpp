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
#include <complex.h>
#include <fftw3.h>
#include "nmatrix.h"
#include "nm_memory.h"
#include "data/complex.h"

#include <iostream>
using namespace std;

static VALUE cNMatrix_FFTW_Plan;

struct fftw_data {
  fftw_complex *input, *output;
  fftw_plan plan;
};
typedef struct fftw_data fftw_data;

static void nm_fftw_cleanup(fftw_data* d)
{
  fftw_destroy_plan(d->plan);
  xfree(d->input);
  xfree(d->output);
}

static VALUE nm_fftw_create_plan(VALUE self, VALUE shape)
{ 
  fftw_data *data = new fftw_data;

  data->input = ALLOC_N(fftw_complex, FIX2INT(shape));
  data->output = ALLOC_N(fftw_complex, FIX2INT(shape));
  data->plan = fftw_plan_dft_1d(FIX2INT(shape), 
    data->input, data->output, FFTW_FORWARD, FFTW_ESTIMATE);

  Data_Wrap_Struct(cNMatrix_FFTW_Plan, NULL, nm_fftw_cleanup, data);

  return self;
}

extern "C" {
  void Init_nmatrix_fftw() 
  {
    VALUE cNMatrix = rb_define_class("NMatrix", rb_cObject);
    VALUE cNMatrix_FFTW = rb_define_module_under(cNMatrix, "FFTW");
    VALUE cNMatrix_FFTW_Plan = rb_define_class_under(cNMatrix_FFTW, "Plan", rb_cObject);

    rb_define_private_method(cNMatrix_FFTW_Plan, "__create_plan__", 
      (METHOD)nm_fftw_create_plan, 1);
  }
}
