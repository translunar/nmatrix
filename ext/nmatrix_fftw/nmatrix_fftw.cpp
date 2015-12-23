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
#include "storage/common.cpp"
#include "nmatrix.h"
#include "nm_memory.h"
#include "data/complex.h"

#include <iostream>
using namespace std;

#define TYPE_COMPLEX_COMPLEX 0
#define TYPE_REAL_COMPLEX    1
#define TYPE_COMPLEX_REAL    2
#define TYPE_REAL_REAL       3

static VALUE cNMatrix_FFTW_Plan_Data;

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

static int* interpret_shape(VALUE rb_shape, const int dimension)
{
  Check_Type(rb_shape, T_ARRAY);

  int *shape = new int[dimension];
  VALUE *arr = RARRAY_PTR(rb_shape);

  for (int i = 0; i < RARRAY_LEN(rb_shape); ++i) {
    shape[i] = FIX2INT(arr[i]);
  }

  return shape;
}

static void nm_fftw_create_complex_complex_plan(fftw_data *data, size_t size, 
  const int dimensions, const int* shape, int sign, unsigned flags)
{
  data->input = ALLOC_N(fftw_complex, size);
  data->output = ALLOC_N(fftw_complex, size);
  data->plan = fftw_plan_dft(dimensions, shape, data->input, data->output, 
    sign, flags);
}

static void nm_fftw_create_real_complex_plan(fftw_data *data, size_t size, 
  const int dimensions, const int* shape, int sign, unsigned flags)
{
  data->input = ALLOC_N(double, size);
  data->output = ALLOC_N(fftw_complex, size/n + 1);
  data->plan = fftw_plan_dft_r2c(dimensions, shape, data->input, data->output, 
    sign, flags);
}

static VALUE nm_fftw_create_plan(VALUE self, VALUE rb_shape, VALUE rb_size,
  VALUE rb_dim, VALUE rb_flags, VALUE rb_direction, VALUE rb_type)
{ 
  fftw_data *data     = ALLOC(fftw_data);
  const int dimensions = FIX2INT(rb_dim);
  const int* shape    = interpret_shape(rb_shape, dimensions);
  size_t size         = FIX2INT(rb_size);
  int sign            = FIX2INT(rb_direction);
  unsigned flags      = FIX2INT(rb_flags);

  switch (FIX2INT(rb_type))
  {
    case TYPE_COMPLEX_COMPLEX:
      nm_fftw_create_complex_complex_plan(data, size, dimensions, shape, sign, flags);
      break;
    case TYPE_REAL_COMPLEX:
      nm_fftw_create_real_complex_plan(data, size, dimensions, shape, sign, flags);
      break;
    case TYPE_COMPLEX_REAL:
      break;
    case TYPE_REAL_REAL:
      break;
    default:
      rb_raise(rb_eArgError, "Invalid type of DFT.");
  }

  return Data_Wrap_Struct(cNMatrix_FFTW_Plan_Data, NULL, nm_fftw_cleanup, data);
}

static VALUE nm_fftw_set_input(VALUE self, VALUE nmatrix, VALUE plan_data, 
  VALUE type)
{
  fftw_data *data;

  Data_Get_Struct(plan_data, fftw_data, data);

  switch(FIX2INT(type))
  {
    case TYPE_COMPLEX_COMPLEX:
    case TYPE_COMPLEX_REAL:
      memcpy(data->input, NM_DENSE_ELEMENTS(nmatrix), 
        sizeof(fftw_complex)*NM_DENSE_COUNT(nmatrix));
      break;
    case TYPE_REAL_COMPLEX:
    case TYPE_REAL_REAL:
      memcpy(data->input, NM_DENSE_ELEMENTS(nmatrix), 
        sizeof(double)*NM_DENSE_COUNT(nmatrix));
      break;
    default:
      rb_raise(rb_eArgError, "Invalid type of DFT.");
  }


  return self;
}

static VALUE nm_fftw_execute(VALUE self, VALUE plan_data, VALUE nmatrix, VALUE type)
{
  fftw_data *data;

  Data_Get_Struct(plan_data, fftw_data, data);
  fftw_execute(data->plan);

  switch(FIX2INT(type))
  {
    case TYPE_COMPLEX_COMPLEX:
    case TYPE_REAL_COMPLEX:
      memcpy(NM_DENSE_ELEMENTS(nmatrix), data->output, 
        sizeof(fftw_complex)*NM_DENSE_COUNT(nmatrix));
      break;
    case TYPE_COMPLEX_REAL:
    case TYPE_REAL_REAL:
      cout << "Jaldi karneka implement.";
      break;
    default:
      rb_raise(rb_eArgError, "Invalid type of DFT.");
  }

  return Qtrue;
}

extern "C" {
  void Init_nmatrix_fftw() 
  {
    VALUE cNMatrix = rb_define_class("NMatrix", rb_cObject);
    VALUE cNMatrix_FFTW = rb_define_module_under(cNMatrix, "FFTW");
    VALUE cNMatrix_FFTW_Plan = rb_define_class_under(cNMatrix_FFTW, "Plan", rb_cObject);
    VALUE cNMatrix_FFTW_Plan_Data = rb_define_class_under(
      cNMatrix_FFTW_Plan, "Data", rb_cObject);

    rb_define_private_method(cNMatrix_FFTW_Plan, "__create_plan__", 
      (METHOD)nm_fftw_create_plan, 6);
    rb_define_private_method(cNMatrix_FFTW_Plan, "__set_input__",
      (METHOD)nm_fftw_set_input, 3);
    rb_define_private_method(cNMatrix_FFTW_Plan, "__execute__",
      (METHOD)nm_fftw_execute, 3);
  }
}
