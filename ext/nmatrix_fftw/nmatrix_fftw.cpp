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

static const int* interpret_shape(VALUE rb_shape, const int dimension)
{
  Check_Type(rb_shape, T_ARRAY);

  const int *shape = new const int[dimension];
  VALUE *arr = RARRAY_PTR(rb_shape);

  for (int i = 0; i < RARRAY_LEN(rb_shape); ++i) {
    shape[i] = FIX2INT(arr[i]);
  }

  return shape;
}

static size_t interpret_size(int* shape, int dimension)
{
  size_t size = 1;

  for (int i = 0; i < dimension; ++i) {
    size *= shape[i]  
  }

  return size;
}

static int interpret_direction(VALUE rb_direction)
{
  switch(SYM2ID(rb_direction))
  {
    case rb_intern("forward"):
      return FFTW_FORWARD;
    case rb_intern("backward"):
      return FFTW_BACKWARD;
    default:
  }
}

static unsigned interpret_flag(VALUE rb_flag)
{

}

static VALUE nm_fftw_create_plan(VALUE self, VALUE rb_shape, 
  VALUE rb_dim, VALUE rb_flag, VALUE rb_direction)
{ 
  fftw_data *data = ALLOC(fftw_data);
  const int dimension = FIX2INT(rb_dim);
  const int* shape      = interpret_shape(rb_shape, dimension);
  size_t size     = interpret_size(shape, dimension);
  int sign        = interpret_direction(rb_direction);
  unsigned flag   = interpret_flag(rb_flag);
  // calculate size of the array to be allocated from the shape and dim
  // figure out the flag from the flag arg
  // figure out direction
  data->input = ALLOC_N(fftw_complex, size);
  data->output = ALLOC_N(fftw_complex, size);
  data->plan = fftw_plan_dft(dimension, shape, data->input, data->output, 
    sign, flag);

  return Data_Wrap_Struct(cNMatrix_FFTW_Plan_Data, NULL, nm_fftw_cleanup, data);
}

static VALUE nm_fftw_set_input(VALUE self, VALUE nmatrix, VALUE plan_data)
{
  fftw_data *data;

  Data_Get_Struct(plan_data, fftw_data, data);
  memcpy(data->input, NM_DENSE_ELEMENTS(nmatrix), 
    sizeof(fftw_complex)*NM_DENSE_COUNT(nmatrix));

  return self;
}

static VALUE nm_fftw_execute(VALUE self, VALUE plan_data, VALUE nmatrix)
{
  fftw_data *data;

  Data_Get_Struct(plan_data, fftw_data, data);
  fftw_execute(data->plan);
  memcpy(NM_DENSE_ELEMENTS(nmatrix), data->output, 
    sizeof(fftw_complex)*NM_DENSE_COUNT(nmatrix));

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
      (METHOD)nm_fftw_create_plan, 4);
    rb_define_private_method(cNMatrix_FFTW_Plan, "__set_input__",
      (METHOD)nm_fftw_set_input, 2);
    rb_define_private_method(cNMatrix_FFTW_Plan, "__execute__",
      (METHOD)nm_fftw_execute, 2);
  }
}
