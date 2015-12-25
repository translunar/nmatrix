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

#define TYPE_COMPLEX_COMPLEX 0
#define TYPE_REAL_COMPLEX 1
#define TYPE_COMPLEX_REAL 2
#define TYPE_REAL_REAL 3

static VALUE cNMatrix_FFTW_Plan_Data;

struct fftw_data {
  void* input;
  void* output;
  fftw_plan plan;
};

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

template <typename InputType, typename OutputType>
static void nm_fftw_actually_create_plan(fftw_data* data, 
  size_t input_size, size_t output_size, const int dimensions, const int* shape, 
  int sign, unsigned flags, VALUE rb_type)
{
  data->input  = ALLOC_N(InputType, input_size);
  data->output = ALLOC_N(OutputType, output_size);
  switch (FIX2INT(rb_type))
  {
    case TYPE_COMPLEX_COMPLEX:
      data->plan = fftw_plan_dft(dimensions, shape, (fftw_complex*)data->input, 
        (fftw_complex*)data->output, sign, flags);
      break;
    case TYPE_REAL_COMPLEX:
      data->plan = fftw_plan_dft_r2c(dimensions, shape, (double*)data->input, 
        (fftw_complex*)data->output, flags);
      break;
    case TYPE_COMPLEX_REAL:
      // pending
      break;
    case TYPE_REAL_REAL:
      // pending
      break;
    default:
      rb_raise(rb_eArgError, "Invalid type of DFT.");
  }
}

static VALUE nm_fftw_create_plan(VALUE self, VALUE rb_shape, VALUE rb_size,
  VALUE rb_dim, VALUE rb_flags, VALUE rb_direction, VALUE rb_type)
{ 

  const int dimensions = FIX2INT(rb_dim);
  const int* shape     = interpret_shape(rb_shape, dimensions);
  size_t size          = FIX2INT(rb_size);
  int sign             = FIX2INT(rb_direction);
  unsigned flags       = FIX2INT(rb_flags);
  fftw_data *data      = ALLOC(fftw_data);

  switch (FIX2INT(rb_type))
  {
    case TYPE_COMPLEX_COMPLEX:
      nm_fftw_actually_create_plan <fftw_complex,fftw_complex>(data, 
        size, size, dimensions, shape, sign, flags, rb_type);
      break;
    case TYPE_REAL_COMPLEX:
      nm_fftw_actually_create_plan <double, fftw_complex>(data, 
        size, size/2 + 1, dimensions, shape, sign, flags, rb_type);
      break;
    case TYPE_COMPLEX_REAL:
      // pending
      break;
    case TYPE_REAL_REAL:
      // pending
      break;
    default:
      rb_raise(rb_eArgError, "Invalid type of DFT.");
  }

  return Data_Wrap_Struct(cNMatrix_FFTW_Plan_Data, NULL, nm_fftw_cleanup, data);
}

template <typename InputType>
static void set(VALUE nmatrix, VALUE plan_data)
{
  fftw_data* data;
  Data_Get_Struct(plan_data, fftw_data, data);
  memcpy((InputType*)data->input, NM_DENSE_ELEMENTS(nmatrix), 
    sizeof(InputType)*NM_DENSE_COUNT(nmatrix));
}

static VALUE nm_fftw_set_input(VALUE self, VALUE nmatrix, VALUE plan_data, 
  VALUE type)
{
  switch(FIX2INT(type))
  {
    case TYPE_COMPLEX_COMPLEX:
    case TYPE_COMPLEX_REAL:
      set<fftw_complex>(nmatrix, plan_data);
      break;
    case TYPE_REAL_COMPLEX:
    case TYPE_REAL_REAL:
      set<double>(nmatrix, plan_data);
      break;
    default:
      rb_raise(rb_eArgError, "Invalid type of DFT.");
  }

  return self;
}

template <typename OutputType>
static void execute(VALUE nmatrix, VALUE plan_data)
{
  fftw_data *data;
  Data_Get_Struct(plan_data, fftw_data, data);
  fftw_execute(data->plan);
  memcpy(NM_DENSE_ELEMENTS(nmatrix), data->output, 
    sizeof(OutputType)*NM_DENSE_COUNT(nmatrix));
}

static VALUE nm_fftw_execute(VALUE self, VALUE plan_data, VALUE nmatrix, VALUE type)
{
  switch(FIX2INT(type))
  {
    case TYPE_COMPLEX_COMPLEX:
    case TYPE_REAL_COMPLEX:
      execute<fftw_complex>(nmatrix, plan_data);
      break;
    case TYPE_COMPLEX_REAL:
    case TYPE_REAL_REAL:
      execute<double>(nmatrix, plan_data);
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
