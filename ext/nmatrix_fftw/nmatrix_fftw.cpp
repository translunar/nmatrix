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
#include "data/complex.h"
#include "data/data.h"
#include "storage/storage.h"

#include <iostream>
using namespace std;

void nm_fftw_delete(NMATRIX* mat) {
  static void (*ttable[nm::NUM_STYPES])(STORAGE*) = {
    nm_dense_storage_delete,
    nm_list_storage_delete,
    nm_yale_storage_delete
  };
  ttable[mat->stype](mat->storage);

  fftw_free(mat);
}

/*
 * Create an nmatrix. Used by NMatrix extensions for creating NMatrix objects
 * in C. This function offers greater control over data than rb_nmatrix_dense_create
 * and does not force copying of data.
 *
 * Returns a properly-wrapped Ruby object as a VALUE.
 */
VALUE nm_fftw_create_nmatrix(
  nm::dtype_t dtype, size_t* shape, size_t dim, void* elements, size_t length) 
{
  NMATRIX* nm;
  size_t nm_dim;
  size_t* shape_copy;

  // Do not allow a dim of 1. Treat it as a column or row matrix.
  if (dim == 1) {
    nm_dim        = 2;
    shape_copy    = NM_ALLOC_N(size_t, nm_dim);
    shape_copy[0] = shape[0];
    shape_copy[1] = 1;

  } else {
    nm_dim      = dim;
    shape_copy  = NM_ALLOC_N(size_t, nm_dim);
    memcpy(shape_copy, shape, sizeof(size_t)*nm_dim);
  }

  // allocate and create the matrix and its storage
  nm = nm_create(nm::DENSE_STORE, 
    nm_dense_storage_create(dtype, shape_copy, dim, elements, length));

  nm_register_nmatrix(nm);
  VALUE to_return = Data_Wrap_Struct(cNMatrix, nm_mark, nm_fftw_delete, nm);
  nm_unregister_nmatrix(nm);

  // tell Ruby about the matrix and its storage, particularly how to garbage collect it.
  return to_return;
}

VALUE nm_fftw_create_plan(VALUE self)
{
  // accept the dimension and shape of input and output.
  // allocate fftw_complex arrays of the relevant length.
  // create the plan and store it in a Ruby object with Data_Wrap_Struct.
  // create an nmatrix with the relevant typecasting for input and output and store
  // in instance variables @input and @output.
  // return the wrapped plan object.
}

extern "C" {
  void Init_nmatrix_fftw() 
  {
    VALUE cNMatrix = rb_define_class("NMatrix", rb_cObject);
    VALUE cNMatrix_FFTW = rb_define_module_under(cNMatrix, "FFTW");
    VALUE cNMatrix_FFTW_Plan = rb_define_class_under(cNMatrix_FFTW, "Plan", rb_cObject);

    rb_define_private_method(cNMatrix_FFTW_Plan, "_create_plan_", 
      (METHOD)nm_fftw_create_plan, 0);
  }
}
