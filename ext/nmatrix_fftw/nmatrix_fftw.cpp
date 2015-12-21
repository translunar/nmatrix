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

// Arguments:
//  self - NMatrix::FFTW::Plan
VALUE nm_fftw_create_plan(VALUE self, VALUE shape, VALUE input, VALUE output, 
  VALUE dim, VALUE direction, VALUE flag, VALUE type);
{

}


extern "C" {
  void Init_nmatrix_fftw() 
  {
    VALUE cNMatrix = rb_define_class("NMatrix", rb_cObject);
    VALUE cNMatrix_FFTW = rb_define_module_under(cNMatrix, "FFTW");
    VALUE cNMatrix_FFTW_Plan = rb_define_class_under(cNMatrix_FFTW, "Plan", rb_cObject);

    rb_define_private_method(cNMatrix_FFTW_Plan, "_create_plan_", 
      (METHOD)nm_fftw_create_plan, 8);
  }
}
