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
// SciRuby is Copyright (c) 2010 - 2013, Ruby Science Foundation
// NMatrix is Copyright (c) 2013, Ruby Science Foundation
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
// == enumerator.cpp
//
// n-dimensional enumerator class EnumeratorN, based on Ruby's
// enumerator.c.


// Jacked verbatim from ruby: enumerator.c.
static VALUE
enumerator_block_call(VALUE obj, rb_block_call_func *func, VALUE arg)
{
  int argc = 0;
  VALUE *argv = 0;
  const struct enumerator *e = enumerator_ptr(obj);
  ID meth = e->meth;

  if (e->args) {
    argc = RARRAY_LENINT(e->args);
    argv = RARRAY_PTR(e->args);
  }
  return rb_block_call(e->obj, meth, argc, argv, func, arg);
}


static VALUE enumeratorn_with_indices_ijk(VALUE val, VALUE m, int argc, VALUE* argv) {
  VALUE* memo = (VALUE *)m;
  VALUE  indices = *memo;

  // Increment indices
  for (size_t idx = 0; idx < NM_DIM(val); ++idx) {
    // Ensure that all index array values are integers
    rb_ary_set(memo, idx, rb_to_int( rb_ary_get(memo, idx) ));
  }
}


/*
 * call-seq:
 *   e.with_indices(offsets = [0, ..., 0]) {|(*args), indices| ... }
 *   e.with_indices(offsets = [0, ..., 0])
 *
 * Iterates the given block for each element with indices, which
 * starts from positions given in +offset+. If no block is given,
 * returns a new EnumeratorN that includes the indices, starting from
 * +offsets+.
 *
 * +offsets+:: the starting indices to use
 *
 */
static VALUE enumeratorn_with_indices(int argc, VALUE* argv, VALUE obj) {
  VALUE memo;

  rb_scan_args(argc, argv, "01", &memo);
  RETURN_SIZED_ENUMERATOR(obj, argc, argv, enumeratorn_size);
  if (NIP_P(memo)) { // If no offsets given, start at i=0,j=0,...

    memo = rb_ary_new2(NM_DIM(obj));
    for (size_t idx = 0; idx < NM_DIM(obj); ++idx) {
      rb_ary_push(memo, INT2FIX(0));
    }

  } else {

    for (size_t idx = 0; idx < NM_DIM(obj); ++idx) {
      // Ensure that all index array values are integers
      rb_ary_set(memo, idx, rb_to_int( rb_ary_get(memo, idx) ));
    }

  }

  return enumeratorn_block_call(obj, enumeratorn_with_indices_ijk, (VALUE)&memo);
}


static VALUE enumeratorn_each_with_indices(VALUE obj) {
  return enumeratorn_with_indices(0, NULL, obj);
}


void Init_EnumeratorN(void) {
  rb_define_method(rb_mKernel, "to_enumn", obj_to_enumn, -1);
  rb_define_method(rb_mKernel, "enumn_for", obj_to_enumn, -1);

  cEnumeratorN = rb_define_class("EnumeratorN", rb_cObject);
  rb_include_module(cEnumeratorN, cEnumerableN);

  rb_define_method(cEnumeratorN, "each_with_indices", enumeratorn_each_with_indices, 0);
  rb_define_method(cEnumeratorN, "with_indices", enumeratorn_with_indices, -1);
}