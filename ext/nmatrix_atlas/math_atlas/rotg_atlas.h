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
// == rotg.h
//
// BLAS rotg function in native C++.
//

/*
 *             Automatically Tuned Linear Algebra Software v3.8.4
 *                    (C) Copyright 1999 R. Clint Whaley
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *   1. Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *   2. Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions, and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *   3. The name of the ATLAS group or the names of its contributers may
 *      not be used to endorse or promote products derived from this
 *      software without specific written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE ATLAS GROUP OR ITS CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef ROTG_ATLAS_H
# define ROTG_ATLAS_H

#include "math/rotg.h"

namespace nm { namespace math { namespace atlas {

template <typename DType>
inline void rotg(DType* a, DType* b, DType* c, DType* s) {
  //call internal implementation if no specialization below
  nm::math::rotg(a, b, c, s);
}

template <>
inline void rotg(float* a, float* b, float* c, float* s) {
  cblas_srotg(a, b, c, s);
}

template <>
inline void rotg(double* a, double* b, double* c, double* s) {
  cblas_drotg(a, b, c, s);
}

template <>
inline void rotg(Complex64* a, Complex64* b, Complex64* c, Complex64* s) {
  cblas_crotg(a, b, c, s);
}

template <>
inline void rotg(Complex128* a, Complex128* b, Complex128* c, Complex128* s) {
  cblas_zrotg(a, b, c, s);
}


template <typename DType>
inline void cblas_rotg(void* a, void* b, void* c, void* s) {
  rotg<DType>(static_cast<DType*>(a), static_cast<DType*>(b), static_cast<DType*>(c), static_cast<DType*>(s));
}


}}} //nm::math::atlas

#endif // ROTG_ATLAS_H
