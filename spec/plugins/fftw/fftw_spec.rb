# = NMatrix
#
# A linear algebra library for scientific computation in Ruby.
# NMatrix is part of SciRuby.
#
# NMatrix was originally inspired by and derived from NArray, by
# Masahiro Tanaka: http://narray.rubyforge.org
#
# == Copyright Information
#
# SciRuby is Copyright (c) 2010 - 2014, Ruby Science Foundation
# NMatrix is Copyright (c) 2012 - 2014, John Woods and the Ruby Science Foundation
#
# Please see LICENSE.txt for additional copyright notices.
#
# == Contributing
#
# By contributing source code to SciRuby, you agree to be bound by
# our Contributor Agreement:
#
# * https://github.com/SciRuby/sciruby/wiki/Contributor-Agreement
#
# == fftw_spec.rb
#
# Tests for interfaces that are only exposed by nmatrix-fftw
#

require 'spec_helper'
require "./lib/nmatrix/fftw"

describe NMatrix::FFTW, focus: true do
  context ".r2c_one" do
    it "computes correct FFTW" do
      n = NMatrix.new([4], [3.10, 1.73, 1.04, 2.83])
      complex = NMatrix.zeros([3], dtype: :complex128)
      NMatrix::FFTW.r2c_one(n, complex)
      # Expected results obtained from running SciPy's fft on the same Array
      # However, FFTW only computes the first half + 1 element

      exp = NMatrix.new([3], [Complex(8.70, 0), Complex(2.06, 1.1), Complex(-0.42, 0)])
      expect(complex).to eq(exp)
    end

    # TODO: Is this test really needed? Moreover, why should the output nmatrix
    # have shape equal to input nmatrix if FFTW computes only half + 1 elements?
    it "Checks NMatrix in FFTW.r2c_one and input NMatrix have same value for shape" do
      n = NMatrix.new([6], [-3.10,
                            -1.73,
                             1.0,
                             2.84,
                             56.42,
                             -32.1])
      complex = NMatrix.zeros([6], dtype: :complex128)
      fftw = NMatrix::FFTW.r2c_one(n, complex)
      expect(n.shape).to eq(fftw.shape)
    end

    it "Checks NMatrix in FFTW.r2c_one and input NMatrix have same value for size" do
      n = NMatrix.new([6], [-3.10,
                            -1.73,
                             1.0,
                             2.84,
                             56.42,
                             -32.1])
      complex = NMatrix.zeros([6], dtype: :complex128)
      fftw = NMatrix::FFTW.r2c_one(n, complex)
      expect(n.size).to eq(fftw.size)
    end
  end
end
