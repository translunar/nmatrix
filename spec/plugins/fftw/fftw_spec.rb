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
  describe NMatrix::FFTW::Plan do
    context ".new" do
      it "creates a new plan for default DFT (complex input/complex output)" do
        plan = NMatrix::FFTW::Plan.new(10)

        expect(plan.input.class).to eq(NMatrix)
        expect(plan.output.class).to eq(NMatrix)
        expect(plan.input.size).to eq(10)
        expect(plan.output.size).to eq(10)
      end

      it "creates a new plan for multi dimensional DFT with options" do
        plan = NMatrix::FFTW::Plan.new([10,5,8], 
          direction: :forward, flag: :estimate, rank: 3)


      end

      it "creates a new plan for real input/complex output" do
        pending "implement option :type => :r2c"
      end

      it "creates a new plan for real input/real output" do
        pending "implement option :type => :r2r"
      end

      it "creates a new plan for complex input/real output" do
        pending "implement option :type => :c2r"
      end
    end
  end

  context ".r2c_one" do
    it "computes correct FFT" do
      n = NMatrix.new([4], [3.10, 1.73, 1.04, 2.83])
      complex = NMatrix.zeros([3], dtype: :complex128)
      NMatrix::FFTW.r2c_one(n, complex)
      # Expected results obtained from running SciPy's fft on the same Array
      # However, FFTW only computes the first half + 1 element

      exp = NMatrix.new([3], [Complex(8.70, 0), Complex(2.06, 1.1), Complex(-0.42, 0)])
      expect(complex).to eq(exp)
    end
  end
end
