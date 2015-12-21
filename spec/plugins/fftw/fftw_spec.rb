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

        plan.destroy
      end

      it "creates a new plan for multi dimensional DFT with options" do
        plan = NMatrix::FFTW::Plan.new([10,5,8], 
          direction: :forward, flag: :estimate, dim: 3)

        expect(plan.input.class).to eq(NMatrix)
        expect(plan.output.class).to eq(NMatrix)
        expect(plan.input.size).to eq(10*5*8)
        expect(plan.output.size).to eq(10*5*8)
        expect(plan.dim).to eq(3)

        plan.destroy
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

    context "#set_input" do
      it "accepts nothing but complex128 input for the default plan" do
        plan = NMatrix::FFTW::Plan.new(4)
        input = NMatrix.new([4], [23.54,52.34,52.345,64], dtype: :float64)

        expect {
          plan.set_input(input)
        }.to raise_error(ArgumentError)
      end
    end

    context "#destroy" do
      it "destroys the plan" do
        plan = NMatrix::FFTW::Plan.new(10)
        input = NMatrix.new([10], [5]*10, dtype: :complex128)
        plan.set_input input
        plan.execute
        plan.destroy

        expect(plan.execute).to eq(false)

        expect {
          plan.input
        }.to raise_error(FFTWPlanDestroyedError)

        expect {
          plan.output
        }.to raise_error(FFTWPlanDestroyedError)
      end
    end

    context "#execute" do
      it "calculates a basic 1D DFT" do
        input = NMatrix.new([10],
          [
            Complex(9.32,0),
            Complex(44,0),
            Complex(125,0),
            Complex(34,0),
            Complex(31,0),
            Complex(44,0),
            Complex(12,0),
            Complex(1,0),
            Complex(53.23,0),
            Complex(-23.23,0),
          ], dtype: :complex128)

        output = NMatrix.new([10],
          [
            Complex(330.3200,0.0000),
            Complex(-8.4039  ,150.3269),
            Complex(-99.4807 , 68.6579),
            Complex(-143.6861, 20.4273),
            Complex(67.6207  ,  8.5236),
            Complex(130.7800 ,  0.0000),
            Complex(67.6207  ,  8.5236),
            Complex(-143.6861, 20.4273),
            Complex(-99.4807 , 68.6579),
            Complex(-8.4039  ,150.3269)
          ], dtype: :complex128)

        plan = NMatrix::FFTW::Plan.new(10)
        plan.set_input input
        expect(plan.execute).to eq(true)
        expect(plan.output).to eq(output)
      end

      it "calculates 2D DFT with options" do
        input = NMatrix.new([2,2],
          [
            Complex(9.3200,0), Complex(43.0000,0),
            Complex(3.2000,0), Complex(4.0000,0)
          ], dtype: :complex128
        )

        output = NMatrix.new([2,2],
          [
            Complex(59.520,0), Complex(-34.480,0),
            Complex(45.120,0),  Complex(-32.880,0),
          ], dtype: :complex128
        )

        plan = NMatrix::FFTW::Plan.new([2,2],
          direction: :forward, flag: :estimate, dimension: 2)
        plan.set_input input
        expect(plan.execute).to eq(true)
        expect(plan.output).to eq(output)

        plan.destroy
      end

      it "calculates ND DFT with options" do

      end
    end
  end

  context ".compute" do
    it "provides a DSL for neatly computing FFTs" do
      # DSL should take care of destruction of plan etc.
      pending "complete after rest of functions are done"
    end
  end

  # context ".r2c_one" do
  #   it "computes correct FFT" do
  #     n = NMatrix.new([4], [3.10, 1.73, 1.04, 2.83])
  #     complex = NMatrix.zeros([3], dtype: :complex128)
  #     NMatrix::FFTW.r2c_one(n, complex)
  #     # Expected results obtained from running SciPy's fft on the same Array
  #     # However, FFTW only computes the first half + 1 element

  #     exp = NMatrix.new([3], [Complex(8.70, 0), Complex(2.06, 1.1), Complex(-0.42, 0)])
  #     expect(complex).to eq(exp)
  #   end
  # end
end
