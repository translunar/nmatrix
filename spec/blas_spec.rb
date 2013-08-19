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
# SciRuby is Copyright (c) 2010 - 2012, Ruby Science Foundation
# NMatrix is Copyright (c) 2012, Ruby Science Foundation
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
# == blas_spec.rb
#
# Tests for properly exposed BLAS functions.
#

# Can we use require_relative here instead?
require File.join(File.dirname(__FILE__), "spec_helper.rb")

describe NMatrix::BLAS do
  [:rational32, :rational64, :rational128, :float32, :float64, :complex64, :complex128].each do |dtype|
    context dtype do
      # This is not the same as "exposes cblas trsm", which would be for a version defined in blas.rb (which
      # would greatly simplify the calling of cblas_trsm in terms of arguments, and which would be accessible
      # as NMatrix::BLAS::trsm)
      it "exposes unfriendly cblas_trsm" do
        a     = NMatrix.new(:dense, 3, [4,-1.quo(2), -3.quo(4), -2, 2, -1.quo(4), -4, -2, -1.quo(2)], dtype)
        b     = NVector.new(3, [-1, 17, -9], dtype)
        NMatrix::BLAS::cblas_trsm(:row, :right, :lower, :transpose, :nonunit, 1, 3, 1.0, a, 3, b, 3)

        # These test results all come from actually running a matrix through BLAS. We use them to ensure that NMatrix's
        # version of these functions (for rationals) give similar results.

        b[0].should == -1.quo(4)
        b[1].should == 33.quo(4)
        b[2].should == -13

        NMatrix::BLAS::cblas_trsm(:row, :right, :upper, :transpose, :unit, 1, 3, 1.0, a, 3, b, 3)

        b[0].should == -15.quo(2)
        b[1].should == 5
        b[2].should == -13
      end
    end
  end

  [:rational32,:rational64,:rational128].each do |dtype|
    context dtype do
      it "exposes cblas rot"
    end

    context dtype do
      it "exposes cblas rotg"
    end
  end

  [:float32, :float64, :complex64, :complex128, :object].each do |dtype|
    context dtype do

      it "exposes cblas rot" do
        x = NVector.new(5, [1,2,3,4,5], dtype)
        y = NVector.new(5, [-5,-4,-3,-2,-1], dtype)
        x, y = NMatrix::BLAS::rot(x, y, 1.quo(2), Math.sqrt(3).quo(2), -1)

        x[0].should be_within(1e-4).of(-0.3660254037844386)
        x[1].should be_within(1e-4).of(-0.7320508075688772)
        x[2].should be_within(1e-4).of(-1.098076211353316)
        x[3].should be_within(1e-4).of(-1.4641016151377544)
        x[4].should be_within(1e-4).of(-1.8301270189221928)

        y[0].should be_within(1e-4).of(-6.830127018922193)
        y[1].should be_within(1e-4).of(-5.464101615137754)
        y[2].should be_within(1e-4).of(-4.098076211353316)
        y[3].should be_within(1e-4).of(-2.732050807568877)
        y[4].should be_within(1e-4).of(-1.3660254037844386)
      end

    end
  end

  [:float32, :float64, :complex64, :complex128, :object].each do |dtype|
    context dtype do

      it "exposes cblas rotg" do
        pending("broken for :object") if dtype == :object
        ab = NVector.new(2, [6,-8], dtype)
        c,s = NMatrix::BLAS::rotg(ab)

        if [:float32, :float64].include?(dtype)
          ab[0].should be_within(1e-6).of(-10)
          ab[1].should be_within(1e-6).of(-5.quo(3))
          c.should be_within(1e-6).of(-3.quo(5))
        else
          ab[0].should be_within(1e-6).of(10)
          ab[1].should be_within(1e-6).of(5.quo(3))
          c.should be_within(1e-6).of(3.quo(5))
        end
        s.should be_within(1e-6).of(4.quo(5))
      end

      # Note: this exposes gemm, not cblas_gemm (which is the unfriendly CBLAS no-error-checking version)
      it "exposes gemm" do
        #STDERR.puts "dtype=#{dtype.to_s}"
        #STDERR.puts "1"
        n = NMatrix.new([4,3], dtype)
        n[0,0] = 14.0
        n[0,1] = 9.0
        n[0,2] = 3.0
        n[1,0] = 2.0
        n[1,1] = 11.0
        n[1,2] = 15.0
        n[2,0] = 0.0
        n[2,1] = 12.0
        n[2,2] = 17.0
        n[3,0] = 5.0
        n[3,1] = 2.0
        n[3,2] = 3.0

        m = NMatrix.new([3,2], dtype)

        m[0,0] = 12.0
        m[0,1] = 25.0
        m[1,0] = 9.0
        m[1,1] = 10.0
        m[2,0] = 8.0
        m[2,1] = 5.0

        #c = NMatrix.new([4,2], dtype)
        r = NMatrix::BLAS.gemm(n, m) #, c)
        #c.should equal(r) # check that both are same memory address

        r[0,0].should == 273.0
        r[0,1].should == 455.0
        r[1,0].should == 243.0
        r[1,1].should == 235.0
        r[2,0].should == 244.0
        r[2,1].should == 205.0
        r[3,0].should == 102.0
        r[3,1].should == 160.0
      end


      it "exposes gemv" do
        #a = NMatrix.random(3)
        #x = NVector.random(3)
        a = NMatrix.new([4,3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], :float64)
        x = NVector.new(3, [2.0, 1.0, 0.0], :float64)

        NMatrix::BLAS.gemv(a, x)
      end

      it "exposes asum" do
        x = NVector.new(4, [1,2,3,4], :float64)
        NMatrix::BLAS.asum(x).should == 10.0
      end


      it "exposes nrm2" do
        x = NVector.new(4, [2,-4,3,5], :float64)
        NMatrix::BLAS.nrm2(x, 1, 3).should be_within(1e-10).of(5.385164807134504)
      end

    end
  end
end