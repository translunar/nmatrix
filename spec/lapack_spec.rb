        require 'pry'
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
# == lapack_spec.rb
#
# Tests for properly exposed LAPACK functions.
#

# Can we use require_relative here instead?
require File.join(File.dirname(__FILE__), "spec_helper.rb")

describe NMatrix::LAPACK do
  # where integer math is allowed
  [:byte, :int8, :int16, :int32, :int64, :rational32, :rational64, :rational128, :float32, :float64, :complex64, :complex128].each do |dtype|
    context dtype do
      it "exposes clapack laswp" do
        a = NMatrix.new(:dense, [3,4], [1,2,3,4,5,6,7,8,9,10,11,12], dtype)
        NMatrix::LAPACK::clapack_laswp(3, a, 4, 0, 3, [2,1,3,0], 1)
        b = NMatrix.new(:dense, [3,4], [3,2,4,1,7,6,8,5,11,10,12,9], dtype)
        a.should == b
      end
    end
  end

  # where integer math is not allowed
  [:rational32, :rational64, :rational128, :float32, :float64, :complex64, :complex128].each do |dtype|
    context dtype do
      it "exposes clapack getrf" do
        a = NMatrix.new(:dense, 3, [4,9,2,3,5,7,8,1,6], dtype)
        NMatrix::LAPACK::clapack_getrf(:row, 3, 3, a, 3)
        # delta varies for different dtypes
        err = case dtype
                when :float32, :complex64
                  1e-6
                when :float64, :complex128
                  1e-15
                else
                  1e-64 # FIXME: should be 0, but be_within(0) does not work.
              end

        a[0,0].should == 9 # 8
        a[0,1].should be_within(err).of(2.quo(9)) # 1
        a[0,2].should be_within(err).of(4.quo(9)) # 6
        a[1,0].should == 5 # 1.quo(2)
        a[1,1].should be_within(err).of(53.quo(9)) # 17.quo(2)
        a[1,2].should be_within(err).of(7.quo(53)) # -1
        a[2,0].should == 1 # 3.quo(8)
        a[2,1].should be_within(err).of(52.quo(9))
        a[2,2].should be_within(err).of(360.quo(53))
        # FIXME: these are rounded, == won't work
          # be_within(TOLERANCE).of(desired_value) should work
        a[2,1].should be_within(err).of(0.544118)
        a[2,2].should be_within(err).of(5.294118)
      end

      it "exposes clapack potrf" do
        # first do upper
        begin
        a = NMatrix.new(:dense, 3, [25,15,-5, 0,18,0, 0,0,11], dtype)
        NMatrix::LAPACK::clapack_potrf(:row, :upper, 3, a, 3)
        b = NMatrix.new(:dense, 3, [5,3,-1, 0,3,1, 0,0,3], dtype)
        a.should == b
        rescue NotImplementedError => e
          pending e.to_s
        end

        # then do lower
        a = NMatrix.new(:dense, 3, [25,0,0, 15,18,0,-5,0,11], dtype)
        NMatrix::LAPACK::clapack_potrf(:row, :lower, 3, a, 3)
        b = NMatrix.new(:dense, 3, [5,0,0, 3,3,0, -1,1,3], dtype)
        a.should == b
      end

      # Together, these calls are basically xGESV from LAPACK: http://www.netlib.org/lapack/double/dgesv.f
      it "exposes clapack getrs" do
        a     = NMatrix.new(:dense, 3, [-2,4,-3,3,-2,1,0,-4,3], dtype)
        ipiv  = NMatrix::LAPACK::clapack_getrf(:row, 3, 3, a, 3)
        b     = NVector.new(3, [-1, 17, -9], dtype)

        NMatrix::LAPACK::clapack_getrs(:row, false, 3, 1, a, 3, ipiv, b, 3)

        b[0].should == 5
        b[1].should == -15.quo(2)
        b[2].should == -13
      end

      it "exposes clapack getri" do
        a = NMatrix.new(:dense, 3, [1,0,4,1,1,6,-3,0,-10], dtype)
        ipiv = NMatrix::LAPACK::clapack_getrf(:row, 3, 3, a, 3) # get pivot from getrf, use for getri
        begin
        NMatrix::LAPACK::clapack_getri(:row, 3, a, 3, ipiv)

        b = NMatrix.new(:dense, 3, [-5,0,-2,-4,1,-1,1.5,0,0.5], dtype)
        a.should == b
        rescue NotImplementedError => e
          pending e.to_s
        end
      end
      it "exposes gesvd and gesdd via #svd" do 
        # http://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/dgesvd_ex.c.htm
        a = NMatrix.new([5,6], 
          %w|8.79 9.93 9.83 5.45 3.16
            6.11 6.91 5.04 -0.27 7.98 
            -9.15 -7.93 4.86 4.85 3.01 
            9.57 1.64 8.83 0.74 5.80 
            -3.49 4.02 9.80 10.00 4.27 
            9.84 0.15 -8.99 -6.02 -5.31|.map(&:to_f), 
          dtype)
        s_true = NMatrix.new([1,5], [27.47, 22.64, 8.56, 5.99, 2.01], dtype)
        s_true = NMatrix.new([1,5], [27.46873241822185, 22.643185009774704, 8.558388228482581, 5.985723201512133, 2.014899658715756], dtype)
        left_true = NMatrix.new([5,6], 
          %w|-0.59 0.26   0.36   0.31   0.23
            -0.40   0.24  -0.22  -0.75  -0.36
            -0.03  -0.60  -0.45   0.23  -0.31
            -0.43   0.24  -0.69   0.33   0.16
            -0.47  -0.35   0.39   0.16  -0.52
             0.29   0.58  -0.02   0.38  -0.65|.map(&:to_f), 
           dtype)
        right_true = NMatrix.new([5,5], 
          %w|-0.25  -0.40  -0.69  -0.37  -0.41
             0.81   0.36  -0.25  -0.37  -0.10
            -0.26   0.70  -0.22   0.39  -0.49
             0.40  -0.45   0.25   0.43  -0.62
            -0.22   0.14   0.59  -0.63  -0.44|.map(&:to_f),
          dtype)
        err = case dtype
              when :float32, :complex64
                1e-6
              when :float64, :complex128
                1e-15
              else
                1e-64 # FIXME: should be 0, but be_within(0) does not work.
              end
        response = NMatrix::LAPACK.svd(a, :arrays)
        if response.is_a? Array
          sing_vals, left_vals, right_vals = response
          left_vals.row(0).to_a.zip(left_true.row(0).to_a).each do |a_val, t_val|
            a_val.should be_within(err).of(t_val)
          end
          right_vals.row(0).to_a.zip(right_true.row(0).to_a).each do |a_val, t_val|
            a_val.should be_within(err).of(t_val)
          end
        elsif response.is_a? NMatrix
          sing_vals = response
        end
        sing_vals.row(0).to_a.zip(s_true.row(0).to_a).each do |answer_val, truth_val|
          answer_val.should be_within(err).of(truth_val)
        end
      end
    end
  end
end
