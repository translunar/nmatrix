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
# == lapack_spec.rb
#
# Tests for properly exposed LAPACK functions.
#

require 'spec_helper'

require 'lapack_shared'

describe "NMatrix::LAPACK internal implementation" do
  include_examples "LAPACK shared"

  # where integer math is allowed
  [:byte, :int8, :int16, :int32, :int64, :rational32, :rational64, :rational128, :float32, :float64, :complex64, :complex128].each do |dtype|
    context dtype do
      it "exposes clapack laswp" do
        a = NMatrix.new(:dense, [3,4], [1,2,3,4,5,6,7,8,9,10,11,12], dtype)
        NMatrix::LAPACK::clapack_laswp(3, a, 4, 0, 3, [2,1,3,0], 1)
        b = NMatrix.new(:dense, [3,4], [3,2,4,1,7,6,8,5,11,10,12,9], dtype)
        expect(a).to eq(b)
      end

      it "exposes NMatrix#permute_columns and #permute_columns! (user-friendly laswp)" do
        a = NMatrix.new(:dense, [3,4], [1,2,3,4,5,6,7,8,9,10,11,12], dtype)
        b = NMatrix.new(:dense, [3,4], [3,2,4,1,7,6,8,5,11,10,12,9], dtype)
        piv = [2,1,3,0]
        r = a.permute_columns(piv)
        expect(r).not_to eq(a)
        expect(r).to eq(b)
        a.permute_columns!(piv)
        expect(a).to eq(b)
      end
    end
  end

  # where integer math is not allowed
  [:rational32, :rational64, :rational128, :float32, :float64, :complex64, :complex128].each do |dtype|
    context dtype do

      it "exposes clapack_gesv" do
        a = NMatrix[[1.quo(1), 2, 3], [0,1.quo(2),4],[3,3,9]].cast(dtype: dtype)
        b = NMatrix[[1.quo(1)],[2],[3]].cast(dtype: dtype)
        err = case dtype
                when :float32, :complex64
                  1e-6
                when :float64, :complex128
                  1e-8
                else
                  1e-64
              end
        expect(NMatrix::LAPACK::clapack_gesv(:row,a.shape[0],b.shape[1],a,a.shape[0],b,b.shape[0])).to be_within(err).of(NMatrix[[-1.quo(2)], [0], [1.quo(2)]].cast(dtype: dtype))
      end

      it "exposes clapack_potrf" do
        # first do upper
        begin
          a = NMatrix.new(:dense, 3, [25,15,-5, 0,18,0, 0,0,11], dtype)
          NMatrix::LAPACK::clapack_potrf(:row, :upper, 3, a, 3)
          b = NMatrix.new(:dense, 3, [5,3,-1, 0,3,1, 0,0,3], dtype)
          expect(a).to eq(b)
        rescue NotImplementedError => e
          pending e.to_s
        end

        # then do lower
        a = NMatrix.new(:dense, 3, [25,0,0, 15,18,0,-5,0,11], dtype)
        NMatrix::LAPACK::clapack_potrf(:row, :lower, 3, a, 3)
        b = NMatrix.new(:dense, 3, [5,0,0, 3,3,0, -1,1,3], dtype)
        expect(a).to eq(b)
      end

      # Together, these calls are basically xGESV from LAPACK: http://www.netlib.org/lapack/double/dgesv.f
      it "exposes clapack_getrs" do
        a     = NMatrix.new(3, [-2,4,-3,3,-2,1,0,-4,3], dtype: dtype)
        ipiv  = NMatrix::LAPACK::clapack_getrf(:row, 3, 3, a, 3)
        b     = NMatrix.new([3,1], [-1, 17, -9], dtype: dtype)

        NMatrix::LAPACK::clapack_getrs(:row, false, 3, 1, a, 3, ipiv, b, 3)

        expect(b[0]).to eq(5)
        expect(b[1]).to eq(-15.quo(2))
        expect(b[2]).to eq(-13)
      end
    end
  end
end
