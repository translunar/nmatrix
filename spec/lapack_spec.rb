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
# Tests for LAPACK functions that have internal implementations (i.e. they
# don't rely on external libraries). These tests will also be run for the
# plugins that do use external libraries, since they will override the
# internal implmentations.
#

require 'spec_helper'

describe "NMatrix::LAPACK functions with internal implementations" do
  # where integer math is allowed
  [:byte, :int8, :int16, :int32, :int64, :float32, :float64, :complex64, :complex128].each do |dtype|
    context dtype do
      # This spec seems a little weird. It looks like laswp ignores the last
      # element of piv, though maybe I misunderstand smth. It would make
      # more sense if piv were [2,1,3,3]
      it "exposes clapack laswp" do
        a = NMatrix.new(:dense, [3,4], [1,2,3,4,5,6,7,8,9,10,11,12], dtype)
        NMatrix::LAPACK::clapack_laswp(3, a, 4, 0, 3, [2,1,3,0], 1)
        b = NMatrix.new(:dense, [3,4], [3,2,4,1,7,6,8,5,11,10,12,9], dtype)
        expect(a).to eq(b)
      end

      # This spec is OK, because the default behavior for permute_columns
      # is :intuitive, which is different from :lapack (default laswp behavior)
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
  [:float32, :float64, :complex64, :complex128].each do |dtype|
    context dtype do

      #Spec OK
      it "exposes clapack_gesv" do
        a = NMatrix[[1.0, 2, 3], [0,1.0/2,4],[3,3,9]].cast(dtype: dtype)
        b = NMatrix[[1.0],[2],[3]].cast(dtype: dtype)
        err = case dtype
                when :float32, :complex64
                  1e-6
                when :float64, :complex128
                  1e-8
                else
                  1e-64
              end
        expect(NMatrix::LAPACK::clapack_gesv(:row,a.shape[0],b.shape[1],a,a.shape[0],b,b.shape[0])).to be_within(err).of(NMatrix[[-1.0/2], [0], [1.0/2]].cast(dtype: dtype))
      end


      # spec OK, but tricky. It's the upper matrix that has unit diagonals and the permutation is done in columns not rows. See the code details. This is normal for CLAPACK with row-major matrices, but not for plain LAPACK.
      it "exposes clapack_getrf" do
        a = NMatrix.new(3, [4,9,2,3,5,7,8,1,6], dtype: dtype)
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

        expect(a[0,0]).to eq(9)
        expect(a[0,1]).to be_within(err).of(2.0/9)
        expect(a[0,2]).to be_within(err).of(4.0/9)
        expect(a[1,0]).to eq(5)
        expect(a[1,1]).to be_within(err).of(53.0/9)
        expect(a[1,2]).to be_within(err).of(7.0/53)
        expect(a[2,0]).to eq(1)
        expect(a[2,1]).to be_within(err).of(52.0/9)
        expect(a[2,2]).to be_within(err).of(360.0/53)
      end

      # this spec is kind of weird. potrf is supposed to decompose a symmetric
      # positive-definite matrix. The matrix tested below isn't symmetric.
      # But this may not be technically wrong, since potrf just examines the
      # upper/lower half (as requested) of the matrix and assumes it is symmetric.
      # I haven't actually checked that this spec is right.
      # Also, we don't have an internal implementation of this, so maybe it doesn't
      # have to be shared
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
      # Spec OK
      it "exposes clapack_getrs" do
        a     = NMatrix.new(3, [-2,4,-3,3,-2,1,0,-4,3], dtype: dtype)
        ipiv  = NMatrix::LAPACK::clapack_getrf(:row, 3, 3, a, 3)
        b     = NMatrix.new([3,1], [-1, 17, -9], dtype: dtype)

        NMatrix::LAPACK::clapack_getrs(:row, false, 3, 1, a, 3, ipiv, b, 3)

        expect(b[0]).to eq(5)
        expect(b[1]).to eq(-15.0/2)
        expect(b[2]).to eq(-13)
      end

      #spec OK, but getri is only implemented by the atlas plugin, so maybe this one doesn't have to be shared
      it "exposes clapack_getri" do
        a = NMatrix.new(:dense, 3, [1,0,4,1,1,6,-3,0,-10], dtype)
        ipiv = NMatrix::LAPACK::clapack_getrf(:row, 3, 3, a, 3) # get pivot from getrf, use for getri

        begin
          NMatrix::LAPACK::clapack_getri(:row, 3, a, 3, ipiv)

          b = NMatrix.new(:dense, 3, [-5,0,-2,-4,1,-1,1.5,0,0.5], dtype)
          expect(a).to eq(b)
        rescue NotImplementedError => e
          pending e.to_s
        end
      end
    end
  end
end
