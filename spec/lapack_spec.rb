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

      it "exposes geev" do
        pending("temporarily disable WL 2015-06-08")
        pending("needs rational implementation") if dtype.to_s =~ /rational/
        ary = %w|-1.01 0.86 -4.60 3.31 -4.81
                     3.98 0.53 -7.04 5.29 3.55
                     3.30 8.26 -3.89 8.20 -1.51
                     4.43 4.96 -7.66 -7.33 6.18
                     7.31 -6.43 -6.16 2.47 5.58|
        ary = dtype.to_s =~ /complex/ ? ary.map(&:to_c) : ary.map(&:to_f)

        a   = NMatrix.new(:dense, 5, ary, dtype).transpose
        lda = 5
        n   = 5

        wr  = NMatrix.new(:dense, [n,1], 0, dtype)
        wi  = dtype.to_s =~ /complex/ ? nil : NMatrix.new(:dense, [n,1], 0, dtype)
        vl  = NMatrix.new(:dense, n, 0, dtype)
        vr  = NMatrix.new(:dense, n, 0, dtype)
        ldvr = n
        ldvl = n

        info = NMatrix::LAPACK::lapack_geev(:left, :right, n, a.clone, lda, wr.clone, wi.nil? ? nil : wi.clone, vl.clone, ldvl, vr.clone, ldvr, -1)
        expect(info).to eq(0)

        info = NMatrix::LAPACK::lapack_geev(:left, :right, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, 2*n)

        # Negate these and we get a correct result:
        vr = vr.transpose
        vl = vl.transpose

        pending("Need complex example") if dtype.to_s =~ /complex/
        vl_true = NMatrix.new(:dense, 5, [0.04,  0.29,  0.13,  0.33, -0.04,
                                          0.62,  0.0,  -0.69,  0.0,  -0.56,
                                         -0.04, -0.58,  0.39,  0.07,  0.13,
                                          0.28,  0.01,  0.02,  0.19,  0.80,
                                         -0.04,  0.34,  0.40, -0.22, -0.18 ], :float64)

        expect(vl.abs).to be_within(1e-2).of(vl_true.abs)
        # Not checking vr_true.
        # Example from:
        # http://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/lapacke_dgeev_row.c.htm
        #
        # This is what the result should look like:
        # [
        #  [0.10806497186422348,  0.16864821314811707,   0.7322341203689575,                  0.0, -0.46064677834510803]
        #  [0.40631288290023804, -0.25900983810424805, -0.02646319754421711, -0.01694658398628235, -0.33770373463630676]
        #  [0.10235744714736938,  -0.5088024139404297,  0.19164878129959106, -0.29256555438041687,  -0.3087439239025116]
        #  [0.39863115549087524,  -0.0913335531949997, -0.07901126891374588, -0.07807594537734985,   0.7438457012176514]
        #  [ 0.5395349860191345,                  0.0, -0.29160499572753906, -0.49310219287872314, -0.15852922201156616]
        # ]
        #

      end
    end
  end
end
