require 'spec_helper'
require "./lib/nmatrix/lapack"

#should include additional specs now in atlas_spec.rb
describe "NMatrix::LAPACK functions implemented with LAPACKE interface" do
  [:float32, :float64, :complex64, :complex128].each do |dtype|
    context dtype do
      it "exposes lapacke_getrf" do
        a = NMatrix.new(3, [4,9,2,3,5,7,8,1,6], dtype: dtype)
        ipiv = NMatrix::LAPACK::lapacke_getrf(:row, 3, 3, a, 3)

        # delta varies for different dtypes
        err = case dtype
                when :float32, :complex64
                  1e-6
                when :float64, :complex128
                  1e-15
              end

        expect(a[0,0]).to eq(8)
        expect(a[0,1]).to be_within(err).of(1)
        expect(a[0,2]).to be_within(err).of(6)
        expect(a[1,0]).to eq(0.5)
        expect(a[1,1]).to be_within(err).of(8.5)
        expect(a[1,2]).to be_within(err).of(-1)
        expect(a[2,0]).to eq(0.375)
        expect(a[2,1]).to be_within(err).of(37.0/68)
        expect(a[2,2]).to be_within(err).of(90.0/17)

        expect(ipiv[0]).to eq(3)
        expect(ipiv[1]).to eq(3)
        expect(ipiv[2]).to eq(3)
      end

      it "exposes lapacke_getri" do
        a = NMatrix.new(:dense, 3, [1,0,4,1,1,6,-3,0,-10], dtype)
        ipiv = NMatrix::LAPACK::lapacke_getrf(:row, 3, 3, a, 3) # get pivot from getrf, use for getri

        # delta varies for different dtypes
        err = case dtype
                when :float32, :complex64
                  1e-5
                when :float64, :complex128
                  1e-14
              end

        NMatrix::LAPACK::lapacke_getri(:row, 3, a, 3, ipiv)

        b = NMatrix.new(:dense, 3, [-5,0,-2,-4,1,-1,1.5,0,0.5], dtype)
        expect(a).to be_within(err).of(b)
      end

      it "exposes lapacke_getrs with vector solutions" do
        a     = NMatrix.new(3, [-2,4,-3,3,-2,1,0,-4,3], dtype: dtype)
        ipiv  = NMatrix::LAPACK::lapacke_getrf(:row, 3, 3, a, 3)
        b     = NMatrix.new([3,1], [-1, 17, -9], dtype: dtype)

        #be careful! the leading dimenension (lda,ldb) is the number of rows for row-major in LAPACKE. Different from CLAPACK convention!
        NMatrix::LAPACK::lapacke_getrs(:row, false, 3, 1, a, 3, ipiv, b, 1)

        # delta varies for different dtypes
        err = case dtype
                when :float32, :complex64
                  1e-5
                when :float64, :complex128
                  1e-13
              end

        expect(b[0]).to be_within(err).of(5)
        expect(b[1]).to be_within(err).of(-15.0/2)
        expect(b[2]).to be_within(err).of(-13)
      end

      it "exposes lapacke_getrs with matrix solutions" do
        a     = NMatrix.new(3, [-2,4,-3,3,-2,1,0,-4,3], dtype: dtype)
        ipiv  = NMatrix::LAPACK::lapacke_getrf(:row, 3, 3, a, 3)
        b     = NMatrix.new([3,2], [-1, 2, 17, 10, -9, 1], dtype: dtype)

        #be careful! the leading dimenension (lda,ldb) is the number of rows for row-major in LAPACKE. Different from CLAPACK convention!
        NMatrix::LAPACK::lapacke_getrs(:row, false, 3, 2, a, 3, ipiv, b, 2)

        # delta varies for different dtypes
        err = case dtype
                when :float32, :complex64
                  1e-5
                when :float64, :complex128
                  1e-13
              end

        x = NMatrix.new([3,2], [5, -1.5, -7.5, -21.25, -13, -28], dtype: dtype)
        expect(b).to be_within(err).of(x)
      end

      it "exposes lapacke_potrf" do
        # first do upper
        begin
          a = NMatrix.new(:dense, 3, [25,15,-5, 0,18,0, 0,0,11], dtype)
          NMatrix::LAPACK::lapacke_potrf(:row, :upper, 3, a, 3)
          b = NMatrix.new(:dense, 3, [5,3,-1, 0,3,1, 0,0,3], dtype)
          expect(a).to eq(b)
        end

        # then do lower
        a = NMatrix.new(:dense, 3, [25,0,0, 15,18,0,-5,0,11], dtype)
        NMatrix::LAPACK::lapacke_potrf(:row, :lower, 3, a, 3)
        b = NMatrix.new(:dense, 3, [5,0,0, 3,3,0, -1,1,3], dtype)
        expect(a).to eq(b)
      end

      it "exposes lapacke_potri" do
        a = NMatrix.new(3, [4, 0,-1,
                            0, 2, 1,
                            0, 0, 1], dtype: dtype)
        NMatrix::LAPACK::lapacke_potrf(:row, :upper, 3, a, 3)
        NMatrix::LAPACK::lapacke_potri(:row, :upper, 3, a, 3)
        b = NMatrix.new(3, [0.5, -0.5, 1,  0, 1.5, -2,  0, 0, 4], dtype: dtype)
        err = case dtype
                when :float32, :complex64
                  1e-6
                when :float64, :complex128
                  1e-14
              end
        expect(a).to be_within(err).of(b)
      end

      it "exposes lapacke_potrs with vector solution" do
        a = NMatrix.new(3, [4, 0,-1,
                            0, 2, 1,
                            0, 0, 1], dtype: dtype)
        b = NMatrix.new([3,1], [3,0,2], dtype: dtype)

        NMatrix::LAPACK::lapacke_potrf(:row, :upper, 3, a, 3)
        #ldb is different from CLAPACK versions
        NMatrix::LAPACK::lapacke_potrs(:row, :upper, 3, 1, a, 3, b, 1)

        x = NMatrix.new([3,1], [3.5, -5.5, 11], dtype: dtype)

        err = case dtype
                when :float32, :complex64
                  1e-5
                when :float64, :complex128
                  1e-14
              end

        expect(b).to be_within(err).of(x)
      end

      it "exposes lapacke_potrs with matrix solution" do
        a = NMatrix.new(3, [4, 0,-1,
                            0, 2, 1,
                            0, 0, 1], dtype: dtype)
        b = NMatrix.new([3,2], [3,4,
                                0,4,
                                2,0], dtype: dtype)

        NMatrix::LAPACK::lapacke_potrf(:row, :upper, 3, a, 3)
        #ldb is different from CLAPACK versions
        NMatrix::LAPACK::lapacke_potrs(:row, :upper, 3, 2, a, 3, b, 2)

        x = NMatrix.new([3,2], [3.5, 0,
                                -5.5, 4,
                                11, -4], dtype: dtype)

        err = case dtype
                when :float32, :complex64
                  1e-5
                when :float64, :complex128
                  1e-14
              end

        expect(b).to be_within(err).of(x)
      end

      it "calculates the singular value decomposition with lapacke_gesvd" do
        #example from Wikipedia
        m = 4
        n = 5
        mn_min = [m,n].min
        a = NMatrix.new([m,n],[1,0,0,0,2, 0,0,3,0,0, 0,0,0,0,0, 0,4,0,0,0], dtype: dtype)
        s = NMatrix.new([mn_min], 0, dtype: a.abs_dtype) #s is always real and always returned as float/double, never as complex
        u = NMatrix.new([m,m], 0, dtype: dtype)
        vt = NMatrix.new([n,n], 0, dtype: dtype)
        superb = NMatrix.new([mn_min-1], dtype: a.abs_dtype)

        NMatrix::LAPACK.lapacke_gesvd(:row, :a, :a, m, n, a, n, s, u, m, vt, n, superb)

        s_true = NMatrix.new([mn_min], [4,3,Math.sqrt(5),0], dtype: a.abs_dtype)
        u_true = NMatrix.new([m,m], [0,0,1,0, 0,1,0,0, 0,0,0,-1, 1,0,0,0], dtype: dtype)
        vt_true = NMatrix.new([n,n], [0,1,0,0,0, 0,0,1,0,0, Math.sqrt(0.2),0,0,0,Math.sqrt(0.8), 0,0,0,1,0, -Math.sqrt(0.8),0,0,0,Math.sqrt(0.2)], dtype: dtype)

        err = case dtype
                when :float32, :complex64
                  1e-5
                when :float64, :complex128
                  1e-14
              end

        expect(s).to be_within(err).of(s_true)
        expect(u).to be_within(err).of(u_true)
        expect(vt).to be_within(err).of(vt_true)
      end

      #add specs for posv and gesv once we have lapacke versions
    end
  end
end
