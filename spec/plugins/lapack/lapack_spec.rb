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
    end
  end
end
