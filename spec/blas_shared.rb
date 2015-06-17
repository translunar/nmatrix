#Shared spec so that we can test our internal implementation of BLAS functions
#as well as the one provided by ATLAS

RSpec.shared_examples "BLAS shared" do
  [:float32, :float64, :complex64, :complex128, :object].each do |dtype|
    #this spec doesn't check check anything
    context dtype do
      it "exposes gemv" do
        a = NMatrix.new([4,3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], dtype: :float64)
        x = NMatrix.new([3,1], [2.0, 1.0, 0.0], dtype: :float64)

        NMatrix::BLAS.gemv(a, x)
      end
    end
  end

  [:rational32, :rational64, :rational128, :float32, :float64, :complex64, :complex128].each do |dtype|
    context dtype do
      # This is not the same as "exposes cblas trsm", which would be for a version defined in blas.rb (which
      # would greatly simplify the calling of cblas_trsm in terms of arguments, and which would be accessible
      # as NMatrix::BLAS::trsm)
      # I haven't checked this spec -WL
      it "exposes unfriendly cblas_trsm" do
        a     = NMatrix.new(3, [4,-1.quo(2), -3.quo(4), -2, 2, -1.quo(4), -4, -2, -1.quo(2)], dtype: dtype)
        b     = NMatrix.new([3,1], [-1, 17, -9], dtype: dtype)
        NMatrix::BLAS::cblas_trsm(:row, :right, :lower, :transpose, :nonunit, 1, 3, 1.0, a, 3, b, 3)

        # These test results all come from actually running a matrix through BLAS. We use them to ensure that NMatrix's
        # version of these functions (for rationals) give similar results.

        expect(b[0]).to eq(-1.quo(4))
        expect(b[1]).to eq(33.quo(4))
        expect(b[2]).to eq(-13)

        NMatrix::BLAS::cblas_trsm(:row, :right, :upper, :transpose, :unit, 1, 3, 1.0, a, 3, b, 3)

        expect(b[0]).to eq(-15.quo(2))
        expect(b[1]).to eq(5)
        expect(b[2]).to eq(-13)
      end
    end
  end

  [:float32, :float64, :complex64, :complex128, :object].each do |dtype|
    context dtype do
      # Note: this exposes gemm, not cblas_gemm (which is the unfriendly CBLAS no-error-checking version)
      it "exposes gemm" do
        n = NMatrix.new([4,3], [14.0,9.0,3.0, 2.0,11.0,15.0, 0.0,12.0,17.0, 5.0,2.0,3.0], dtype: dtype)
        m = NMatrix.new([3,2], [12.0,25.0, 9.0,10.0, 8.0,5.0], dtype: dtype)

        #c = NMatrix.new([4,2], dtype)
        r = NMatrix::BLAS.gemm(n, m) #, c)
        #c.should equal(r) # check that both are same memory address

        expect(r).to eq(NMatrix.new([4,2], [273,455,243,235,244,205,102,160], dtype: dtype))
      end
    end
  end
end
