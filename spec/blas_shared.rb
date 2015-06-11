#Shared spec so that we can test our internal implementation of BLAS functions
#as well as the one provided by ATLAS

RSpec.shared_examples "BLAS shared" do
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
