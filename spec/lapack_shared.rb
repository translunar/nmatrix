#Define a set of shared examples so that we can test multiple implemntations
#of these LAPACK functions.

RSpec.shared_examples "LAPACK shared" do
  [:rational32, :rational64, :rational128, :float32, :float64, :complex64, :complex128].each do |dtype|
    context dtype do
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
