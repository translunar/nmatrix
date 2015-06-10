#Define a set of shared examples so that we can test multiple implemntations
#of these LAPACK functions.

RSpec.shared_examples "LAPACK shared" do
  [:rational32, :rational64, :rational128, :float32, :float64, :complex64, :complex128].each do |dtype|
    context dtype do
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

        expect(a[0,0]).to eq(9) # 8
        expect(a[0,1]).to be_within(err).of(2.quo(9)) # 1
        expect(a[0,2]).to be_within(err).of(4.quo(9)) # 6
        expect(a[1,0]).to eq(5) # 1.quo(2)
        expect(a[1,1]).to be_within(err).of(53.quo(9)) # 17.quo(2)
        expect(a[1,2]).to be_within(err).of(7.quo(53)) # -1
        expect(a[2,0]).to eq(1) # 3.quo(8)
        expect(a[2,1]).to be_within(err).of(52.quo(9))
        expect(a[2,2]).to be_within(err).of(360.quo(53))
      end

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
