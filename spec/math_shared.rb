#Shared specs for methods that make use of LAPACK/BLAS when it is available

RSpec.shared_examples "math shared" do
  NON_INTEGER_DTYPES.each do |dtype|
    next if dtype == :object
    context dtype do
      before do
        @m = NMatrix.new(:dense, 3, [4,9,2,3,5,7,8,1,6], dtype)
      end

      #haven't check this spec yet. Also it doesn't check all the elements of the matrix.
      it "should correctly factorize a matrix" do
        a = @m.factorize_lu
        expect(a[0,0]).to eq(8)
        expect(a[0,1]).to eq(1)
        expect(a[0,2]).to eq(6)
        expect(a[1,0]).to eq(0.5)
        expect(a[1,1]).to eq(8.5)
        expect(a[1,2]).to eq(-1)
        expect(a[2,0]).to eq(0.375)
      end

      it "also returns the permutation matrix" do
        a, p = @m.factorize_lu perm_matrix: true

        expect(a[0,0]).to eq(8)
        expect(a[0,1]).to eq(1)
        expect(a[0,2]).to eq(6)
        expect(a[1,0]).to eq(0.5)
        expect(a[1,1]).to eq(8.5)
        expect(a[1,2]).to eq(-1)
        expect(a[2,0]).to eq(0.375)

        puts p
        expect(p[1,0]).to eq(1)
        expect(p[2,1]).to eq(1)
        expect(p[0,2]).to eq(1)
      end
    end

    #not totally sure the deal with these specs
    context dtype do
      it "should correctly invert a matrix in place (bang)" do
        a = NMatrix.new(:dense, 3, [1,2,3,0,1,4,5,6,0], dtype)
        b = NMatrix.new(:dense, 3, [-24,18,5,20,-15,-4,-5,4,1], dtype)
        begin
          a.invert!
        rescue NotImplementedError => e
          if dtype.to_s =~ /rational/
            pending "getri needs rational implementation"
          else
            pending e.to_s
          end
        end
        expect(a.round).to eq(b)
      end

      unless NMatrix.has_clapack? #why?
        it "should correctly invert a matrix in place" do #this doesn't look in place
          a = NMatrix.new(:dense, 5, [1, 8,-9, 7, 5, 
                                      0, 1, 0, 4, 4, 
                                      0, 0, 1, 2, 5, 
                                      0, 0, 0, 1,-5,
                                      0, 0, 0, 0, 1 ], dtype)
          b = NMatrix.new(:dense, 5, [1,-8, 9, 7, 17,
                                      0, 1, 0,-4,-24,
                                      0, 0, 1,-2,-15,
                                      0, 0, 0, 1,  5,
                                      0, 0, 0, 0,  1,], dtype)
          expect(a.invert).to eq(b)
        end
      end

      it "should correctly invert a matrix out-of-place" do
        a = NMatrix.new(:dense, 3, [1,2,3,0,1,4,5,6,0], dtype)
        b = NMatrix.new(:dense, 3, [-24,18,5,20,-15,-4,-5,4,1], dtype)

        expect(a.invert(3,3)).to eq(b) #these arguments don't do anything??
      end
    end
  end
end
