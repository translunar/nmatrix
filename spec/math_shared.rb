#Shared specs for methods that make use of LAPACK/BLAS when it is available

RSpec.shared_examples "math shared" do
  # TODO: Get it working with ROBJ too
  [:byte,:int8,:int16,:int32,:int64,:float32,:float64,:rational64,:rational128].each do |left_dtype|
    [:byte,:int8,:int16,:int32,:int64,:float32,:float64,:rational64,:rational128].each do |right_dtype|

      # Won't work if they're both 1-byte, due to overflow.
      next if [:byte,:int8].include?(left_dtype) && [:byte,:int8].include?(right_dtype)

      # For now, don't bother testing int-int mult.
      #next if [:int8,:int16,:int32,:int64].include?(left_dtype) && [:int8,:int16,:int32,:int64].include?(right_dtype)
      it "dense handles #{left_dtype.to_s} dot #{right_dtype.to_s} matrix multiplication" do
        #STDERR.puts "dtype=#{dtype.to_s}"
        #STDERR.puts "2"

        nary = if left_dtype.to_s =~ /complex/
                 COMPLEX_MATRIX43A_ARRAY
               elsif left_dtype.to_s =~ /rational/
                 RATIONAL_MATRIX43A_ARRAY
               else
                 MATRIX43A_ARRAY
               end

        mary = if right_dtype.to_s =~ /complex/
                 COMPLEX_MATRIX32A_ARRAY
               elsif right_dtype.to_s =~ /rational/
                 RATIONAL_MATRIX32A_ARRAY
               else
                 MATRIX32A_ARRAY
               end

        n = NMatrix.new([4,3], nary, dtype: left_dtype, stype: :dense)
        m = NMatrix.new([3,2], mary, dtype: right_dtype, stype: :dense)

        expect(m.shape[0]).to eq(3)
        expect(m.shape[1]).to eq(2)
        expect(m.dim).to eq(2)

        expect(n.shape[0]).to eq(4)
        expect(n.shape[1]).to eq(3)
        expect(n.dim).to eq(2)

        expect(n.shape[1]).to eq(m.shape[0])

        r = n.dot m

        expect(r[0,0]).to eq(273.0)
        expect(r[0,1]).to eq(455.0)
        expect(r[1,0]).to eq(243.0)
        expect(r[1,1]).to eq(235.0)
        expect(r[2,0]).to eq(244.0)
        expect(r[2,1]).to eq(205.0)
        expect(r[3,0]).to eq(102.0)
        expect(r[3,1]).to eq(160.0)

        #r.dtype.should == :float64 unless left_dtype == :float32 && right_dtype == :float32
      end
    end
  end

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

  #determinant calculation (sometimes) uses getrf, so it goes here
  context "determinants" do
    ALL_DTYPES.each do |dtype|
      next if dtype == :object
      context dtype do
        before do
          @a = NMatrix.new([2,2], [1,2,
                                   3,4], dtype: dtype)
          @b = NMatrix.new([3,3], [1,2,3,
                                   5,0,1,
                                   4,1,3], dtype: dtype)
          @c = NMatrix.new([4,4], [1, 0, 1, 1,
                                   1, 2, 3, 1,
                                   3, 3, 3, 1,
                                   1, 2, 3, 4], dtype: dtype)
          @err = case dtype
                  when :float32, :complex64
                    1e-6
                  when :float64, :complex128
                    1e-14 #this was originally 1e-15, this seemed to work when using ATLAS, but not with internal implementation? Look into this?
                  else
                    1e-64 # FIXME: should be 0, but be_within(0) does not work.
                end
        end
        it "computes the determinant of 2x2 matrix" do
          expect(@a.det).to be_within(@err).of(-2)
        end
        it "computes the determinant of 3x3 matrix" do
          expect(@b.det).to be_within(@err).of(-8)
        end
        it "computes the determinant of 4x4 matrix" do
          expect(@c.det).to be_within(@err).of(-18)
        end
      end
    end
  end

end
