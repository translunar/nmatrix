
# Can we use require_relative here instead?
require File.join(File.dirname(__FILE__), "spec_helper.rb")

describe NMatrix do
 #[:dense, :list, :yale].each do |storage_type|
 [:list].each do |storage_type|
    context storage_type do
      it "can be duplicated" do
        n = NMatrix.new(storage_type, [2,3], storage_type == :yale ? :float64 : 1.1)
        n.stype.should equal(storage_type)

        n[0,0] = 0.0
        n[0,1] = 0.1
        n[1,0] = 1.0

        m = n.dup
        m.shape.should == n.shape
        m.dim.should == n.dim
        m.object_id.should_not == n.object_id
        m.stype.should equal(storage_type)
        m[0,0].should == n[0,0]
        m[0,0] = 3.0
        m[0,0].should_not == n[0,0]
      end

      it "sets and gets" do
        n = NMatrix.new(storage_type, 2, storage_type == :yale ? :int8 : 0)
        n[0,1] = 1
        n[0,0].should == 0
        n[1,0].should == 0
        n[0,1].should == 1
        n[1,1].should == 0
      end

      it "sets and gets references" do
        n = NMatrix.new(storage_type, 2, storage_type == :yale ? :int8 : 0)
        (n[0,1] = 1).should == 1
        n[0,1].should == 1
      end

      # Tests Ruby object versus any C dtype (in this case we use :int64)
      [:object, :int64].each do |dtype|
        c = dtype == :object ? "Ruby object" : "non-Ruby object"
        context c do
          it "allows iteration of matrices" do
            pending("yale and list not implemented yet") unless storage_type == :dense
            n = NMatrix.new(:dense, [3,3], [1,2,3,4,5,6,7,8,9], dtype)
            n.each do |x|
              puts x
            end
          end

          it "allows storage-based iteration of matrices" do
            #pending("list not implemented yet") if storage_type == :list
            n = storage_type == :yale ? NMatrix.new(storage_type, [3,3], dtype) : NMatrix.new(storage_type, [3,3], 0, dtype)
            if not n
              n = NMatrix.new(storage_type, [3,3], dtype)
            end
            n[0,0] = 1
            n[0,1] = 2
            n[2,2] = 3
            n[2,1] = 4

            values = []
            is = []
            js = []
            n.each_stored_with_indices do |v,i,j|
              if storage_type == :list
                puts "V: #{v}, I: #{i}, J: #{j}"
              end
              values << v
              is << i
              js << j
            end

            if storage_type == :yale
              values.should == [1,0,3,2,4]
              is.should     == [0,1,2,0,2]
              js.should     == [0,1,2,1,1]
            elsif storage_type == :list
              puts "TESTING STORAGE_TYPE :list on #each_stored_with_indices"
              p values
              p is
              p js
              values.should == [1,2,4,3]
              is.should     == [0,0,2,2]
              js.should     == [0,1,2,1]
            elsif storage_type == :dense
              values.should == [1,2,0,0,0,0,0,4,3]
              is.should     == [0,0,0,1,1,1,2,2,2]
              js.should     == [0,1,2,0,1,2,0,1,2]
            else 
              values.should_not be_empty()
            end
          end
        end
      end

    end

    # dense and list, not yale
    context "(storage: #{storage_type})" do
      it "gets default value" do
        NMatrix.new(storage_type, 3, 0)[1,1].should   == 0
        NMatrix.new(storage_type, 3, 0.1)[1,1].should == 0.1
        NMatrix.new(storage_type, 3, 1)[1,1].should   == 1
      end

      it "returns shape and dim" do
        NMatrix.new(storage_type, [3,2,8], 0).shape.should == [3,2,8]
        NMatrix.new(storage_type, [3,2,8], 0).dim.should  == 3
      end
      
      it "returns number of rows and columns" do
        NMatrix.new(storage_type, [7, 4], 3).rows.should == 7
        NMatrix.new(storage_type, [7, 4], 3).cols.should == 4
      end
    end unless storage_type == :yale
  end


  it "handles dense construction" do
    NMatrix.new(3,0)[1,1].should == 0
    lambda { NMatrix.new(3,:int8)[1,1] }.should_not raise_error
  end

  it "calculates the complex conjugate in-place" do
    n = NMatrix.new(:dense, 3, [1,2,3,4,5,6,7,8,9], :complex128)
    n.complex_conjugate!
    # FIXME: Actually test that values are correct.
  end

  it "converts from list to yale properly" do
    m = NMatrix.new(:list, 3, 0)
    m[0,2] = 333 
    m[2,2] = 777
    n = m.cast(:yale, :int32)
    puts n.capacity
    n.extend NMatrix::YaleFunctions
    puts n.yale_ija.inspect
    puts n.yale_a.inspect
    n[0,0].should == 0
    n[0,1].should == 0
    n[0,2].should == 333
    n[1,0].should == 0
    n[1,1].should == 0
    n[1,2].should == 0
    n[2,0].should == 0
    n[2,1].should == 0
    n[2,2].should == 777
  end

  it "should return an enumerator when each is called without a block" do
    a = NMatrix.new(2, 1)
    b = NMatrix.new(2, [-1,0,1,0])
    enums = [a.each, b.each]

    begin
      atans = []
      atans << Math.atan2(*enums.map(&:next)) while true
    rescue StopIteration
    end
  end
end
