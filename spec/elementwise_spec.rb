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
# SciRuby is Copyright (c) 2010 - 2012, Ruby Science Foundation
# NMatrix is Copyright (c) 2012, Ruby Science Foundation
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
# == nmatrix_spec.rb
#
# Element-wise operation tests.
#

# Can we use require_relative here instead?
require File.join(File.dirname(__FILE__), "spec_helper.rb")

describe NMatrix do

  context "yale" do
    before :each do
      @n = NMatrix.new(:yale, 3, :int64)
      @m = NMatrix.new(:yale, 3, :int64)
      @n[0,0] = 52
      @n[0,2] = 5
      @n[1,1] = 40
      @n[0,1] = 30
      @n[2,0] = 6
      @m[1,1] = -48
      @m[0,2] = -5
      @n.extend NMatrix::YaleFunctions
    end

    it "should perform scalar math" do
      x = @n * 3
      x[0,0].should == 52 * 3
      x[0,1].should == 30 * 3
      x[0,2].should == 5 * 3
      x[1,1].should == 40 * 3
      x[2,0].should == 6 * 3

      r = NMatrix.new(:yale, 3, :int64)
      y = r + 3
      y[0,0].should == 3
    end

    it "should refuse to perform a dot operation on a yale with non-zero default" do
      r = NMatrix.new(:yale, 3, :int64)
      y = r + 3
      expect { y.dot(r) }.to raise_error
      expect { r.dot(y) }.to raise_error
    end

    it "should perform element-wise addition" do
      (@n+@m).should == NMatrix.new(:dense, 3, [52,30,0,0,-8,0,6,0,0], :int64).cast(:yale, :int64)
    end

    it "should perform element-wise subtraction" do
      (@n-@m).should == NMatrix.new(:dense, 3, [52,30,10,0,88,0,6,0,0], :int64).cast(:yale, :int64)
    end

    it "should perform element-wise multiplication" do
      r = NMatrix.new(:dense, 3, [0,0,-25,0,-1920,0,0,0,0], :int64).cast(:yale, :int64)
      m = NMatrix.new(:yale, 2, :int64)
      (@n*@m).should == r
    end

    it "should perform element-wise division" do
      r = NMatrix.new(:dense, 3, [52, 30, -2, 0, -1, 0, 6, 0, 0], :int64).cast(:yale, :int64)
      (@n/(@m+1)).should == r
    end

    it "should perform element-wise modulo" do
      m = NMatrix.new(:yale, 3, :int64) + 5
      (@n % m).should == NMatrix.new(:dense, 3, [2,0,0,0,0,0,1,0,0], :int64).cast(:yale, :int64)
    end

    it "should handle element-wise equality (=~)" do
      (@n =~ @m).should == NMatrix.new(:dense, 3, [false,false,false,true,false,true,false,true,true], :object).cast(:yale, :object, false)
    end

    it "should handle element-wise inequality (!~)" do
      (@n !~ @m).should == NMatrix.new(:dense, 3, [true,true,true,false,true,false,true,false,false], :object).cast(:yale, :object, true)
    end

    it "should handle element-wise less-than (<)" do
      (@m < @n).should == NMatrix.new(:dense, 3, [true,true,true,false,true,false,true,false,false], :object).cast(:yale, :object, true)
    end

    it "should handle element-wise greater-than (>)" do
      (@n > @m).should == NMatrix.new(:dense, 3, [true,true,true,false,true,false,true,false,false], :object).cast(:yale, :object, false)
    end

    it "should handle element-wise greater-than-or-equals (>=)" do
      (@n >= @m).should == NMatrix.new(:dense, 3, true, :object).cast(:yale,:object, true)
    end

    it "should handle element-wise less-than-or-equals (<=)" do
      r = NMatrix.new(:dense, 3, [false,false,false,true,false,true,false,true,true], :object).cast(:yale, :object, false)
      (@n <= @m).should == r
    end
  end


  context "list" do
    before :each do
      @n = NMatrix.new(:list, 2, 0, :int64)
      @m = NMatrix.new(:list, 2, 0, :int64)
      @n[0,0] = 52
      @m[1,1] = -48
      @n[1,1] = 40
    end

    it "should perform scalar math" do
      x = @n * 3
      x[0,0].should == 52 * 3
      x[1,1].should == 40 * 3
      x[0,1].should == 0

      r = NMatrix.new(:list, 3, 1)
      y = r + 3
      y[0,0].should == 4
    end

    it "should perform element-wise addition" do
      r = NMatrix.new(:list, 2, 0, :int64)
      r[0,0] = 52
      r[1,1] = -8
      q = @n + @m
      q.should == r
    end

    it "should perform element-wise subtraction" do
      r = NMatrix.new(:dense, 2, [52, 0, 0, 88], :int64).cast(:list, :int64)
      (@n-@m).should == r
    end

    it "should perform element-wise multiplication" do
      r = NMatrix.new(:dense, 2, [52, 0, 0, -1920], :int64).cast(:list, :int64)
      m = NMatrix.new(:list, 2, 1, :int64)
      m[1,1] = -48
      (@n*m).should == r
    end

    it "should perform element-wise division" do
      m = NMatrix.new(:list, 2, 1, :int64)
      m[1,1] = 2
      r = NMatrix.new(:dense, 2, [52, 0, 0, 20], :int64).cast(:list, :int64)
      (@n/m).should == r
    end

    it "should perform element-wise modulo" do
      m = NMatrix.new(:list, 2, 1, :int64)
      m[0,0] = 50
      m[1,1] = 40
      (@n % m)
    end

    it "should handle element-wise equality (=~)" do
      r = NMatrix.new(:list, 2, false, :object)
      r[0,1] = true
      r[1,0] = true

      (@n =~ @m).should == r
    end

    it "should handle element-wise inequality (!~)" do
      r = NMatrix.new(:list, 2, false, :object)
      r[0,0] = true
      r[1,1] = true

      (@n !~ @m).should == r
    end

    it "should handle element-wise less-than (<)" do
      (@n < @m).should == NMatrix.new(:list, 2, false, :object)
    end

    it "should handle element-wise greater-than (>)" do
      r = NMatrix.new(:list, 2, false, :object)
      r[0,0] = true
      r[1,1] = true
      (@n > @m).should == r
    end

    it "should handle element-wise greater-than-or-equals (>=)" do
      (@n >= @m).should == NMatrix.new(:list, 2, true, :object)
    end

    it "should handle element-wise less-than-or-equals (<=)" do
      r = NMatrix.new(:list, 2, false, :object)
      r[0,1] = true
      r[1,0] = true
      (@n <= @m).should == r
    end
  end

  context "dense" do
    context "scalar arithmetic" do
      before :each do
        @n = NMatrix.new(:dense, 2, [1,2,3,4], :int64)
      end

      it "works for integers" do
        (@n+1).should == NMatrix.new(:dense, 2, [2,3,4,5], :int64)
      end

      #it "works for complex64" do
      #  n = @n.cast(:dtype => :complex64)
      #  (n + 10.0).to_a.should == [Complex(11.0), Complex(12.0), Complex(13.0), Complex(14.0)]
      #end
    end

    context "elementwise arithmetic" do
      before :each do
        @n = NMatrix.new(:dense, 2, [1,2,3,4], :int64)
        @m = NMatrix.new(:dense, 2, [-4,-1,0,66], :int64)
      end

      it "adds" do
        r = @n+@m
        r.should == NMatrix.new(:dense, [2,2], [-3, 1, 3, 70], :int64)
      end

      it "subtracts" do
        r = @n-@m
        r.should == NMatrix.new(:dense, [2,2], [5, 3, 3, -62], :int64)
      end

      it "multiplies" do
        r = @n*@m
        r.should == NMatrix.new(:dense, [2,2], [-4, -2, 0, 264], :int64)
      end

      it "divides in the Ruby way" do
        m = @m.clone
        m[1,0] = 3
        r = @n/m
        r.should == NMatrix.new(:dense, [2,2], [-1, -2, 1, 0], :int64)
      end

      it "exponentiates" do
        r = @n ** 2
        # TODO: We might have problems with the dtype.
        r.should == NMatrix.new(:dense, [2,2], [1, 4, 9, 16], :int64)
      end

      it "modulo" do
        (@n % (@m + 2)).should == NMatrix.new(:dense, [2,2], [-1, 0, 1, 4], :int64)
      end
    end

    context "elementwise comparisons" do
      before :each do
        @n = NMatrix.new(:dense, 2, [1,2,3,4], :int64)
        @m = NMatrix.new(:dense, 2, [-4,-1,3,2], :int64)
      end

      it "equals" do
        r = @n =~ @m
        r.should == NMatrix.new(:dense, [2,2], [false, false, true, false], :object)
      end

      it "is not equal" do
        r = @n !~ @m
        r.should == NMatrix.new(:dense, [2,2], [true, true, false, true], :object)
      end

      it "is less than" do
        r = @n < @m
        r.should == NMatrix.new(:dense, [2,2], false, :object)
      end

      it "is greater than" do
        r = @n > @m
        r.should == NMatrix.new(:dense, [2,2], [true, true, false, true], :object)
      end

      it "is less than or equal to" do
        r = @n <= @m
        r.should == NMatrix.new(:dense, [2,2], [false, false, true, false], :object)
      end

      it "is greater than or equal to" do
        n = NMatrix.new(:dense, [2,2], [1, 2, 2, 4], :int64)
        r = n >= @m
        r.should == NMatrix.new(:dense, [2,2], [true, true, false, true], :object)
      end
    end
  end
end
