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
# == slice_set_spec.rb
#
# Test of slice set operations.

require File.dirname(__FILE__) + "/spec_helper.rb"

describe "Set slice operation" do

  [:dense, :yale, :list].each do |stype|
    context "for #{stype}" do
      before :each do
        @m = create_matrix(stype)
      end

      if stype == :list || stype == :yale
        it "should correctly set and unset a range of entries" do
          pending("yale needed") if stype == :yale
          slice_result_a = NMatrix.new(:dense, 2, 100, @m.dtype).cast(stype)
          slice_result_b = NMatrix.new(:dense, 2, 0,   @m.dtype).cast(stype)
          STDERR.puts "A"
          m = @m.clone
          m[0..1,0..1] = 100
          m[0..1,0..1].should == slice_result_a
          m[2,0..1].should == @m[2,0..1]
          m[0..1,2].should == @m[0..1,2]
          m[0..1,0..1] = 0
          m[0..1,0..1].should == slice_result_b

          STDERR.puts "B"
          m = @m.clone
          m[1..2,0..1] = 100
          m[1..2,0..1].should == slice_result_a
          m[0,0..1].should == @m[0,0..1]
          m[1..2,2].should == @m[1..2,2]
          m[1..2,0..1] = 0
          m[1..2,0..1].should == slice_result_b

          STDERR.puts "C"
          m = @m.clone
          m[1..2,1..2] = 100
          m[1..2,1..2].should == slice_result_a
          m[0,1..2].should == @m[0,1..2]
          m[1..2,0].should == @m[1..2,0]
          m[1..2,1..2] = 0
          m[1..2,1..2].should == slice_result_b

          STDERR.puts "D"
          m = @m.clone
          m[0..1,1..2] = 100
          m[0..1,1..2].should == slice_result_a
          m[2,1..2].should == @m[2,1..2]
          m[0..1,0].should == @m[0..1,0]
          m[0..1,1..2] = 0
          m[0..1,1..2].should == slice_result_b
        end
      else
        it "should correctly set a range of entries to a single value" do
          slice_result = NMatrix.new(:dense, 2, 100, @m.dtype).cast(stype)
          STDERR.puts "A"
          m = @m.clone
          m[0..1,0..1] = 100
          m[0..1,0..1].should == slice_result
          m[2,0..1].should == @m[2,0..1]
          m[0..1,2].should == @m[0..1,2]

          STDERR.puts "B"
          m = @m.clone
          m[1..2,0..1] = 100
          m[1..2,0..1].should == slice_result
          m[0,0..1].should == @m[0,0..1]
          m[1..2,2].should == @m[1..2,2]

          STDERR.puts "C"
          m = @m.clone
          m[1..2,1..2] = 100
          m[1..2,1..2].should == slice_result
          m[0,1..2].should == @m[0,1..2]
          m[1..2,0].should == @m[1..2,0]

          STDERR.puts "D"
          m = @m.clone
          m[0..1,1..2] = 100
          m[0..1,1..2].should == slice_result
          m[2,1..2].should == @m[2,1..2]
          m[0..1,0].should == @m[0..1,0]
        end
      end

      it "should correctly set a single entry" do
        pending if stype == :yale
        n = @m.clone
        old_val = @m[0,0]
        @m[0,0] = 100
        @m[0,0].should == 100
        @m[0,0] = old_val
        @m.should == n
      end

    end
  end

end
