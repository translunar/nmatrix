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
# == nvector_spec.rb
#
# Basic tests for NVector.
#

require File.dirname(__FILE__) + "/spec_helper.rb"

describe NVector do
  it "initializes" do
    v = NVector.new(5, 0, :float64)
    v.shape[0].should == 1
    v.shape[1].should == 5
  end

  it "permits setting and getting contents" do
    v = NVector.new(5, 0, :float64)
    v[0] = 1.555
    v[0].should == 1.555
  end
  
  it "transpose() changes raw and column stored structure" do
    v = NVector.new 5, :float64
    v = v.transpose
    v.shape[0].should == 5
    v.shape[1].should == 1
    v[0] = 1.555
    v[0].should == 1.555
  end

  it "dot() multiples itself by another NVector" do
    v1 = NVector.new(2, :float64)
    v2 = NVector.new(2, :float64).transpose
    v1[0] = 1.5
    v1[1] = 2.3
    v2[0] = 1.3
    v2[1] = 2.5
    v1.dot(v2).should be_within(0.000000001).of(7.7)
  end

  it "dot!() multiples itself destructively by another NVector" do
    pending "dot! not yet implemented"
    v1 = NVector.new 2, :float64
    v2 = NVector.new(2, :float64).transpose
    v1[0] = 1.5
    v1[1] = 2.3
    v2[0] = 1.3
    v2[1] = 2.5
    v1.dot!(v2)
    v1.should be_within(0.000000001).of(7.7)
  end

  it "pretty_print() prints values to standard output with a pretty format" do
    pending "pretty_print formatting is finalized"
    v = NVector.new(5, 0)
    $stdout = StringIO.new
    v.pretty_print
    out = $stdout.string
    $stdout = STDOUT
    out.should == "0  0  0  0  0\n"
  end

  it "inspect() formats the output with inspected, namely human readable format" do
    pending "inspect output is finalized"
    v = NVector.new(5, 0)
    $stdout = StringIO.new
    p v
    out = $stdout.string
    $stdout = STDOUT
    out.should == "0  0  0  0  0\n"
  end

  [:dense, :list, :yale].each do |storage_type|
    context "for #{storage_type}" do
      before :each do
        @m = create_vector(storage_type)
      end

      it "converts to an Array" do
        a = @m.to_a
        a.each.with_index { |v,idx| @m[idx].should equal(v) }
      end

      it "shuffles" do
        n = @m.shuffle
        n.to_a.hash.should_not == @m.to_a.hash
        n.to_a.sort.hash.should equal(@m.to_a.sort.hash)
      end
    end
  end
end
