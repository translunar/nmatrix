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
# == stat_spec.rb
#
# Tests for statistical functions in NMatrix.
#

# Can we use require_relative here instead?
require File.join(File.dirname(__FILE__), "spec_helper.rb")
require 'pry'

describe "Statistical functions" do
  context "mapping and reduction related functions" do

    before :each do
      @nm_1d = NMatrix[5.0,0.0,1.0,2.0,3.0]
      @nm_2d = NMatrix[[0.0,1.0],[2.0,3.0]]
    end

    it "behaves like Enumerable#reduce with no argument to reduce" do
      @nm_1d.reduce_along_dim(0) { |acc, el| acc + el }.to_f.should eq 11
      @nm_2d.reduce_along_dim(1) { |acc, el| acc + el }.should eq NMatrix[[1, 5]]
    end

    it "should calculate the mean along the specified dimension" do
      @nm_1d.mean.should eq NMatrix[2.2]
      @nm_2d.mean.should eq NMatrix[[1.0,2.0]]
    end

    it "should calculate the minimum along the specified dimension" do
      @nm_1d.min.should eq 0.0
      @nm_2d.min.should eq NMatrix[[0.0, 1.0]]
      @nm_2d.min(1).should eq NMatrix[[0.0], [2.0]]
    end

    it "should calculate the maximum along the specified dimension" do
      @nm_1d.max.should eq 5.0
      @nm_2d.max.should eq NMatrix[[2.0, 3.0]]
    end

    it "should calculate the variance along the specified dimension" do
      @nm_1d.variance.should eq NMatrix[3.7]
      @nm_2d.variance(1).should eq NMatrix[[0.5], [0.5]]
    end

    it "should calculate the sum along the specified dimension" do
      @nm_1d.sum.should eq NMatrix[11]
      @nm_2d.sum.should eq NMatrix[[2], [4]]
    end

    it "should calculate the standard deviation along the specified dimension" do
      @nm_1d.std.should eq NMatrix[Math.sqrt(3.7)]
      @nm_2d.std(1).should eq NMatrix[[Math.sqrt(0.5)], [Math.sqrt(0.5)]]
    end

    it "should raise an ArgumentError when any invalid dimension is provided" do
      expect { @nm_1d.mean(3) }.to raise_exception(RangeError)
    end

    it "should convert to float if it contains only a single element" do
      NMatrix[4.0].to_f.should eq 4.0
      NMatrix[[[[4.0]]]].to_f.should eq 4.0
    end

    it "should raise an index error if it contains more than a single element" do
      expect { @nm_1d.to_f }.to raise_error(IndexError)
    end

    it "should map a block to all elements" do
      @nm_1d.map { |e| e ** 2 }.should eq NMatrix[25.0,0.0,1.0,4.0,9.0]
      @nm_2d.map { |e| e ** 2 }.should eq NMatrix[[0.0,1.0],[4.0,9.0]]
    end

    it "should map! a block to all elements in place" do
      fct = Proc.new { |e| e ** 2 }
      expected1 = @nm_1d.map &fct
      expected2 = @nm_2d.map &fct
      @nm_1d.map! &fct
      @nm_1d.should eq expected1
      @nm_2d.map! &fct
      @nm_2d.should eq expected2
    end

    it "should return an enumerator for map without a block" do
      @nm_1d.map.should be_a Enumerator
    end

    it "should return an enumerator for reduce without a block" do
      @nm_1d.reduce_along_dim(0).should be_a Enumerator
    end

    it "should return an enumerator for each_along_dim without a block" do
      @nm_1d.each_along_dim(0).should be_a Enumerator
    end

    it "should iterate correctly for map without a block" do
      en = @nm_1d.map
      en.each { |e| e**2 }.should eq @nm_1d.map { |e| e**2 }
      en = @nm_2d.map
      en.each { |e| e**2 }.should eq @nm_2d.map { |e| e**2 }
    end

    it "should iterate correctly for reduce without a block" do
      en = @nm_1d.reduce_along_dim(0, 1.0)
      en.each { |a, e| a+e }.to_f.should eq 12
      en = @nm_2d.reduce_along_dim(1, 1.0)
      en.each { |a, e| a+e }.should eq NMatrix[[2.0],[6.0]]
    end

    it "should iterate correctly for each_along_dim without a block" do
      res = NMatrix.zeros_like(@nm_1d[0...1])
      en = @nm_1d.each_along_dim(0)
      en.each { |e| res += e }
      res.to_f.should eq 11

      res = NMatrix.zeros_like (@nm_2d[0...2, 0])
      en = @nm_2d.each_along_dim(1)
      en.each { |e| res += e }
      res.should eq NMatrix[[1.0], [5.0]]
    end

    it "should yield matrices of matching dtype for each_along_dim" do
      m = NMatrix.new([2,3], [1,2,3,3,4,5], dtype: :complex128)
      m.each_along_dim(1) do |sub_m|
        sub_m.dtype.should eq :complex128
      end
    end

    it "should reduce to a matrix of matching dtype for reduce_along_dim" do
      m = NMatrix.new([2,3], [1,2,3,3,4,5], dtype: :complex128)
      m.reduce_along_dim(1) do |acc, sub_m|
        sub_m.dtype.should eq :complex128
        acc
      end

      m = NMatrix.new([2,3], [1,2,3,3,4,5], dtype: :complex128)
      m.reduce_along_dim(1, 0.0) do |acc, sub_m|
        sub_m.dtype.should eq :complex128
        acc
      end
    end

    it "should allow overriding the dtype for reduce_along_dim" do
      m = NMatrix[[1,2,3], [3,4,5], dtype: :complex128]
      m.reduce_along_dim(1, 0.0, :float64) do |acc, sub_m|
        acc.dtype.should eq :float64
        acc
      end

      m = NMatrix[[1,2,3], [3,4,5], dtype: :complex128]
      m.reduce_along_dim(1, nil, :float64) do |acc, sub_m|
        acc.dtype.should eq :float64
        acc
      end
    end

    it "should convert integer dtypes to float when calculating mean" do
      m = NMatrix[[1,2,3], [3,4,5], dtype: :int32]
      m.mean(0).dtype.should eq :float64
    end

    it "should convert integer dtypes to float when calculating variance" do
      m = NMatrix[[1,2,3], [3,4,5], dtype: :int32]
      m.variance(0).dtype.should eq :float64
    end

    it "should convert integer dtypes to float when calculating standard deviation" do
      m = NMatrix[[1,2,3], [3,4,5], dtype: :int32]
      m.std(0).dtype.should eq :float64
    end
  end
end