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
# == enum_spec.rb
#
# Enumerator tests for NMatrix.
#

require File.dirname(__FILE__) + "/spec_helper.rb"

describe "NMatrix enumeration for" do
  [:dense, :yale, :list].each do |stype|
    context stype do

      before :each do
        @n = create_rectangular_matrix(stype)
      end

      if stype == :yale
        it "should visit each stored element of the matrix in order by indices" do
          vv = []
          ii = []
          jj = []
          @n.each_ordered_stored_with_indices do |v,i,j|
            vv << v
            ii << i
            jj << j
          end
          require 'pry'
          binding.pry

          vv.should == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 16, 17, 18, 19, 20]
          ii.should == [[0]*5, [1]*5, [2]*5, [3]*1, [4]*5].flatten
          jj.should == [0,1,2,3,4,  0,1,2,3,5,  0,1,2,4,5,  3,  0,2,3,4,5]
        end
      end

      it "should visit each cell in the matrix as if dense, making indices available" do
        vv = []
        ii = []
        jj = []
        @n.each_with_indices do |v,i,j|
          vv << v
          ii << i
          jj << j
        end

        vv.should == [1,2,3,4,5,0,6,7,8,9,0,10,11,12,13,0,14,15,0,0,0,0,0,0,16,0,17,18,19,20]
        ii.should == [[0]*6, [1]*6, [2]*6, [3]*6, [4]*6].flatten
        jj.should == [0,1,2,3,4,5]*5
      end

      it "should visit each cell in the slice as if dense, making indices available" do
        m = @n[1..4,1..3]
        vv = []
        ii = []
        jj = []
        m.each_with_indices do |v,i,j|
          vv << v
          ii << i
          jj << j
        end
        jj.should == [0,1,2]*4
        ii.should == [[0]*3, [1]*3, [2]*3, [3]*3].flatten
        vv.should == [7,8,9,12,13,0,0,0,0,0,17,18]

      end


    end
  end
end