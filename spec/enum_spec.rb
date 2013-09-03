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

describe "NMatrix enumeration" do
  [:dense, :yale, :list].each do |stype|
    context stype do

      it "should visit each cell in the matrix as if dense, making indices available" do
        n = create_rectangular_matrix(stype)

        vv = []
        ii = []
        jj = []
        n.each_with_indices do |v,i,j|
          vv << v
          ii << i
          jj << j
        end

        vv.should == [1,2,3,4,0,5,6,7,0,8,9,10,0,11,12]
        ii.should == [0,0,0,0,0,1,1,1,1,1,2,2,2,2,2]
        jj.should == [0,1,2,3,4,0,1,2,3,4,0,1,2,3,4]
      end


    end
  end
end