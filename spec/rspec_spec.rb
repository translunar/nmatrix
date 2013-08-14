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
# == rspec_spec.rb
#
# A spec for testing monkey patches to RSpec for NMatrix.
#
require File.join(File.dirname(__FILE__), "spec_helper.rb")

describe "RSpec" do
  it "should permit #be_within to be used on a dense NMatrix" do
    (NMatrix.new(:dense, [4,1], 1.0, :complex128) / 10000.0).should be_within(0.00000001).of(NMatrix.new(:dense, [4,1], 0.0001, :float64))
    (NMatrix.new(:dense, [4,1], 1.0, :complex128) / 10000.0).should_not be_within(0.00000001).of(NMatrix.new(:dense, [4,1], 1.0, :float64))
  end
end
