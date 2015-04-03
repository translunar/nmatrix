#--
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
# SciRuby is Copyright (c) 2010 - 2015, Ruby Science Foundation
# NMatrix is Copyright (c) 2012 - 2015, John Woods and the Ruby Science Foundation
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
# == io/packable.rb
#
# Makes it easy to pack NMatrix into a binary format that can be sent to rinruby or
# elsewhere.
#
#++

require 'packable'

class NMatrix
  include Packable
  
  def write_packed(packedio, options)
    packedio << self.dim
    self.shape.each do |x|
      packedio << x
    end
    self.each do |x|
      packedio << x
    end
    return packedio
  end

  def self.read_packed(packedio, options)
    d = packedio.read(Integer)
    shape = []
    d.times do |x|
      shape << packedio.read(Integer)
    end
    data = []
    n = shape.inject(:*)
    n.times do |i|
      data << packedio.read(Float)
    end
    h = NMatrix.new(shape, data)
    return h
  end
end


