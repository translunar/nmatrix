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
# SciRuby is Copyright (c) 2010 - 2013, Ruby Science Foundation
# NMatrix is Copyright (c) 2013, Ruby Science Foundation
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
# == yale_functions.rb
#
# This file contains some shortcut functions for the specialty
# Yale matrix extensions (mostly for debugging and experimental
# purposes, but sometimes applicable when you need to speed up
# your code a lot).
#++

module NMatrix::YaleFunctions
  # call-seq:
  #     yale_nd_row_as_array -> Array
  #
  # Returns the non-diagonal column indices which are stored in a given row.
  def yale_nd_row_as_array i
    yale_nd_row(i, :array)
  end

  # call-seq:
  #     yale_nd_row_as_set -> Set
  #
  # Returns the non-diagonal column indices which are stored in a given row, as a Set.
  def yale_nd_row_as_set i
    require 'set'
    yale_nd_row(i, :array).to_set
  end

  # call-seq:
  #     yale_nd_row_as_sorted_set -> SortedSet
  #
  # Returns the non-diagonal column indices which are stored in a given row, as a Set.
  def yale_nd_row_as_sorted_set i
    require 'set'
    SortedSet.new(yale_nd_row(i, :array))
  end

  # call-seq:
  #     yale_nd_row_as_hash -> Hash
  #
  # Returns the non-diagonal column indices and entries stored in a given row.
  def yale_nd_row_as_hash i
    yale_nd_row(i, :hash)
  end

  # call-seq:
  #     yale_row_as_array -> Array
  #
  # Returns the diagonal and non-digonal column indices stored in a given row.
  def yale_row_as_array i
    ary = yale_nd_row(i, :array)
    return ary if i >= self.shape[1] || self[i,i].nil? || self[i,i] == 0
    ary << i
  end

  # call-seq:
  #     yale_row_as_set -> Set
  #
  # Returns the diagonal and non-diagonal column indices stored in a given row.
  def yale_row_as_set i
    require 'set'
    yale_row_as_array(i).to_set
  end

  # call-seq:
  #     yale_row_as_sorted_set -> SortedSet
  #
  # Returns the diagonal and non-diagonal column indices stored in a given row.
  def yale_row_as_sorted_set i
    require 'set'
    SortedSet.new(yale_row_as_array(i))
  end

  # call-seq:
  #     yale_row_as_hash -> Hash
  #
  # Returns the diagonal and non-diagonal column indices and entries stored in a given row.
  def yale_row_as_hash i
    h = yale_nd_row(i, :hash)
    return h if i >= self.shape[1] || self[i,i].nil? || self[i,i] == 0
    h[i] = self[i,i]
  end
end