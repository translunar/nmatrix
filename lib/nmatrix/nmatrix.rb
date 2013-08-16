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
# == nmatrix.rb
#
# This file contains those core functionalities which can be
# implemented efficiently (or much more easily) in Ruby (e.g.,
# inspect, pretty_print, element-wise operations).
#++

require_relative './lapack.rb'
require_relative './yale_functions.rb'

class NMatrix

  # Read and write extensions for NMatrix. These are only loaded when needed.
  #
  module IO
    module Matlab
      class << self
        def load_mat file_path
          NMatrix::IO::Matlab::Mat5Reader.new(File.open(file_path, "rb+")).to_ruby
        end
      end

      # FIXME: Remove autoloads
      autoload :MatReader, 'nmatrix/io/mat_reader'
      autoload :Mat5Reader, 'nmatrix/io/mat5_reader'
    end

    # FIXME: Remove autoloads
    autoload :Market, 'nmatrix/io/market'
  end

  class << self
    #
    # call-seq:
    #     load_file(path) -> Mat5Reader
    #
    # * *Arguments* :
    #   - +path+ -> The path to a version 5 .mat file.
    # * *Returns* :
    #   - A Mat5Reader object.
    #
    def load_file(file_path)
      NMatrix::IO::Mat5Reader.new(File.open(file_path, 'rb')).to_ruby
    end
  end

  # TODO: Make this actually pretty.
  def pretty_print(q) #:nodoc:
    if self.dim > 3 || self.dim == 1
      self.to_a.pretty_print(q)
    else
      # iterate through the whole matrix and find the longest number
      longest = Array.new(self.shape[1], 0)
      self.each_column.with_index do |col, j|
        col.each do |elem|
          elem_len   = elem.to_s.size
          longest[j] = elem_len if longest[j] < elem_len
        end
      end

      if self.dim == 3
        q.group(0, "\n{ layers:", "}") do
          self.each_layer.with_index do |layer,k|
            q.group(0, "\n  [\n", "  ]\n") do
              layer.each_row.with_index do |row,i|
                q.group(0, "    [", "]\n") do
                  q.seplist(self[i,0...self.shape[1],k].to_flat_array, lambda { q.text ", "}, :each_with_index) { |v,j| q.text v.to_s.rjust(longest[j]) }
                end
              end
            end
          end
        end
      else # dim 2
        q.group(0, "\n[\n", "]") do
          self.each_row.with_index do |row,i|
            q.group(1, "  [", "]") do
              q.seplist(self.dim > 2 ? row.to_a[0] : row.to_a, lambda { q.text ", " }, :each_with_index) { |v,j| q.text v.to_s.rjust(longest[j]) }
            end
            q.breakable
          end
        end
      end
    end
  end
  #alias :pp :pretty_print



  #
  # call-seq:
  #     cast(stype, dtype, default) -> NMatrix
  #     cast(stype, dtype) -> NMatrix
  #     cast(stype) -> NMatrix
  #     cast(options) -> NMatrix
  #
  # This is a user-friendly helper for calling #cast_full. The easiest way to call this function is using an
  # options hash, e.g.,
  #
  #     n.cast(:stype => :yale, :dtype => :int64, :default => false)
  #
  # For list and yale, :default sets the "default value" or "init" of the matrix. List allows a bit more freedom
  # since non-zeros are permitted. For yale, unpredictable behavior may result if the value is not false, nil, or
  # some version of 0. Dense discards :default.
  #
  # dtype and stype are inferred from the matrix upon which #cast is called -- so you only really need to provide
  # one. You can actually call this function with no arguments, in which case it functions like #clone.
  #
  # If your dtype is :object and you are converting from :dense to a sparse type, it is recommended that you
  # provide a :default, as 0 may behave differently from its Float, Rational, or Complex equivalent. If no option
  # is given, Fixnum 0 will be used.
  def cast(*params)
    if (params.size > 0 && params[0].is_a?(Hash))
      opts = {
          :stype => self.stype,
          :dtype => self.dtype,
          :default => self.stype == :dense ? 0 : self.default_value
      }.merge(params[0])

      self.cast_full(opts[:stype], opts[:dtype], opts[:default])
    else
      params << self.stype if params.size == 0
      params << self.dtype if params.size == 1
      params << (self.stype == :dense ? 0 : self.default_value) if params.size == 2

      self.cast_full(*params)
    end

  end

  #
  # call-seq:
  #     rows -> Integer
  #
  # This shortcut use #shape to return the number of rows (the first dimension)
  # of the matrix.
  #
  def rows
    shape[0]
  end

  #
  # call-seq:
  #     cols -> Integer
  #
  # This shortcut use #shape to return the number of columns (the second
  # dimension) of the matrix.
  #
  def cols
    shape[1]
  end

  #
  # call-seq:
  #     to_hash -> Hash
  #
  # Create a Ruby Hash from an NMatrix.
  #
  def to_hash
    if stype == :yale
      h = {}
      each_stored_with_indices do |val,i,j|
        next if val == 0 # Don't bother storing the diagonal zero values -- only non-zeros.
        if h.has_key?(i)
          h[i][j] = val
        else
          h[i] = {j => val}
        end
      end
      h
    else # dense and list should use a C internal function.
      # FIXME: Write a C internal to_h function.
      m = stype == :dense ? self.cast(:list, self.dtype) : self
      m.__list_to_hash__
    end
  end
  alias :to_h :to_hash


  def inspect #:nodoc:
    original_inspect = super()
    original_inspect = original_inspect[0...original_inspect.size-1]
    original_inspect + " " + inspect_helper.join(" ") + ">"
  end

  def __yale_ary__to_s(sym) #:nodoc:
    ary = self.send("__yale_#{sym.to_s}__".to_sym)

    '[' + ary.collect { |a| a ? a : 'nil'}.join(',') + ']'
  end


  ##
  # call-seq: 
  #   integer_dtype?() -> Boolean
  #
  # Checks if dtype is an integer type
  #
  def integer_dtype?
    [:byte, :int8, :int16, :int32, :int64].include?(self.dtype)
  end


  #
  # call-seq:
  #     to_f -> Float
  #
  # Converts an nmatrix with a single element (but any number of dimensions)
  #  to a float.
  #
  # Raises an IndexError if the matrix does not have just a single element.
  #
  def to_f
    raise IndexError, 'to_f only valid for matrices with a single element' unless shape.all? { |e| e == 1 }
    self[*Array.new(shape.size, 0)]
  end

  #
  # call-seq:
  #     to_flat_array -> Array
  #     to_flat_a -> Array
  #
  # Converts an NMatrix to a one-dimensional Ruby Array.
  #
  def to_flat_array
    ary = Array.new(self.size)
    self.each.with_index { |v,i| ary[i] = v }
    ary
  end
  alias :to_flat_a :to_flat_array

  #
  # call-seq:
  #     size -> Fixnum
  #
  # Returns the total size of the NMatrix based on its shape.
  #
  def size
    s = self.shape
    (0...self.dimensions).inject(1) { |x,i| x * s[i] }
  end


  def to_s #:nodoc:
    self.to_flat_array.to_s
  end

  #
  # call-seq:
  #     nvector? -> true or false
  #
  # Shortcut function for determining whether the effective dimension is less than the dimension.
  # Useful when we take slices of n-dimensional matrices where n > 2.
  #
  def nvector?
    self.effective_dim < self.dim
  end

  #
  # call-seq:
  #     vector? -> true or false
  #
  # Shortcut function for determining whether the effective dimension is 1. See also #nvector?
  #
  def vector?
    self.effective_dim == 1
  end


  #
  # call-seq:
  #     to_a -> Array
  #
  # Converts an NMatrix to an array of arrays, or an NMatrix of effective dimension 1 to an array.
  #
  # Does not yet work for dimensions > 2
  def to_a(dimen=nil)
    if self.dim == 2

      return self.to_flat_a if self.shape[0] == 1

      ary = []
      begin
        self.each_row do |row|
          ary << row.to_flat_a
        end
      #rescue NotImplementedError # Oops. Try copying instead
      #  self.each_row(:copy) do |row|
      #    ary << row.to_a.flatten
      #  end
      end
      ary
    else
      to_a_rec(0)
    end
  end


  #
  # call-seq:
  #     rank(dimension, row_or_column_number) -> NMatrix
  #     rank(dimension, row_or_column_number, :reference) -> NMatrix reference slice
  #
  # Returns the rank (e.g., row, column, or layer) specified, using slicing by copy as default.
  #
  # See @row (dimension = 0), @column (dimension = 1)
  def rank(shape_idx, rank_idx, meth = :copy)

    params = Array.new(self.dim)
    params.each.with_index do |v,d|
      params[d] = d == shape_idx ? rank_idx : 0...self.shape[d]
    end

    meth == :reference ? self[*params] : self.slice(*params)
  end

  #
  # call-seq:
  #     column(column_number) -> NMatrix
  #     column(column_number, get_by) -> NMatrix
  #
  # Returns the column specified. Uses slicing by copy as default.
  #
  # * *Arguments* :
  #   - +column_number+ -> Integer.
  #   - +get_by+ -> Type of slicing to use, +:copy+ or +:reference+.
  # * *Returns* :
  #   - A NMatrix representing the requested column as a column vector.
  #
  # Examples:
  #
  #   m = NMatrix.new(2, [1, 4, 9, 14], :int32) # =>  1   4
  #                                                   9  14
  #
  #   m.column(1) # =>   4
  #                     14
  #
  def column(column_number, get_by = :copy)
    rank(1, column_number, get_by)
  end

  alias :col :column

  #
  # call-seq:
  #     row(row_number) -> NMatrix
  #     row(row_number, get_by) -> NMatrix
  #
  # * *Arguments* :
  #   - +row_number+ -> Integer.
  #   - +get_by+ -> Type of slicing to use, +:copy+ or +:reference+.
  # * *Returns* :
  #   - A NMatrix representing the requested row as a row vector.
  #
  def row(row_number, get_by = :copy)
    rank(0, row_number, get_by)
  end

  #
  # call-seq:
  #     layer(layer_number) -> NMatrix
  #     row(layer_number, get_by) -> NMatrix
  #
  # * *Arguments* :
  #   - +layer_number+ -> Integer.
  #   - +get_by+ -> Type of slicing to use, +:copy+ or +:reference+.
  # * *Returns* :
  #   - A NMatrix representing the requested layer as a layer vector.
  #
  def layer(layer_number, get_by = :copy)
    rank(2, layer_number, get_by)
  end



  #
  # call-seq:
  #     shuffle! -> ...
  #     shuffle!(random: rng) -> ...
  #
  # Re-arranges the contents of an NVector.
  #
  # TODO: Write more efficient version for Yale, list.
  # TODO: Generalize for more dimensions.
  def shuffle!(*args)
    method_missing(:shuffle!, *args) if self.effective_dim > 1
    ary = self.to_flat_a
    ary.shuffle!(*args)
    ary.each.with_index { |v,idx| self[idx] = v }
    self
  end


  #
  # call-seq:
  #     shuffle -> ...
  #     shuffle(rng) -> ...
  #
  # Re-arranges the contents of an NVector.
  #
  # TODO: Write more efficient version for Yale, list.
  # TODO: Generalize for more dimensions.
  def shuffle(*args)
    method_missing(:shuffle!, *args) if self.effective_dim > 1
    t = self.clone
    t.shuffle!(*args)
  end


  def method_missing name, *args, &block #:nodoc:
    if name.to_s =~ /^__list_elementwise_.*__$/
      raise NotImplementedError, "requested undefined list matrix element-wise operation"
    elsif name.to_s =~ /^__yale_scalar_.*__$/
      raise NotImplementedError, "requested undefined yale scalar element-wise operation"
    else
      super(name, *args, &block)
    end
  end


  def respond_to?(method) #:nodoc:
    if [:shuffle, :shuffle!, :each_with_index].include?(method.intern) # vector-only methods
      return vector?
    elsif [:each_layer, :layer].include?(method.intern) # 3-or-more dimensions only
      return dim > 2
    else
      super(method)
    end
  end


  # This is how you write an individual element-wise operation function:
  #def __list_elementwise_add__ rhs
  #  self.__list_map_merged_stored__(rhs){ |l,r| l+r }.cast(self.stype, NMatrix.upcast(self.dtype, rhs.dtype))
  #end

  def inspect_helper #:nodoc:
    ary = []
    ary << "shape:[#{shape.join(',')}]" << "dtype:#{dtype}" << "stype:#{stype}"

    if stype == :yale
      ary <<	"capacity:#{capacity}"

      # These are enabled by the DEBUG_YALE compiler flag in extconf.rb.
      if respond_to?(:__yale_a__)
        ary << "ija:#{__yale_ary__to_s(:ija)}" << "ia:#{__yale_ary__to_s(:ia)}" <<
          "ja:#{__yale_ary__to_s(:ja)}" << "a:#{__yale_ary__to_s(:a)}" << "d:#{__yale_ary__to_s(:d)}" <<
          "lu:#{__yale_ary__to_s(:lu)}" << "yale_size:#{__yale_size__}"
      end

    end

    ary
  end


  # Helper for converting a matrix into an array of arrays recursively
  def to_a_rec(dimen = 0) #:nodoc:
    return self.flat_map { |v| v } if dimen == self.dim-1

    ary = []
    self.each_rank(dimen) do |sect|
      ary << sect.to_a_rec(dimen+1)
    end
    ary
  end
end

require_relative './shortcuts.rb'
require_relative './math.rb'
require_relative './enumerate.rb'