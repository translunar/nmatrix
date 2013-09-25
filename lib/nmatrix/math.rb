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
# == math.rb
#
# Math functionality for NMatrix, along with any NMatrix instance
# methods that correspond to ATLAS/BLAS/LAPACK functions (e.g.,
# laswp).
#++

class NMatrix

  #
  # Define the set of math methods present in the ruby core Math module for NMatrix objects.
  # 
  # Dtype is preserved, so integer types will produce integer results.
  #
  # Methods that do not return a single number (frexp and lgamma) are not implemented since
  # this would break dtype preservation.
  #
  # The module is added as class methods of NMatrix, so use is, e.g.:
  #
  # a = N[0.0, 1.0, 2.0]
  # >> puts N.sin(a)
  # => [0.0, 0.8414709848078965, 0.9092974268256817]
  #
  module NMMath
    METHODS_ARITY_2 = [:atan2, :ldexp, :hypot]
    METHODS_ARITY_1 = [:cos, :sin, :tan, :acos, :asin, :atan, :cosh, :sinh, :tanh, :acosh,
      :asinh, :atanh, :exp, :log2, :log10, :sqrt, :cbrt, :erf, :erfc, :gamma]

    METHODS_ARITY_1.each do |meth|
      define_method meth do |nm|
        nm.map { |e| Math.send(meth, e) }
      end
    end

    METHODS_ARITY_2.each do |meth|
      define_method meth do |nm0, nm1|
        nm_out = nm0.dup
        nm0.each_with_indices do |a|
          e = a[0]
          i = a[1...a.length]
          nm_out[*i] = Math.send(meth, e, nm1[*i]) 
        end
        nm_out
      end
    end

    #
    # call-seq:
    #   log(nm) -> NMatrix
    #   log(nm, base) -> NMatrix
    #
    # Calculates the natural logarithm elementwise of the supplied NMatrix.
    # If a base is supplied, calculate the logarithm with that base.
    #
    # @param [NMatrix] nm  the NMatrix of which to calculate the log
    # @param [Numeric] base the base of the logarithm (default: natural logarithm)
    #
    # @return [NMatrix] an NMatrix with log elements
    #
    def log(nm, base=Math::E)
      nm.map { |e| Math.log(e, base) }
    end

  end

  # self.extend NMMath # add the Math functions as class methods of NMatrix

  #
  # call-seq:
  #     invert! -> NMatrix
  #
  # Use LAPACK to calculate the inverse of the matrix (in-place). Only works on
  # dense matrices.
  #
  # Note: If you don't have LAPACK, e.g., on a Mac, this may not work yet.
  #
  def invert!
    # Get the pivot array; factor the matrix
    pivot = self.getrf!

    # Now calculate the inverse using the pivot array
    NMatrix::LAPACK::clapack_getri(:row, self.shape[1], self, self.shape[1], pivot)

    self
  end

  #
  # call-seq:
  #     invert -> NMatrix
  #
  # Make a copy of the matrix, then invert it (requires LAPACK).
  #
  # * *Returns* :
  #   - A dense NMatrix.
  #
  def invert
    self.cast(:dense, self.dtype).invert!
  end
  alias :inverse :invert

  #
  # call-seq:
  #     getrf! -> NMatrix
  #
  # LU factorization of a general M-by-N matrix +A+ using partial pivoting with
  # row interchanges. Only works in dense matrices.
  #
  # * *Returns* :
  #   - The IPIV vector. The L and U matrices are stored in A.
  # * *Raises* :
  #   - +StorageTypeError+ -> ATLAS functions only work on dense matrices.
  #
  def getrf!
    raise(StorageTypeError, "ATLAS functions only work on dense matrices") unless self.stype == :dense
    NMatrix::LAPACK::clapack_getrf(:row, self.shape[0], self.shape[1], self, self.shape[1])
  end

  #
  # call-seq:
  #     factorize_lu -> ...
  #
  # LU factorization of a matrix.
  #
  # FIXME: For some reason, getrf seems to require that the matrix be transposed first -- and then you have to transpose the
  # FIXME: result again. Ideally, this would be an in-place factorize instead, and would be called nm_factorize_lu_bang.
  #
  def factorize_lu
    raise(NotImplementedError, "only implemented for dense storage") unless self.stype == :dense
    raise(NotImplementedError, "matrix is not 2-dimensional") unless self.dimensions == 2

    t = self.transpose
    NMatrix::LAPACK::clapack_getrf(:row, t.shape[0], t.shape[1], t, t.shape[1])
    t.transpose
  end

  #
  # call-seq:
  #     gesvd! -> [u, sigma, v_transpose]
  #     gesvd! -> [u, sigma, v_conjugate_transpose] # complex
  #
  # Compute the singular value decomposition of a matrix using LAPACK's GESVD function. 
  # This is destructive, modifying the source NMatrix.  See also #gesdd.
  #
  # Optionally accepts a +workspace_size+ parameter, which will be honored only if it is larger than what LAPACK
  # requires.
  #
  def gesvd!(workspace_size=1)
    NMatrix::LAPACK::gesvd(self, workspace_size)
  end

  #
  # call-seq:
  #     gesvd -> [u, sigma, v_transpose]
  #     gesvd -> [u, sigma, v_conjugate_transpose] # complex
  #
  # Compute the singular value decomposition of a matrix using LAPACK's GESVD function.
  #
  # Optionally accepts a +workspace_size+ parameter, which will be honored only if it is larger than what LAPACK
  # requires.
  #
  def gesvd(workspace_size=1)
    self.clone.gesvd!(workspace_size)
  end



  #
  # call-seq:
  #     gesdd! -> [u, sigma, v_transpose]
  #     gesdd! -> [u, sigma, v_conjugate_transpose] # complex
  #
  # Compute the singular value decomposition of a matrix using LAPACK's GESDD function. This uses a divide-and-conquer
  # strategy. This is destructive, modifying the source NMatrix.  See also #gesvd.
  #
  # Optionally accepts a +workspace_size+ parameter, which will be honored only if it is larger than what LAPACK
  # requires.
  #
  def gesdd!(workspace_size=1)
    NMatrix::LAPACK::gesdd(self, workspace_size)
  end

  #
  # call-seq:
  #     gesdd -> [u, sigma, v_transpose]
  #     gesdd -> [u, sigma, v_conjugate_transpose] # complex
  #
  # Compute the singular value decomposition of a matrix using LAPACK's GESDD function. This uses a divide-and-conquer
  # strategy. See also #gesvd.
  #
  # Optionally accepts a +workspace_size+ parameter, which will be honored only if it is larger than what LAPACK
  # requires.
  #
  def gesdd(workspace_size=1)
    self.clone.gesdd!(workspace_size)
  end
  #
  # call-seq:
  #     laswp!(ary) -> NMatrix
  #
  # In-place permute the columns of a dense matrix using LASWP according to the order given in an Array +ary+.
  # Not yet implemented for yale or list.
  def laswp!(ary)
    NMatrix::LAPACK::laswp(self, ary)
  end

  #
  # call-seq:
  #     laswp(ary) -> NMatrix
  #
  # Permute the columns of a dense matrix using LASWP according to the order given in an Array +ary+.
  # Not yet implemented for yale or list.
  def laswp(ary)
    self.clone.laswp!(ary)
  end

  #
  # call-seq:
  #     det -> determinant
  #
  # Calculate the determinant by way of LU decomposition. This is accomplished
  # using clapack_getrf, and then by summing the diagonal elements. There is a
  # risk of underflow/overflow.
  #
  # There are probably also more efficient ways to calculate the determinant.
  # This method requires making a copy of the matrix, since clapack_getrf
  # modifies its input.
  #
  # For smaller matrices, you may be able to use +#det_exact+.
  #
  # This function is guaranteed to return the same type of data in the matrix
  # upon which it is called.
  # In other words, if you call it on a rational matrix, you'll get a rational
  # number back.
  #
  # Integer matrices are converted to rational matrices for the purposes of
  # performing the calculation, as xGETRF can't work on integer matrices.
  #
  # * *Returns* :
  #   - The determinant of the matrix. It's the same type as the matrix's dtype.
  # * *Raises* :
  #   - +NotImplementedError+ -> Must be used in 2D matrices.
  #
  def det
    raise(NotImplementedError, "determinant can be calculated only for 2D matrices") unless self.dim == 2

    # Cast to a dtype for which getrf is implemented
    new_dtype = [:byte,:int8,:int16,:int32,:int64].include?(self.dtype) ? :rational128 : self.dtype
    copy = self.cast(:dense, new_dtype)

    # Need to know the number of permutations. We'll add up the diagonals of
    # the factorized matrix.
    pivot = copy.getrf!

    prod = pivot.size % 2 == 1 ? -1 : 1 # odd permutations => negative
    [shape[0],shape[1]].min.times do |i|
      prod *= copy[i,i]
    end

    # Convert back to an integer if necessary
    new_dtype != self.dtype ? prod.to_i : prod
  end

  #
  # call-seq:
  #     complex_conjugate -> NMatrix
  #     complex_conjugate(new_stype) -> NMatrix
  #
  # Get the complex conjugate of this matrix. See also complex_conjugate! for
  # an in-place operation (provided the dtype is already +:complex64+ or
  # +:complex128+).
  #
  # Doesn't work on list matrices, but you can optionally pass in the stype you
  # want to cast to if you're dealing with a list matrix.
  #
  # * *Arguments* :
  #   - +new_stype+ -> stype for the new matrix.
  # * *Returns* :
  #   - If the original NMatrix isn't complex, the result is a +:complex128+ NMatrix. Otherwise, it's the original dtype.
  #
  def complex_conjugate(new_stype = self.stype)
    self.cast(new_stype, NMatrix::upcast(dtype, :complex64)).complex_conjugate!
  end

  #
  # call-seq:
  #     conjugate_transpose -> NMatrix
  #
  # Calculate the conjugate transpose of a matrix. If your dtype is already
  # complex, this should only require one copy (for the transpose).
  #
  # * *Returns* :
  #   - The conjugate transpose of the matrix as a copy.
  #
  def conjugate_transpose
    self.transpose.complex_conjugate!
  end

  #
  # call-seq:
  #     hermitian? -> Boolean
  #
  # A hermitian matrix is a complex square matrix that is equal to its
  # conjugate transpose. (http://en.wikipedia.org/wiki/Hermitian_matrix)
  #
  # * *Returns* :
  #   - True if +self+ is a hermitian matrix, nil otherwise.
  #
  def hermitian?
    return false if self.dim != 2 or self.shape[0] != self.shape[1]

    if [:complex64, :complex128].include?(self.dtype)
      # TODO: Write much faster Hermitian test in C
      self.eql?(self.conjugate_transpose)
    else
      symmetric?
    end
  end

  ##
  # call-seq:
  #   mean() -> NMatrix
  #   mean(dimen) -> NMatrix
  #
  # Calculates the mean along the specified dimension.
  #
  # This will force integer types to float64 dtype.
  #
  # @see #inject_rank
  #
  def mean(dimen=0)
    reduce_dtype = nil
    if integer_dtype? then
      reduce_dtype = :float64
    end
    inject_rank(dimen, 0.0, reduce_dtype) do |mean, sub_mat|
      mean + sub_mat/shape[dimen]
    end
  end

  ##
  # call-seq:
  #   sum() -> NMatrix
  #   sum(dimen) -> NMatrix
  #
  # Calculates the sum along the specified dimension.
  #
  # @see #inject_rank
  def sum(dimen=0)
    inject_rank(dimen, 0.0) do |sum, sub_mat|
      sum + sub_mat
    end
  end


  ##
  # call-seq:
  #   min() -> NMatrix
  #   min(dimen) -> NMatrix
  #
  # Calculates the minimum along the specified dimension.
  #
  # @see #inject_rank
  #
  def min(dimen=0)
    inject_rank(dimen) do |min, sub_mat|
      if min.is_a? NMatrix then
        min * (min <= sub_mat).cast(self.stype, self.dtype) + ((min)*0.0 + (min > sub_mat).cast(self.stype, self.dtype)) * sub_mat
      else
        min <= sub_mat ? min : sub_mat
      end
    end
  end

  ##
  # call-seq:
  #   max() -> NMatrix
  #   max(dimen) -> NMatrix
  #
  # Calculates the maximum along the specified dimension.
  #
  # @see #inject_rank
  #
  def max(dimen=0)
    inject_rank(dimen) do |max, sub_mat|
      if max.is_a? NMatrix then
        max * (max >= sub_mat).cast(self.stype, self.dtype) + ((max)*0.0 + (max < sub_mat).cast(self.stype, self.dtype)) * sub_mat
      else
        max >= sub_mat ? max : sub_mat
      end
    end
  end


  ##
  # call-seq:
  #   variance() -> NMatrix
  #   variance(dimen) -> NMatrix
  #
  # Calculates the sample variance along the specified dimension.
  #
  # This will force integer types to float64 dtype.
  #
  # @see #inject_rank
  #
  def variance(dimen=0)
    reduce_dtype = nil
    if integer_dtype? then
      reduce_dtype = :float64
    end
    m = mean(dimen)
    inject_rank(dimen, 0.0, reduce_dtype) do |var, sub_mat|
      var + (m - sub_mat)*(m - sub_mat)/(shape[dimen]-1)
    end
  end

  ##
  # call-seq:
  #   std() -> NMatrix
  #   std(dimen) -> NMatrix
  #
  #
  # Calculates the sample standard deviation along the specified dimension.
  #
  # This will force integer types to float64 dtype.
  #
  # @see #inject_rank
  #
  def std(dimen=0)
    variance(dimen).map! { |e| Math.sqrt(e) }
  end


  #
  # call-seq:
  #     abs_dtype -> Symbol
  #
  # Returns the dtype of the result of a call to #abs. In most cases, this is the same as dtype; it should only differ
  # for :complex64 (where it's :float32) and :complex128 (:float64).
  def abs_dtype
    if self.dtype == :complex64
      :float32
    elsif self.dtype == :complex128
      :float64
    else
      self.dtype
    end
  end


  #
  # call-seq:
  #     abs -> NMatrix
  #
  # Maps all values in a matrix to their absolute values.
  def abs
    if stype == :dense
      self.__dense_map__ { |v| v.abs }
    elsif stype == :list
      # FIXME: Need __list_map_stored__, but this will do for now.
      self.__list_map_merged_stored__(nil, nil) { |v,dummy| v.abs }
    else
      self.__yale_map_stored__ { |v| v.abs }
    end.cast(self.stype, abs_dtype)
  end

  alias :permute_columns  :laswp
  alias :permute_columns! :laswp!

protected
  # Define the element-wise operations for lists. Note that the __list_map_merged_stored__ iterator returns a Ruby Object
  # matrix, which we then cast back to the appropriate type. If you don't want that, you can redefine these functions in
  # your own code.
  {add: :+, sub: :-, mul: :*, div: :/, pow: :**, mod: :%}.each_pair do |ewop, op|
    define_method("__list_elementwise_#{ewop}__") do |rhs|
      self.__list_map_merged_stored__(rhs, nil) { |l,r| l.send(op,r) }.cast(stype, NMatrix.upcast(dtype, rhs.dtype))
    end
    define_method("__dense_elementwise_#{ewop}__") do |rhs|
      self.__dense_map_pair__(rhs) { |l,r| l.send(op,r) }.cast(stype, NMatrix.upcast(dtype, rhs.dtype))
    end
    define_method("__yale_elementwise_#{ewop}__") do |rhs|
      self.__yale_map_merged_stored__(rhs, nil) { |l,r| l.send(op,r) }.cast(stype, NMatrix.upcast(dtype, rhs.dtype))
    end
    define_method("__list_scalar_#{ewop}__") do |rhs|
      self.__list_map_merged_stored__(rhs, nil) { |l,r| l.send(op,r) }.cast(stype, NMatrix.upcast(dtype, NMatrix.min_dtype(rhs)))
    end
    define_method("__yale_scalar_#{ewop}__") do |rhs|
      self.__yale_map_stored__ { |l| l.send(op,rhs) }.cast(stype, NMatrix.upcast(dtype, NMatrix.min_dtype(rhs)))
    end
    define_method("__dense_scalar_#{ewop}__") do |rhs|
      self.__dense_map__ { |l| l.send(op,rhs) }.cast(stype, NMatrix.upcast(dtype, NMatrix.min_dtype(rhs)))
    end
  end

  # These don't actually take an argument -- they're called reverse-polish style on the matrix.
  # This group always gets casted to float64.
  %i{log log2 log10 sqrt}.each do |ewop|
    define_method("__list_unary_#{ewop}__") do
      self.__list_map_stored__(nil) { |l| Math.send(ewop, l) }.cast(stype, NMatrix.upcast(dtype, :float64))
    end
    define_method("__yale_unary_#{ewop}__") do
      self.__yale_map_stored__ { |l| Math.send(ewop, l) }.cast(stype, NMatrix.upcast(dtype, :float64))
    end
    define_method("__dense_unary_#{ewop}__") do
      self.__dense_map__ { |l| Math.send(ewop, l) }.cast(stype, NMatrix.upcast(dtype, :float64))
    end
  end

  # These don't actually take an argument -- they're called reverse-polish style on the matrix.
  # FIXME: Unclear how these should be cast, too busy to think about it right now.
  %i{sin cos tan acos asin atan cosh sinh tanh acosh asinh atanh exp}.each do |ewop|
    define_method("__list_unary_#{ewop}__") do
      self.__list_map_stored__(nil) { |l| Math.send(ewop, l) } #.cast(stype, NMatrix.upcast(dtype, :float64))
    end
    define_method("__yale_unary_#{ewop}__") do
      self.__yale_map_stored__ { |l| Math.send(ewop, l) } #.cast(stype, NMatrix.upcast(dtype, :float64))
    end
    define_method("__dense_unary_#{ewop}__") do
      self.__dense_map__ { |l| Math.send(ewop, l) } #.cast(stype, NMatrix.upcast(dtype, :float64))
    end
  end

  # These take two arguments. One might be a matrix, and one might be a scalar.=
=begin
  %i{atan2 ldexp hypot}.each do |ewop|
    define_method("__list_elementwise_#{ewop}__") do |rhs,order|
      if order
        rhs.__list_map_merged_stored__(self, nil) { |r,l| Math.send(ewop,l,r) }
      else
        self.__list_map_merged_stored__(rhs, nil) { |l,r| Math.send(ewop,l,r) }
      end.cast(stype, NMatrix.upcast(dtype, rhs.dtype))
    end

    define_method("__dense_elementwise_#{ewop}__") do |rhs,order|
      if order
        rhs.__dense_map_pair__(self) { |r,l| Math.send(ewop,l,r) }
      else
        self.__dense_map_pair__(rhs) { |l,r| Math.send(ewop,l,r) }
      end.cast(stype, NMatrix.upcast(dtype, rhs.dtype))
    end

    define_method("__yale_elementwise_#{ewop}__") do |rhs,order|
      if order
        rhs.__yale_map_merged_stored__(self, nil) { |r,l| Math.send(ewop,l,r) }
      else
        self.__yale_map_merged_stored__(rhs, nil) { |l,r| Math.send(ewop,l,r) }
      end.cast(stype, NMatrix.upcast(dtype, rhs.dtype))
    end

    define_method("__list_scalar_#{ewop}__") do |rhs,order|
      if order
        rhs.__list_map_stored__ { |r| Math.send(ewop, self, r) }
      else
        self.__list_map_stored__ { |l| Math.send(ewop, l, rhs) }
      end.cast(stype, NMatrix.upcast(dtype, NMatrix.min_dtype(rhs)))
    end

    define_method("__yale_scalar_#{ewop}__") do |rhs,order|
      if order
        rhs.__yale_map_stored__ { |r| Math.send(ewop, self, r) }
      else
        self.__yale_map_stored__ { |l| Math.send(ewop, l, rhs) }
      end.cast(stype, NMatrix.upcast(dtype, NMatrix.min_dtype(rhs)))
    end

    define_method("__dense_scalar_#{ewop}__") do |rhs,order|
      if order
        rhs.__dense_map__ { |r| Math.send(ewop, self, r) }
      else
        self.__dense_map__ { |l| Math.send(ewop, l, rhs) }
      end.cast(stype, NMatrix.upcast(dtype, NMatrix.min_dtype(rhs)))
    end
  end
=end

  # Equality operators do not involve a cast. We want to get back matrices of TrueClass and FalseClass.
  {eqeq: :==, neq: :!=, lt: :<, gt: :>, leq: :<=, geq: :>=}.each_pair do |ewop, op|
    define_method("__list_elementwise_#{ewop}__") do |rhs|
      self.__list_map_merged_stored__(rhs, nil) { |l,r| l.send(op,r) }
    end
    define_method("__dense_elementwise_#{ewop}__") do |rhs|
      self.__dense_map_pair__(rhs) { |l,r| l.send(op,r) }
    end
    define_method("__yale_elementwise_#{ewop}__") do |rhs|
      self.__yale_map_merged_stored__(rhs, nil) { |l,r| l.send(op,r) }
    end

    define_method("__list_scalar_#{ewop}__") do |rhs|
      self.__list_map_merged_stored__(rhs, nil) { |l,r| l.send(op,r) }
    end
    define_method("__yale_scalar_#{ewop}__") do |rhs|
      self.__yale_map_stored__ { |l| l.send(op,rhs) }
    end
    define_method("__dense_scalar_#{ewop}__") do |rhs|
      self.__dense_map__ { |l| l.send(op,rhs) }
    end
  end
end
