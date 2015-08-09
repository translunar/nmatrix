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
# SciRuby is Copyright (c) 2010 - 2014, Ruby Science Foundation
# NMatrix is Copyright (c) 2012 - 2014, John Woods and the Ruby Science Foundation
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
# == lapack_core.rb
#
# This file contains friendlier interfaces to LAPACK functions
# implemented in C.
# This file is only for functions available with the core nmatrix gem
# (no external libraries needed).
#
# Note: most of these functions are borrowed from ATLAS, which is available under a BSD-
# style license.
#++

class NMatrix

  module LAPACK

    #Add functions from C extension to main LAPACK module
    class << self
      NMatrix::Internal::LAPACK.singleton_methods.each do |m|
        define_method m, NMatrix::Internal::LAPACK.method(m).to_proc
      end
    end

    class << self
      # Solve the matrix equation AX = B, where A is a symmetric (or Hermitian)
      # positive-definite matrix. If A is a nxn matrix, B must be mxn.
      # Depending on the value of uplo, only the upper or lower half of +a+
      # is read.
      # This uses the Cholesky decomposition so it should be faster than
      # the generic NMatrix#solve method.
      # Doesn't modify inputs.
      # Requires either the nmatrix-atlas or nmatrix-lapacke gem.
      # * *Arguments* :
      #   - +uplo+ -> Either +:upper+ or +:lower+. Specifies which half of +a+ to read.
      #   - +a+ -> The matrix A.
      #   - +b+ -> The right-hand side B.
      # * *Returns* :
      #   - The solution X
      def posv(uplo, a, b)
        raise(NotImplementedError, "Either the nmatrix-atlas or nmatrix-lapacke gem must be installed to use posv")
      end

      #     laswp(matrix, ipiv) -> NMatrix
      #
      # Permute the columns of a matrix (in-place) according to the Array +ipiv+.
      #
      def laswp(matrix, ipiv)
        raise(ArgumentError, "expected NMatrix for argument 0") unless matrix.is_a?(NMatrix)
        raise(StorageTypeError, "LAPACK functions only work on :dense NMatrix instances") unless matrix.stype == :dense
        raise(ArgumentError, "expected Array ipiv to have no more entries than NMatrix a has columns") if ipiv.size > matrix.shape[1]

        clapack_laswp(matrix.shape[0], matrix, matrix.shape[1], 0, ipiv.size-1, ipiv, 1)
      end

      def alloc_svd_result(matrix)
        [
          NMatrix.new(matrix.shape[0], dtype: matrix.dtype),
          NMatrix.new([[matrix.shape[0],matrix.shape[1]].min,1], dtype: matrix.dtype),
          NMatrix.new(matrix.shape[1], dtype: matrix.dtype)
        ]
      end


      #
      # call-seq:
      #     gesvd(matrix) -> [u, sigma, v_transpose]
      #     gesvd(matrix) -> [u, sigma, v_conjugate_transpose] # complex
      #
      # Compute the singular value decomposition of a matrix using LAPACK's GESVD function.
      #
      # Optionally accepts a +workspace_size+ parameter, which will be honored only if it is larger than what LAPACK
      # requires.
      #
      def gesvd(matrix, workspace_size=1)
        result = alloc_svd_result(matrix)
        # The LAPACK functions *gesvd expect a column-major matrix and our
        # matrices are stored row-major. However, all is not lost!
        # Our m x n matrix, M, stored row-major is equivalent to the n x m
        # transpose, M^T, stored column-major.
        # So we get LAPACK to give us the SVD of M^T.
        # Given an SVD of M^T, we can easily get a SVD of M:
        # M^T = U S V^(*T)
        # M = V^* S^T U^T
        # V^* and U^T will still be unitary and S^T will still be diagonal, so this satisfies all the criteria for a SVD
        NMatrix::LAPACK::lapack_gesvd(:a, :a, matrix.shape[1], matrix.shape[0], matrix, matrix.shape[1], result[1], result[2], matrix.shape[1], result[0], matrix.shape[0], workspace_size)
        result
      end

      #
      # call-seq:
      #     gesdd(matrix) -> [u, sigma, v_transpose]
      #     gesdd(matrix) -> [u, sigma, v_conjugate_transpose] # complex
      #
      # Compute the singular value decomposition of a matrix using LAPACK's GESDD function. This uses a divide-and-conquer
      # strategy. See also #gesvd.
      #
      # Optionally accepts a +workspace_size+ parameter, which will be honored only if it is larger than what LAPACK
      # requires.
      #
      def gesdd(matrix, workspace_size=nil)
        min_workspace_size = matrix.shape.min * (6 + 4 * matrix.shape.min) + matrix.shape.max
        workspace_size = min_workspace_size if workspace_size.nil? || workspace_size < min_workspace_size
        result = alloc_svd_result(matrix)
        NMatrix::LAPACK::lapack_gesdd(:a, matrix.shape[1], matrix.shape[0], matrix, matrix.shape[1], result[1], result[2], matrix.shape[1], result[0], matrix.shape[0], workspace_size)
        result
      end

      def alloc_evd_result(matrix)
        [
          NMatrix.new(matrix.shape[0], dtype: matrix.dtype),
          NMatrix.new(matrix.shape[0], dtype: matrix.dtype),
          NMatrix.new([matrix.shape[0],1], dtype: matrix.dtype),
          NMatrix.new([matrix.shape[0],1], dtype: matrix.dtype),
        ]
      end


      #
      # call-seq:
      #     geev(matrix) -> [eigenvalues, left_eigenvectors, right_eigenvectors]
      #     geev(matrix, :left) -> [eigenvalues, left_eigenvectors]
      #     geev(matrix, :right) -> [eigenvalues, right_eigenvectors]
      #
      # Perform eigenvalue decomposition on a matrix using LAPACK's xGEEV function.
      #
      # +eigenvalues+ is a n-by-1 NMatrix containing the eigenvalues.
      #
      # +right_eigenvalues+ is a n-by-n matrix such that its j'th column
      # contains the (right) eigenvalue of +matrix+ corresponding
      # to the j'th eigenvalue.
      # This means that +matrix+ = RDR^(-1),
      # where R is +right_eigenvalues+ and D is the diagonal matrix formed
      # from +eigenvalues+.
      #
      # +left_eigenvalues+ is n-by-n and its columns are the left
      # eigenvalues of +matrix+, using the {definition of left eigenvalue
      # from LAPACK}[https://software.intel.com/en-us/node/521147].
      #
      # For real dtypes, +eigenvalues+ and the eigenvector matrices
      # will be complex if and only if +matrix+ has complex eigenvalues.
      #
      # Only available if nmatrix-lapack or nmatrix-atlas is installed.
      #
      def geev(matrix, which=:both)
        raise(NotImplementedError, "geev requires either the nmatrix-atlas or nmatrix-lapack gem")
      end

    end
  end
end
