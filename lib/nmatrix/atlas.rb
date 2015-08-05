require 'nmatrix' #need to have nmatrix required first or else bad things will happen
require_relative 'lapack_ext_common'

NMatrix.register_lapack_extension("nmatrix-atlas")

require "nmatrix_atlas.so"

class NMatrix

  #Add functions from the ATLAS C extension to the main LAPACK and BLAS modules.
  #This will overwrite the original functions where applicable.
  module LAPACK
    class << self
      NMatrix::ATLAS::LAPACK.singleton_methods.each do |m|
        define_method m, NMatrix::ATLAS::LAPACK.method(m).to_proc
      end
    end
  end

  module BLAS
    class << self
      NMatrix::ATLAS::BLAS.singleton_methods.each do |m|
        define_method m, NMatrix::ATLAS::BLAS.method(m).to_proc
      end
    end
  end

  module LAPACK
    class << self
      def posv(uplo, a, b)
        raise(ShapeError, "a must be square") unless a.dim == 2 && a.shape[0] == a.shape[1]
        raise(ShapeError, "number of rows of b must equal number of cols of a") unless a.shape[1] == b.shape[0]
        raise(StorageTypeError, "only works with dense matrices") unless a.stype == :dense && b.stype == :dense
        raise(DataTypeError, "only works for non-integer, non-object dtypes") if 
          a.integer_dtype? || a.object_dtype? || b.integer_dtype? || b.object_dtype?

        x     = b.clone
        clone = a.clone
        n = a.shape[0]
        nrhs = b.shape[1]
        clapack_potrf(:row, uplo, n, clone, n)
        # Must transpose b before and after: http://math-atlas.sourceforge.net/faq.html#RowSolve
        x = x.transpose
        clapack_potrs(:row, uplo, n, nrhs, clone, n, x, n)
        x = x.transpose
      end
    end
  end

  def invert!
    raise(StorageTypeError, "invert only works on dense matrices currently") unless self.dense?
    raise(ShapeError, "Cannot invert non-square matrix") unless shape[0] == shape[1]
    raise(DataTypeError, "Cannot invert an integer matrix in-place") if self.integer_dtype?

    # Get the pivot array; factor the matrix
    # We can't used getrf! here since it doesn't have the clapack behavior,
    # so it doesn't play nicely with clapack_getri
    n = self.shape[0]
    pivot = NMatrix::LAPACK::clapack_getrf(:row, n, n, self, n)
    # Now calculate the inverse using the pivot array
    NMatrix::LAPACK::clapack_getri(:row, n, self, n, pivot)

    self
  end

  def potrf!(which)
    raise(StorageTypeError, "ATLAS functions only work on dense matrices") unless self.dense?
    raise(ShapeError, "Cholesky decomposition only valid for square matrices") unless self.dim == 2 && self.shape[0] == self.shape[1]

    NMatrix::LAPACK::clapack_potrf(:row, which, self.shape[0], self, self.shape[1])
  end

end
