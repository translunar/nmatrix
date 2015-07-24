# This is the main ruby file for the nmatrix-lapack gem
require 'nmatrix' #need to have nmatrix required first or else bad things will happen
require_relative 'lapack_ext_common'

NMatrix.register_lapack_extension("nmatrix-lapack")

require "nmatrix_lapack.so"

class NMatrix
  #Add functions from the LAPACKE C extension to the main LAPACK and BLAS modules.
  #This will overwrite the original functions where applicable.
  module LAPACK
    class << self
      NMatrix::LAPACKE::LAPACK.singleton_methods.each do |m|
        define_method m, NMatrix::LAPACKE::LAPACK.method(m).to_proc
      end
    end
  end

  module BLAS
    class << self
      NMatrix::LAPACKE::BLAS.singleton_methods.each do |m|
        define_method m, NMatrix::LAPACKE::BLAS.method(m).to_proc
      end
    end
  end

  def getrf!
    raise(StorageTypeError, "ATLAS functions only work on dense matrices") unless self.dense?

    ipiv = NMatrix::LAPACK::lapacke_getrf(:row, self.shape[0], self.shape[1], self, self.shape[1])

    return ipiv
  end
end
