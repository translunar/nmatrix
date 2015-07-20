require 'nmatrix' #need to have nmatrix required first or else bad things will happen
require_relative 'lapack_ext_common'

NMatrix.register_lapack_extension("nmatrix-atlas")

require "nmatrix_atlas.so"

class NMatrix
  def test_return_3
    3
  end

  #Add functions from the ATLAS C extension to the main LAPACK and BLAS.
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
end
