# This is the main ruby file for the nmatrix-lapack gem
require 'nmatrix' #need to have nmatrix required first or else bad things will happen
require_relative 'lapack_ext_common'

NMatrix.register_lapack_extension("nmatrix-lapack")

require "nmatrix_lapack.so"
