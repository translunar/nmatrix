# This file `require`s either nmatrix-atlas or nmatrix-lapacke depending on which
# is available.
#
# The idea is that if a developer wants to use a LAPACK feature which is provided
# by both of these gems (e.g. NMatrix#potrf! or NMatrix::LAPACK.geev),
# but doesn't care which one is installed, they can
# just `require 'nmatrix/lapack_plugin'` rather than having to choose between
# `require 'nmatrix/lapacke'` or `require 'nmatrix/lapacke'` 

begin
  require 'nmatrix/atlas'
rescue LoadError
  begin
    require 'nmatrix/lapacke'
  rescue LoadError
    raise(LoadError,"Either nmatrix-atlas or nmatrix-lapacke must be installed")
  end
end
