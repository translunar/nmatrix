#!/bin/bash

set -ev #fail at the first command that returns non-zero exit value

if [ "$1" = "before_install" ]
then
  gem install bundler -v '~> 1.6'
  sudo apt-get update -qq

  if [ -n "$USE_ATLAS" ]
  then
    sudo apt-get install -y libatlas-base-dev
  fi

  # travis-ci runs on Ubuntu 12.04, where the openblas package doesn't
  # provide a liblapack.so, so we test using the blas from openblas
  # and the reference lapack implementation. Not entirely sure if
  # this will work.
  if [ -n "$USE_OPENBLAS" ]
  then
    sudo apt-get install -y libopenblas-dev
    # Since we install libopenblas first, liblapack won't try to install
    # libblas (the reference BLAS implementation).
    sudo apt-get install -y liblapack-dev
  fi

  if [ -n "$USE_REF" ]
  then
    sudo apt-get install -y liblapack-dev
  fi
fi

if [ "$1" = "script" ]
then
  if [ -n "$USE_ATLAS" ]
  then
    # Need to put these commands on separate lines (rather than use &&)
    # so that bash set -e will work.
    bundle exec rake compile nmatrix_plugins=atlas
    bundle exec rake spec nmatrix_plugins=atlas
  fi

  if [ -n "$USE_OPENBLAS" ]
  then
    bundle exec rake compile nmatrix_plugins=lapacke
    bundle exec rake spec nmatrix_plugins=lapacke
  fi

  if [ -n "$USE_REF" ]
  then
    bundle exec rake compile nmatrix_plugins=lapacke
    bundle exec rake spec nmatrix_plugins=lapacke
  fi

  if [ -n "$NO_EXTERNAL_LIB" ]
  then
    bundle exec rake compile
    bundle exec rake spec
  fi
fi
