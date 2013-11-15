#!/bin/bash

VERSION="4.7.2" # Script should also work with GCC 4.7.2.
PREFIX="/usr/gcc-${VERSION}"
LANGUAGES="c,c++,fortran"
MAKE="make -j 4"

brew-path() { brew info $1 | head -n3 | tail -n1 | cut -d' ' -f1; }

# Prerequisites

brew install gmp mpfr libmpc

# Try using the following for GCC 4.9:
#
#   brew install gmp4 mpfr2 libmpc08 isl011 cloog018
#
#

# For Mavericks, you may also need to install isl and cloog. I couldn't get this script to work
# in Mavericks (at least for GCC 4.8.2), but was able to do:
#
#   brew install gcc49 --enable-fortran
#
# Source for this is: http://stackoverflow.com/questions/19535422/os-x-10-9-gcc-links-to-clang
#
# Note that NMatrix doesn't currently require Fortran, but dependent libraries may eventually.
# If you wanted to install isl and cloog, you'd need to do the following:
#
#   brew install isl
#
# if you uncomment the above line, also add --with-isl=$(brew --prefix isl)
# to the configure command below. cloog is:
#
#   brew install cloog  # this needs --with-cloog=$(brew --prefix cloog)
#


# Next, download & install the latest GCC:

mkdir -p $PREFIX
mkdir temp-gcc
cd temp-gcc
wget ftp://ftp.gnu.org/gnu/gcc/gcc-$VERSION/gcc-$VERSION.tar.gz
tar xfz gcc-$VERSION.tar.gz
rm gcc-$VERSION.tar.gz
cd gcc-$VERSION

mkdir build
cd build

# Older versions of brew need brew-path instead of brew --prefix.
../configure \
     --prefix=$PREFIX \
     --with-gmp=$(brew --prefix gmp) \
     --with-mpfr=$(brew --prefix mpfr) \
     --with-mpc=$(brew --prefix libmpc) \
     --program-suffix=-$VERSION \
     --enable-languages=$LANGUAGES \
     --with-system-zlib \
     --enable-stage1-checking \
     --enable-plugin \
     --enable-lto \
     --disable-multilib

$MAKE bootstrap

make install

# Uncomment for cleanup …
# cd ../../..
# rm -r temp-gcc
