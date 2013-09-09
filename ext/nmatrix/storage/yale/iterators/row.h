/////////////////////////////////////////////////////////////////////
// = NMatrix
//
// A linear algebra library for scientific computation in Ruby.
// NMatrix is part of SciRuby.
//
// NMatrix was originally inspired by and derived from NArray, by
// Masahiro Tanaka: http://narray.rubyforge.org
//
// == Copyright Information
//
// SciRuby is Copyright (c) 2010 - 2013, Ruby Science Foundation
// NMatrix is Copyright (c) 2013, Ruby Science Foundation
//
// Please see LICENSE.txt for additional copyright notices.
//
// == Contributing
//
// By contributing source code to SciRuby, you agree to be bound by
// our Contributor Agreement:
//
// * https://github.com/SciRuby/sciruby/wiki/Contributor-Agreement
//
// == row.h
//
// Iterator for traversing a matrix row by row. Includes an
// orthogonal iterator for visiting each stored entry in a row.
//

#ifndef YALE_ITERATORS_ROW_H
# define YALE_ITERATORS_ROW_H

namespace nm { namespace yale_storage {

template <typename D,
          typename RefType,
          typename YaleRef = typename std::conditional<
            std::is_const<RefType>::value,
            const nm::YaleStorage<D>,
            nm::YaleStorage<D>
          >::type>
class row_iterator_T {

  template <typename RowRef = typename std::conditional<
              std::is_const<RefType>::value,
              const row_iterator_T<D,RefType,YaleRef>,
              row_iterator_T<D,RefType,YaleRef>
            >::type>
  class row_stored_iterator_T {
    RowRef& r;
    size_t p_;
    size_t j_;


  };
protected:
  YaleRef& y;
  size_t i_;

public:
  row_iterator_T(YaleRef& obj, size_t ii = 0)
  : basic_iterator_T<D,RefType,YaleRef>(obj, ii)
  { }

  row_iterator_T<D,RefType,YaleRef>& operator++() {
    ++i_;
    return *this;
  }

  size_t shape() const {
    return y.shape(0);
  }

  size_t offset() const {
    return y.offset(0);
  }
};

} } // end of nm::yale_storage namespace

#endif // YALE_ITERATORS_ROW_H