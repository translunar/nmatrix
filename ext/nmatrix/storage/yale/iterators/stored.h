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
// == stored.h
//
// Yale storage iterator -- visits all diagonal stored elements,
// which may or may not be zero, and all non-diagonal stored
// elements.
//

#ifndef YALE_ITERATORS_STORED_H
# define YALE_ITERATORS_STORED_H

#include <type_traits>
#include <typeinfo>

namespace nm { namespace yale_storage {

/*
 * Iterate across a matrix in storage order.
 *
 * Note: It is not recommended that this iterator be compared to an iterator from another matrix, as storage order
 * determines iteration -- and storage order will differ for slices which are positioned differently relative to the
 * original matrix.
 */
template <typename D,
          typename RefType,
          typename YaleRef = typename std::conditional<
            std::is_const<RefType>::value,
            const nm::YaleStorage<D>,
            nm::YaleStorage<D>
          >::type>
class stored_iterator_T : public basic_iterator_T<D,RefType,YaleRef> {
  using basic_iterator_T<D,RefType,YaleRef>::i_;
  using basic_iterator_T<D,RefType,YaleRef>::p_;
  using basic_iterator_T<D,RefType,YaleRef>::y;
protected:
  virtual bool diag() const { return iter->diag(); }
  virtual bool nondiag() const { return iter->nondiag(); }

  basic_iterator_T<D,RefType,YaleRef>* iter;

public:
  stored_iterator_T(YaleRef& obj, size_t ii = 0) : basic_iterator_T<D,RefType,YaleRef>(obj, 0, 0) {
    if (ii < obj.shape(0)) {
      iter = new stored_diagonal_iterator_T<D,RefType,YaleRef>(obj, ii);

      // if we're past the diagonal already, delete it and create a nondiagonal iterator
      if (!diag()) {
        delete iter;
        iter = new stored_nondiagonal_iterator_T<D,RefType,YaleRef>(obj, ii);
      }
    } else {
      iter = new stored_nondiagonal_iterator_T<D,RefType,YaleRef>(obj, ii);
    }
  }

  ~stored_iterator_T() {
    delete iter;
  }

  stored_iterator_T(const stored_iterator_T<D,RefType,YaleRef>& rhs) : basic_iterator_T<D,RefType,YaleRef>(rhs) {
    if (rhs.diag()) {
      iter = new stored_diagonal_iterator_T<D,RefType,YaleRef>(*(stored_diagonal_iterator_T<D,RefType,YaleRef>*)iter);
    } else {
      iter = new stored_nondiagonal_iterator_T<D,RefType,YaleRef>(*(stored_nondiagonal_iterator_T<D,RefType,YaleRef>*)iter);
    }
  }

  virtual size_t j() const { return iter->j(); }
  virtual size_t i() const { return iter->i(); }
  virtual size_t p() const { return iter->p(); }
  virtual size_t real_j() const { return iter->real_j(); }
  virtual size_t real_i() const { return iter->real_i(); }

  stored_iterator_T<D,RefType,YaleRef>& operator++() {
    if (diag()) {
      ++(*iter);
      if (!iter.diag()) {
        delete iter;
        iter = new stored_nondiagonal_iterator_T<D,RefType,YaleRef>(y);
      }
    } else {
      ++(*iter);
    }
    // Need to keep these up to date for easier copy construction.
    return *this;
  }

  stored_iterator_T<D,RefType,YaleRef> operator++(int dummy) const {
    stored_iterator_T<D,RefType,YaleRef> iter(*this);
    return ++iter;
  }

  virtual bool operator==(const stored_iterator_T<D,RefType,YaleRef>& rhs) const {
    return *this == *(rhs.iter);
  }

  virtual bool operator!=(const stored_iterator_T<D,RefType,YaleRef>& rhs) const {
    return *this != *(rhs.iter);
  }


  // De-reference the iterator
  RefType& operator*() {  return &(**iter); }
  RefType& operator*() const {  return &(**iter); }

};


} } // end of namespace nm::yale_storage

#endif // YALE_ITERATORS_STORED_H