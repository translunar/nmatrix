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
// This one cannot be de-referenced; you have to de-reference
// the column.

#ifndef YALE_ITERATORS_ROW_H
# define YALE_ITERATORS_ROW_H

#include <stdexcept>

namespace nm { namespace yale_storage {

template <typename D,
          typename RefType,
          typename YaleRef = typename std::conditional<
            std::is_const<RefType>::value,
            const nm::YaleStorage<D>,
            nm::YaleStorage<D>
          >::type>
class row_iterator_T {

protected:
  YaleRef& y;
  size_t i_;
  size_t p_first, p_last; // first and last IJA positions in the row
public:
/*  typedef row_stored_iterator_T<D,RefType,YaleRef>                  row_stored_iterator;
  typedef row_stored_nd_iterator_T<D,RefType,YaleRef>               row_stored_nd_iterator;
  typedef row_stored_iterator_T<D,const RefType,const YaleRef>      const_row_stored_iterator;
  typedef row_stored_nd_iterator_T<D,const RefType,const YaleRef>   const_row_stored_nd_iterator;*/

  template <typename E, typename ERefType, typename EYaleRef> friend class row_iterator_T;
  friend class row_stored_iterator_T<D,RefType,YaleRef>;
  friend class row_stored_nd_iterator_T<D,RefType,YaleRef>;//row_stored_iterator;
  //friend row_stored_nd_iterator;

  inline size_t ija(size_t pp) const { return y.ija(pp); }
  inline RefType& a(size_t p) const  { return y.a_p()[p]; }
  inline RefType& a(size_t p)        { return y.a_p()[p]; }

  void update_positions() {
    if (i_ < y.shape(0)) {
      p_first = p_real_first();
      p_last  = p_real_last();
      if (!nd_empty()) {
        // try to find new p_first
        p_first = y.real_find_left_boundary_pos(p_first, p_last, y.offset(1));
        if (!nd_empty()) {
          // also try to find new p_last
          p_last = y.real_find_left_boundary_pos(p_first, p_last, y.offset(1) + y.shape(1) - 1);
          if (y.ija(p_last) - y.offset(1) >= shape(1)) --p_last; // searched too far.
        }
      }
    } else { // invalid row -- this is an end iterator.
      p_first = y.ija(y.real_shape(0));
      p_last  = y.ija(y.real_shape(0))-1; // mark as empty
    }
  }

  row_iterator_T(YaleRef& obj, size_t ii = 0)
  : y(obj), i_(ii)
  {
    update_positions();
  }

  template <typename E, typename ERefType = typename std::conditional<std::is_const<RefType>::value, const E, E>::type>
  bool operator!=(const row_iterator_T<E,ERefType>& rhs) const {
    return i_ != rhs.i_;
  }

  template <typename E, typename ERefType = typename std::conditional<std::is_const<RefType>::value, const E, E>::type>
  bool operator==(const row_iterator_T<E,ERefType>& rhs) const {
    return i_ == rhs.i_;
  }

  template <typename E, typename ERefType = typename std::conditional<std::is_const<RefType>::value, const E, E>::type>
  bool operator<(const row_iterator_T<E,ERefType>& rhs) const {
    return i_ < rhs.i_;
  }

  template <typename E, typename ERefType = typename std::conditional<std::is_const<RefType>::value, const E, E>::type>
  bool operator>(const row_iterator_T<E,ERefType>& rhs) const {
    return i_ > rhs.i_;
  }

  row_iterator_T<D,RefType,YaleRef>& operator++() {
    if (is_end()) throw std::out_of_range("attempted to iterate past end of slice (vertically)");
    ++i_;
    update_positions();
    return *this;
  }

  row_iterator_T<D,RefType,YaleRef> operator++(int dummy) const {
    row_iterator_T<D,RefType,YaleRef> next(*this);
    return ++next;
  }

  bool is_end() const {
    return i_ == y.shape(0) && p_first == y.ija(y.real_shape(0));
  }

  size_t real_i() const {
    return i_ + y.offset(0);
  }

  size_t i() const {
    return i_;
  }

  // last element of the real row
  size_t p_real_last() const {
    return y.ija(real_i()+1)-1;
  }

  // first element of the real row
  size_t p_real_first() const {
    return y.ija(real_i());
  }

  // Is the real row of the original matrix totally empty of NDs?
  bool real_nd_empty() const {
    return p_real_last() < p_real_first();
  }

  bool nd_empty() const {
    return p_last < p_first;
  }

  // slice j coord of the diag.
  size_t diag_j() const {
    if (!has_diag())
      throw std::out_of_range("don't call diag_j unless you've checked for one");
    return real_i() - y.offset(1);
  }

  // return the actual position of the diagonal element for this real row, regardless of whether
  // it's in range or not.
  size_t p_diag() const {
    return real_i();
  }

  // Checks to see if there is a diagonal within the slice
  bool has_diag() const {
    // real position of diag is real_i == real_j. Is it in range?
    return (p_diag() >= y.offset(1) && p_diag() - y.offset(1) < y.shape(1));
  }

  // Checks to see if the diagonal is the first entry in the slice.
  bool is_diag_first() const {
    if (!has_diag()) return false;
    if (nd_empty())  return true;
    return diag_j() < y.ija(p_first) - y.offset(1);
  }

  // Checks to see if the diagonal is the last entry in the slice.
  bool is_diag_last() const {
    if (!has_diag()) return false;
    if (nd_empty())  return true;
    return diag_j() > y.ija(p_last);
  }

  // Is the row of the slice totally empty of NDs and Ds?
  // We can only determine that it's empty of Ds if the diagonal
  // is not a part of the sliced portion of the row.
  bool empty() const {
    return nd_empty() && has_diag();
  }


  size_t shape(size_t pp) const {
    return y.shape(pp);
  }

  size_t offset(size_t pp) const {
    return y.offset(pp);
  }

  inline VALUE rb_i() const { return LONG2NUM(i()); }


  row_stored_iterator_T<D,RefType,YaleRef> begin() {  return row_stored_iterator_T<D,RefType,YaleRef>(*this, p_first);  }
  row_stored_nd_iterator_T<D,RefType,YaleRef> ndbegin() {  return row_stored_nd_iterator_T<D,RefType,YaleRef>(*this, p_first);  }
  row_stored_iterator_T<D,RefType,YaleRef> end() { return row_stored_iterator_T<D,RefType,YaleRef>(*this, p_last+1, true); }
  row_stored_nd_iterator_T<D,RefType,YaleRef> ndend() {  return row_stored_nd_iterator_T<D,RefType,YaleRef>(*this, p_last+1); }

  row_stored_iterator_T<D,RefType,YaleRef> begin() const {  return row_stored_iterator_T<D,RefType,YaleRef>(*this, p_first);  }
  row_stored_nd_iterator_T<D,RefType,YaleRef> ndbegin() const {  return row_stored_nd_iterator_T<D,RefType,YaleRef>(*this, p_first);  }
  row_stored_iterator_T<D,RefType,YaleRef> end() const { return row_stored_iterator_T<D,RefType,YaleRef>(*this, p_last+1, true); }
  row_stored_nd_iterator_T<D,RefType,YaleRef> ndend() const {  return row_stored_nd_iterator_T<D,RefType,YaleRef>(*this, p_last+1); }


  row_stored_nd_iterator_T<D,RefType,YaleRef> lower_bound(const size_t& j) const {
    row_stored_nd_iterator_T<D,RefType,YaleRef>(*this, y.real_find_left_boundary_pos(p_first, p_last, y.offset(1)));
  }

};

} } // end of nm::yale_storage namespace

#endif // YALE_ITERATORS_ROW_H