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
// == row_stored.h
//
// Iterator for traversing a single stored row of a matrix (needed
// for row.h). FIXME: This is not as efficient as it could be; it uses
// two binary searches to find the beginning and end of each slice.
// The end search shouldn't be necessary, but I couldn't make it
// work without it, and eventually decided my dissertation should
// be a priority.
//

#ifndef YALE_ITERATORS_ROW_STORED_H
# define YALE_ITERATORS_ROW_STORED_H

namespace nm { namespace yale_storage {


/*
 * Iterator for visiting each stored element in a row, including diagonals.
 */
template <typename D,
          typename RefType,
          typename YaleRef = typename std::conditional<
            std::is_const<RefType>::value,
            const nm::YaleStorage<D>,
            nm::YaleStorage<D>
          >::type,
          typename RowRef = typename std::conditional<
            std::is_const<RefType>::value,
            const row_iterator_T<D,RefType,YaleRef>,
            row_iterator_T<D,RefType,YaleRef>
          >::type>
class row_stored_iterator_T : public row_stored_nd_iterator_T<D,RefType,YaleRef,RowRef> {
protected:
  using row_stored_nd_iterator_T<D,RefType,YaleRef,RowRef>::r;
  using row_stored_nd_iterator_T<D,RefType,YaleRef,RowRef>::p_;
  bool d_visited, d;

public:

  row_stored_iterator_T(RowRef& row, size_t pp)
  : row_stored_nd_iterator_T<D,RefType,YaleRef,RowRef>(row, pp),
    d_visited(!row.has_diag()), // if the row has no diagonal, just marked it as visited.
    d(r.is_diag_first())        // do we start at the diagonal?
  {
    std::cerr << "row_stored_iterator::init d_visited = " << std::boolalpha << d_visited << ", d = " << d << std::endl;
    std::cerr << "   end() ? " << end() << "\tp=" << p_ << ", p_last=" << r.p_last << std::endl;
  }

  virtual bool end() const {
    return !d && p_ > r.p_last;
  }

  row_stored_iterator_T<D,RefType,YaleRef,RowRef>& operator++() {
    if (end()) throw std::out_of_range("cannot increment row stored iterator past end of stored row");
    if (d) {
      std::cerr << "row_stored_iterator: operator++: from diag" << std::endl;
      d_visited = true;
      d         = false;
    } else {
      ++p_;
      std::cerr << "row_stored_iterator: operator++: from p = " << p_;
      // Are we at a diagonal?
      if (!d_visited && p_ > r.p_diag()) {
        std::cerr << " to diag";
        d = true;
      }
      std::cerr << "\tend() ? " << end() << std::endl;
    }

    return *this;
  }

  row_stored_iterator_T<D,RefType,YaleRef,RowRef> operator++(int dummy) const {
    row_stored_iterator_T<D,RefType,YaleRef,RowRef> r(*this);
    return ++r;
  }

  size_t j() const {
    if (end()) throw std::out_of_range("cannot dereference an end pointer");
    return (d ? r.p_diag() : r.ija(p_)) - r.offset(1);
  }

  // Need to declare all row_stored_iterator_T friends of each other.
  template <typename E, typename ERefType, typename EYaleRef, typename ERowRef> friend class row_stored_iterator_T;

/*  virtual bool operator==(const row_stored_iterator_T<D,RefType,YaleRef,RowRef>& rhs) const {
    if (r != rhs.r)     return false;
    if (end())          return rhs.end();
    else if (rhs.end()) return false;
    return j() == rhs.j();
  }

  virtual bool operator!=(const row_stored_iterator_T<D,RefType,YaleRef,RowRef>& rhs) const {
    if (r.i() != rhs.r.i()) return true;
    if (end())              return !rhs.end();
    else if (rhs.end())     return true;
    return j() != rhs.j();
  }

  template <typename E, typename ERefType = typename std::conditional<std::is_const<RefType>::value, const E, E>::type>
  bool operator<(const row_stored_iterator_T<E,ERefType>& rhs) const {
    if (r < rhs.r)      return true;
    if (r > rhs.r)      return false;

    // r == rhs.r
    if (end())        return false;
    if (rhs.end())    return true;
    return j() < rhs.j();
  }*/

  // De-reference the iterator
  RefType& operator*()       {
    return d ? r.a(r.p_diag()) : r.a(p_);
  }

  RefType& operator*() const {
    return d ? r.a(r.p_diag()) : r.a(p_);
  }

  // Ruby VALUE de-reference
  VALUE operator~() const {
    if (typeid(D) == typeid(RubyObject)) return (**this); // FIXME: return rval instead, faster;
    else return RubyObject(*(*this)).rval;
  }

};

}} // end of namespace nm::yale_storage

#endif // YALE_ITERATORS_ROW_STORED_H