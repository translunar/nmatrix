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
// == ordered.h
//
// Yale ordered iterator which only goes over stored elements (inc
// diagonal zeros)
//

#ifndef YALE_ITERATORS_ORDERED_H
# define YALE_ITERATORS_ORDERED_H

#include <type_traits>
#include <typeinfo>

namespace nm { namespace yale_storage {


/*
 * The trickiest of all the iterators for Yale. We only want to visit the stored indices, but we want to visit them
 * in matrix order.
 */
template <typename D,
          typename RefType,
          typename YaleRef = typename std::conditional<
            std::is_const<RefType>::value,
            const nm::YaleStorage<D>,
            nm::YaleStorage<D>
          >::type>
class ordered_iterator_T : public basic_iterator_T<D,RefType,YaleRef> {
  using basic_iterator_T<D,RefType,YaleRef>::y;
  using basic_iterator_T<D,RefType,YaleRef>::p_;
  using basic_iterator_T<D,RefType,YaleRef>::i_;
  using basic_iterator_T<D,RefType,YaleRef>::shape;
  using basic_iterator_T<D,RefType,YaleRef>::real_shape;
  using basic_iterator_T<D,RefType,YaleRef>::offset;
protected:
  stored_diagonal_iterator_T<D,RefType,YaleRef>     next_d;
  stored_nondiagonal_iterator_T<D,RefType,YaleRef>  next_nd;

public:
  ordered_iterator_T(YaleRef& obj, size_t ii = 0)
  : basic_iterator_T<D,RefType,YaleRef>(obj, ii),
    next_d(obj, ii),
    next_nd(obj, ii)
  {
    bool found = false;

    while (!found && i_ < shape(0)) {
      if (next_d.i() == i_) {
        if (next_nd.i() == i_) {
          if (next_d.j() < next_nd.j()) {
            p_ = next_d.p();
          } else {
            p_ = next_nd.p();
          }
        } else { // only next_d is in this row
          p_ = next_d.p();
        }
        found = true;
      } else if (next_nd.i() == i_) { // only nd is in this row
        p_ = next_nd.p();
        found = true;
      } else {
        ++i_;
      }
    }

    if (found) {
      if (p_ == next_nd.p())      ++next_nd;
      else if (p_ == next_d.p())  ++next_d;
    } else {
      i_ = shape(0); // put us at the end
      p_ = y.ija(i_ + offset(0));
    }
  }

  ordered_iterator_T<D,RefType,YaleRef>& operator=(const ordered_iterator_T<D,RefType,YaleRef>& rhs) {
    if (&y != &(rhs.y)) throw std::logic_error("can only be used on iterators with the same matrix");
    i_      = rhs.i_;
    p_      = rhs.p_;
    next_d  = rhs.next_d;
    next_nd = rhs.next_nd;
    return *this;
  }

  virtual size_t j() const {
    return real_j() - offset(1);
  }

  virtual size_t real_j() const {
    if (i_ == shape(0))       return offset(1);
    if (p_ <= real_shape(0))  return p_;
    else                      return y.ija(p_);
  }


  ordered_iterator_T<D,RefType,YaleRef>& operator++() {
    while (i_ < shape(0)) {
      if (next_d.i() == i_) {
        if (next_nd.i() == i_) {
          if (next_d.j() < next_nd.j()) {
            p_ = next_d.p();
            ++next_d;
            if (next_d.p() == p_) std::cerr << "++A" << std::endl;
          } else {
            p_ = next_nd.p();
            ++next_nd;
            if (next_nd.p() == p_) std::cerr << "++B" << std::endl;
          }
        } else { // only next_d is in this row
          p_ = next_d.p();
          ++next_d;
          if (next_d.p() == p_) std::cerr << "++C" << std::endl;
        }
        break; // found
      } else if (next_nd.i() == i_) { // only nd is in this row
        p_ = next_nd.p();
        ++next_nd;
        if (next_nd.p() == p_) std::cerr << "++D" << std::endl;
        break; // found
      } else {
        ++i_; // not found -- no break
        p_ = y.ija(i_ + offset(0)); // place at beginning of next row
        std::cerr << "++E" << std::endl;
      }
    }

    return *this;
  }


  ordered_iterator_T<D,RefType,YaleRef> operator++(int dummy) const {
    ordered_iterator_T<D,RefType,YaleRef> iter(*this);
    return ++iter;
  }

  template <typename E, typename ERefType = typename std::conditional<std::is_const<RefType>::value, const E, E>::type>
  bool operator==(const ordered_iterator_T<E,ERefType>& rhs) const {
    return this->dense_location() == rhs.dense_location();
  }

  template <typename E, typename ERefType = typename std::conditional<std::is_const<RefType>::value, const E, E>::type>
  bool operator==(const stored_nondiagonal_iterator_T<E,ERefType>& rhs) const {
    return this->dense_location() == rhs.dense_location();
  }

  template <typename E, typename ERefType = typename std::conditional<std::is_const<RefType>::value, const E, E>::type>
  bool operator!=(const ordered_iterator_T<E,ERefType>& rhs) const {
    return this->dense_location() != rhs.dense_location();
  }

  template <typename E, typename ERefType = typename std::conditional<std::is_const<RefType>::value, const E, E>::type>
  bool operator!=(const stored_nondiagonal_iterator_T<E,ERefType>& rhs) const {
    return this->dense_location() != rhs.dense_location();
  }

  template <typename E, typename ERefType = typename std::conditional<std::is_const<RefType>::value, const E, E>::type>
  bool operator<(const stored_nondiagonal_iterator_T<E,ERefType>& rhs) const {
    return this->dense_location() < rhs.dense_location();
  }

  template <typename E, typename ERefType = typename std::conditional<std::is_const<RefType>::value, const E, E>::type>
  bool operator<(const ordered_iterator_T<E,ERefType>& rhs) const {
    return this->dense_location() < rhs.dense_location();
  }

  template <typename E, typename ERefType = typename std::conditional<std::is_const<RefType>::value, const E, E>::type>
  bool operator>(const stored_nondiagonal_iterator_T<E,ERefType>& rhs) const {
    return this->dense_location() > rhs.dense_location();
  }

  template <typename E, typename ERefType = typename std::conditional<std::is_const<RefType>::value, const E, E>::type>
  bool operator>(const ordered_iterator_T<E,ERefType>& rhs) const {
    return this->dense_location() > rhs.dense_location();
  }


  // De-reference the iterator
  RefType& operator*()       {
    return y.a(p_);
  }
  RefType& operator*() const {
    return y.a(p_);
  }

//    virtual const RefType operator*() const {
//      return d ? *d_iter : *nd_iter;
//    }
};

} } // end of namespace nm::yale_storage

#endif // YALE_ITERATORS_ORDERED_H