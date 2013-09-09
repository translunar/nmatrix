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
// == stored_nondiagonal.h
//
// Yale storage nondiagonal-storage iterator
//

#ifndef YALE_ITERATORS_STORED_NONDIAGONAL_H
# define YALE_ITERATORS_STORED_NONDIAGONAL_H

#include <type_traits>
#include <typeinfo>

namespace nm { namespace yale_storage {

/*
 * Iterate across the stored non-diagonals.
 */
template <typename D,
          typename RefType,
          typename YaleRef = typename std::conditional<
            std::is_const<RefType>::value,
            const nm::YaleStorage<D>,
            nm::YaleStorage<D>
          >::type>
class stored_nondiagonal_iterator_T : public basic_iterator_T<D,RefType,YaleRef> {
  using basic_iterator_T<D,RefType,YaleRef>::i_;
  using basic_iterator_T<D,RefType,YaleRef>::p_;
  using basic_iterator_T<D,RefType,YaleRef>::y;
  using basic_iterator_T<D,RefType,YaleRef>::offset;
  using basic_iterator_T<D,RefType,YaleRef>::shape;
  using basic_iterator_T<D,RefType,YaleRef>::ija;

protected:

  virtual bool is_valid_nonempty_real_row() const {
    return i_ < shape(0) && ija(i_ + offset(0)) < ija(i_ + offset(0)+1);
  }

  virtual bool is_valid_empty_real_row() const {
    return i_ < shape(0) && ija(i_ + offset(0)) == ija(i_ + offset(0)+1);
  }

  // Key loop for forward row iteration in the non-diagonal portion of the matrix. Called during construction and by
  // the ++ operators.
  void advance_to_first_valid_entry() {
    // p is already set to the initial position for row i_ (may be outside the slice)
    // try to put it inside the slice. If we can't, we'll want to loop and increase i_
    if (is_valid_nonempty_real_row() && ija(p_) < offset(1)) {
      p_ = y.find_pos_for_insertion(i_, 0);
      std::cerr << "A: " << p_ << std::endl;
    }

    // If the row is entirely empty, we know we need to go to the next row.
    // Even if the row isn't empty, an invalid p needs to be advanced.
    while (is_valid_empty_real_row() || ija(p_) < offset(1) || ija(p_) - offset(1) >= shape(1)) {
      ++i_;

      // If we're in a valid row, we can try looking for p again.
      if (is_valid_nonempty_real_row()) {
        p_ = y.find_pos_for_insertion(i_, 0);
        std::cerr << "B1: " << p_ << std::endl;
      } else {
        p_ = ija(i_ + offset(0) + 1); // beginning of next valid row
        std::cerr << "B2: " << p_ << std::endl;
      }

      if (p_ >= y.size()) { // abort -- empty matrix
        p_ = y.size();
        i_ = y.shape(0);
        break;
      }
    }

    std::cerr << "advance: i = " << i_ << ", p_ = " << p_ << std::endl;
  }


public:
  stored_nondiagonal_iterator_T(YaleRef& obj, size_t ii = 0)
  : basic_iterator_T<D,RefType,YaleRef>(obj, ii, obj.ija(ii + obj.offset(0)))
  {
    if (ii < shape(0)) {
      advance_to_first_valid_entry();

      std::cerr << "initial: p=" << p_ << ", i=" << i_ << std::endl;
    }
  }


  stored_nondiagonal_iterator_T(YaleRef& obj, size_t ii, size_t pp) : basic_iterator_T<D,RefType,YaleRef>(obj, ii, pp) { }


  // Pre-condition: p >= real_row_begin
  bool find_valid_p_for_row() {
    size_t real_row_begin = ija(i_ + offset(0)),
           real_row_end   = ija(i_ + offset(0) + 1);
    // if (p_ < real_row_begin) return false; // This is a pre-condition
    if (real_row_begin == real_row_end) return false;
    if (p_ >= real_row_end) return false;

    if (ija(p_) < offset(1)) {
      std::cerr << "p=" << p_ << " is pre-slice" << std::endl;
      size_t next_p = y.find_pos_for_insertion(i_, 0);
      if (p_ == next_p) {
        std::cerr << "    find_pos_for_insertion returned current p (returning false)" << std::endl;
        return false;
      }
      else p_ = next_p;

      std::cerr << "    p <- next_p. Now: " << p_ << std::endl;

      if (p_ >= real_row_end) {
        std::cerr << "    p is past the end of the row (returning false)" << std::endl;
        return false;
      }
      if (ija(p_) < offset(1)) {
        std::cerr << "    p is still pre-slice (returning false)" << std::endl;
        return false;
      }
      if (j() < shape(1)) {
        std::cerr << "    j()=" << j() << " is positive and less than shape (returning TRUE)" << std::endl;
        return true; // binary search worked!
      }
      std::cerr << "    j()=" << j() << " is post-slice (returning false)" << std::endl;
      return false; // out of range. not in row.
    } else if (j() < shape(1)) return true; // within the necessary range.
    else
      return false; // out of range. not in row.
  }


  stored_nondiagonal_iterator_T<D,RefType,YaleRef>& operator++() {
    if (!end()) {
      size_t real_row_begin = ija(i_ + offset(0)),
             real_row_end   = ija(i_ + offset(0) + 1);
      ++p_;

      std::cerr << "new p=" << p_ << std::endl;

      if (p_ >= y.size()) {
        i_ = shape(0);
        return *this;
      }

      while (i_ < shape(0) && !find_valid_p_for_row()) { // skip forward
        ++i_;

        //if (i_ < shape(0)) {
        p_ = ija(i_ + offset(0)); // update p to row beginning.
        std::cerr << " new p=" << p_ << std::endl;
        //}
        std::cerr << "  new i=" << i_ << std::endl;
      }
    }

    return *this;
  }

  stored_nondiagonal_iterator_T<D,RefType,YaleRef> operator++(int dummy) const {
    stored_nondiagonal_iterator_T<D,RefType,YaleRef> iter(*this);
    return ++iter;
  }

  template <typename E, typename ERefType = typename std::conditional<std::is_const<RefType>::value, const E, E>::type>
  inline bool has_same_y(const stored_nondiagonal_iterator_T<E,ERefType>& rhs) const {
    return &y == &(rhs.y);
  }

  inline bool end() const {
   /* std::cerr << "end: p_=" << p_ << ", capacity=" << y.capacity() << ", size=" << y.size() << std::endl;
    if (p_ == 22 && y.size() == 22 && y.capacity() == 24)
      throw std::logic_error("loop");*/
    return p_ >= y.size();
  }

  // FIXME: need a better way to do j() comparison between iterators with different y's. If p is out of range, it will have memory problems.
  template <typename E, typename ERefType = typename std::conditional<std::is_const<RefType>::value, const E, E>::type>
  bool operator==(const stored_nondiagonal_iterator_T<E,ERefType>& rhs) const {
    if (typeid(E) == typeid(D) && has_same_y<E,ERefType>(rhs)) return p_ == rhs.p_;
    if (i_ > shape(0) && rhs.i_ > rhs.shape(0)) return true;   // handles sndend()
    if (i_ != rhs.i_) return false;
    else return j() == rhs.j();
  }

  template <typename E, typename ERefType = typename std::conditional<std::is_const<RefType>::value, const E, E>::type>
  bool operator!=(const stored_nondiagonal_iterator_T<E,ERefType>& rhs) const {
    if (typeid(E) == typeid(D) && has_same_y<E,ERefType>(rhs)) return p_ != rhs.p_;
    if (i_ > shape(0)) {
      if (rhs.i_ > rhs.shape(0)) return false;  // all sndend iterators are ==
      else                       return true;   // one is sndend and one isn't
    } else if (rhs.i_ > rhs.shape(0)) return true; // one is sndend and one isn't

    return (i_ != rhs.i_ || j() != rhs.j());
  }

  template <typename E, typename ERefType = typename std::conditional<std::is_const<RefType>::value, const E, E>::type>
  bool operator<(const stored_diagonal_iterator_T<E,ERefType>& rhs) const {
    if (i_ < rhs.i()) return true;
    else if (i_ == rhs.i()) {
      if (i_ == shape(0)) return false;  // if both are at end, definitely not <
      else                return j() < rhs.j();
    }
    else return false;
  }

  template <typename E, typename ERefType = typename std::conditional<std::is_const<RefType>::value, const E, E>::type>
  bool operator>(const stored_diagonal_iterator_T<E,ERefType>& rhs) const {
    if (i_ > rhs.i()) return true;
    else if (i_ == rhs.i()) {
      if (i_ == shape(0)) return false; // both are at end, definitely not >
      else                return j() > rhs.j();
    }
    else return false;
  }


  virtual inline size_t j() const {
    if (end()) return 0; // end
    return ija(p_) - offset(1);
  }


  virtual inline size_t p() const {
    return p_;
  }

  RefType& operator*() { return y.a(p_); }
  RefType& operator*() const { return y.a(p_); }
};


} } // end of namespace nm::yale_storage

#endif // YALE_ITERATORS_STORED_NONDIAGONAL_H