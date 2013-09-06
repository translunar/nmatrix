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
// == yale_iterators.h
//
// Helpful iterators for Yale storage.
//

#ifndef YALE_ITERATORS_H
# define YALE_ITERATORS_H

#include <type_traits>
#include <typeinfo> // typeid()

namespace nm {

template <typename DType> class YaleStorage;

namespace yale_storage {

/*
 * Iterator base class (pure virtual).
 */
template <typename D,
          typename RefType,
          typename YaleRef = typename std::conditional<
            std::is_const<RefType>::value,
            const nm::YaleStorage<D>,
            nm::YaleStorage<D>
          >::type>
class basic_iterator_T {

protected:
  YaleRef& y;
  size_t i_;
  size_t p_;

public:
  size_t offset(size_t d) const { return y.offset(d); }
  size_t shape(size_t d) const { return y.shape(d); }
  size_t real_shape(size_t d) const { return y.real_shape(d); }

  template <typename T = typename std::conditional<std::is_const<RefType>::value, const size_t, size_t>::type>
  T& ija(size_t pp) const { return y.ija(pp); }

  template <typename T = typename std::enable_if<std::is_const<RefType>::value, const size_t>::type>
  T& ija(size_t pp) { return y.ija(pp); }

  virtual bool diag() const {
    return p_ < std::min(y.real_shape(0), y.real_shape(1));
  }
  virtual bool done_with_diag() const {
    return p_ == std::min(y.real_shape(0), y.real_shape(1));
  }
  virtual bool nondiag() const {
    return p_ > std::min(y.real_shape(0), y.real_shape(1));
  }

  basic_iterator_T(YaleRef& obj, size_t ii = 0, size_t pp = 0) : y(obj), i_(ii), p_(pp) { }

  virtual inline size_t i() const { return i_; }
  virtual size_t j() const = 0;

  virtual inline VALUE rb_i() const { return LONG2NUM(i()); }
  virtual inline VALUE rb_j() const { return LONG2NUM(j()); }

  virtual size_t real_i() const { return offset(0) + i(); }
  virtual size_t real_j() const { return offset(1) + j(); }
  virtual size_t p() const { return p_; }
  virtual bool real_ndnz_exists() const { return !y.real_row_empty(real_i()) && ija(p_) == real_j(); }

  virtual RefType& operator*() = 0;
  virtual RefType& operator*() const = 0;


  // Ruby VALUE de-reference
  virtual VALUE operator~() const {
    if (typeid(D) == typeid(RubyObject)) return (**this); // FIXME: return rval instead, faster;
    else return RubyObject(*(*this)).rval;
  }

  virtual bool operator==(const std::pair<size_t,size_t>& ij) {
    if (p() >= ija(real_shape(0))) return false;
    else return i() == ij.first && j() == ij.second;
  }

  virtual bool operator==(const basic_iterator_T<D,RefType,YaleRef>& rhs) const {
    return i() == rhs.i() && j() == rhs.j();
  }
  virtual bool operator!=(const basic_iterator_T<D,RefType,YaleRef>& rhs) const {
    return i() != rhs.i() || j() != rhs.j();
  }
};


/*
 * Iterate across the stored diagonal.
 */
template <typename D,
          typename RefType,
          typename YaleRef = typename std::conditional<
            std::is_const<RefType>::value,
            const nm::YaleStorage<D>,
            nm::YaleStorage<D>
          >::type>
class stored_diagonal_iterator_T : public basic_iterator_T<D,RefType,YaleRef> {
  using basic_iterator_T<D,RefType,YaleRef>::p_;
  using basic_iterator_T<D,RefType,YaleRef>::y;
  using basic_iterator_T<D,RefType,YaleRef>::offset;
  using basic_iterator_T<D,RefType,YaleRef>::shape;
public:
  stored_diagonal_iterator_T(YaleRef& obj, size_t d = 0)
  : basic_iterator_T<D,RefType,YaleRef>(obj,                // y
                   std::max(obj.offset(0), obj.offset(1)) + d - obj.offset(0), // i_
                   std::max(obj.offset(0), obj.offset(1)) + d) // p_
  {
//      std::cerr << "sdbegin: d=" << d << ", p_=" << p_ << ", i()=" << i() << ", j()=" << j() << std::endl;
    // p_ can range from max(y.offset(0), y.offset(1)) to min(y.real_shape(0), y.real_shape(1))
  }


  size_t d() const {
    return p_ - std::max(offset(0), offset(1));
  }

  stored_diagonal_iterator_T<D,RefType,YaleRef>& operator++() {
    ++p_;
    return *this;
  }

  stored_diagonal_iterator_T<D,RefType,YaleRef> operator++(int dummy) const {
    stored_diagonal_iterator_T<D,RefType,YaleRef> iter(*this);
    return ++iter;
  }

  // Indicates if we're at the end of the iteration.
  bool end() const {
    return p_ >= std::min( shape(0) + offset(0), shape(1) + offset(1) );
  }

  // i() and j() are how we know if we're past-the-end. i will be shape(0) and j will be 0.
  size_t i() const {
    return end() ? shape(0) : p_ - offset(0);
  }

  size_t j() const {
    return end() ? 0 : p_ - offset(1);
  }


  template <typename E, typename ERefType = typename std::conditional<std::is_const<RefType>::value, const E, E>::type>
  bool operator!=(const stored_diagonal_iterator_T<E,ERefType>& rhs) const { return d() != rhs.d(); }

  template <typename E, typename ERefType = typename std::conditional<std::is_const<RefType>::value, const E, E>::type>
  bool operator==(const stored_diagonal_iterator_T<E,ERefType>& rhs) const { return !(*this != rhs); }

  template <typename E, typename ERefType = typename std::conditional<std::is_const<RefType>::value, const E, E>::type>
  bool operator<(const stored_diagonal_iterator_T<E,ERefType>& rhs) const {  return d() < rhs.d(); }

  template <typename E, typename ERefType = typename std::conditional<std::is_const<RefType>::value, const E, E>::type>
  bool operator<=(const stored_diagonal_iterator_T<E,ERefType>& rhs) const {
    return d() <= rhs.d();
  }

  template <typename E, typename ERefType = typename std::conditional<std::is_const<RefType>::value, const E, E>::type>
  bool operator>(const stored_diagonal_iterator_T<E,ERefType>& rhs) const {
    return d() > rhs.d();
  }

  template <typename E, typename ERefType = typename std::conditional<std::is_const<RefType>::value, const E, E>::type>
  bool operator>=(const stored_diagonal_iterator_T<E,ERefType>& rhs) const {
    return d() >= rhs.d();
  }

  RefType& operator*() { return y.a(p_); }
  RefType& operator*() const { return y.a(p_); }

};



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
      p_ = ija(i_ + offset(0)); // update p to row beginning.
      std::cerr << "  new i=" << i_ << std::endl;
      std::cerr << " new p=" << p_ << std::endl;
    }

    return *this;
  }

  stored_nondiagonal_iterator_T<D,RefType,YaleRef> operator++(int dummy) const {
    stored_nondiagonal_iterator_T<D,RefType,YaleRef> iter(*this);
    return ++iter;
  }

  template <typename E, typename ERefType = typename std::conditional<std::is_const<RefType>::value, const E, E>::type>
  bool operator==(const stored_nondiagonal_iterator_T<E,ERefType>& rhs) const {
    if (i_ != rhs.i_) return false;
    if (i_ > shape(0) && rhs.i_ > rhs.shape(0)) return true;   // handles sndend()
    else return j() == rhs.j();
  }

  template <typename E, typename ERefType = typename std::conditional<std::is_const<RefType>::value, const E, E>::type>
  bool operator!=(const stored_nondiagonal_iterator_T<E,ERefType>& rhs) const {
    if (i_ > shape(0)) {
      if (rhs.i_ > rhs.shape(0)) return false;  // all sndend iterators are ==
      else                        return true;   // one is sndend and one isn't
    } else if (rhs.i_ > rhs.shape(0)) return true;

    return (i_ != rhs.i_ || j() != rhs.j());
  }

  template <typename E, typename ERefType = typename std::conditional<std::is_const<RefType>::value, const E, E>::type>
  bool operator<(const stored_diagonal_iterator_T<E,ERefType>& rhs) const {
    if (i_ < rhs.i()) return true;
    else if (i_ == rhs.i()) return j() < rhs.j();
    else return false;
  }

  template <typename E, typename ERefType = typename std::conditional<std::is_const<RefType>::value, const E, E>::type>
  bool operator>(const stored_diagonal_iterator_T<E,ERefType>& rhs) const {
    if (i_ > rhs.i()) return true;
    else if (i_ == rhs.i()) return j() > rhs.j();
    else return false;
  }


  virtual inline size_t j() const {
    return ija(p_) - offset(1);
  }


  virtual inline size_t p() const {
    return p_;
  }

  RefType& operator*() { return y.a(p_); }
  RefType& operator*() const { return y.a(p_); }
};


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

  stored_iterator_T(const stored_iterator_T<D,RefType,YaleRef>& rhs) {
    y = rhs.y;
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
    return *this == *(rhs->iter);
  }

  virtual bool operator!=(const stored_iterator_T<D,RefType,YaleRef>& rhs) const {
    return *this != *(rhs->iter);
  }


  // De-reference the iterator
  RefType& operator*() {  return &(**iter); }
  RefType& operator*() const {  return &(**iter); }

};

/*
 * Iterator for traversing matrix class as if it were dense (visits each entry in order).
 */
template <typename D,
          typename RefType,
          typename YaleRef = typename std::conditional<
            std::is_const<RefType>::value,
            const nm::YaleStorage<D>,
            nm::YaleStorage<D>
          >::type>
class iterator_T : public basic_iterator_T<D,RefType,YaleRef> {
  using basic_iterator_T<D,RefType,YaleRef>::i_;
  using basic_iterator_T<D,RefType,YaleRef>::p_;
  using basic_iterator_T<D,RefType,YaleRef>::y;
  using basic_iterator_T<D,RefType,YaleRef>::offset;
  using basic_iterator_T<D,RefType,YaleRef>::shape;
  using basic_iterator_T<D,RefType,YaleRef>::ija;

protected:
  size_t j_; // These are relative to the slice.

public:
  // Create an iterator. May select the row since this is O(1).
  iterator_T(YaleRef& obj, size_t ii = 0) : basic_iterator_T<D,RefType,YaleRef>(obj, ii, obj.ija(ii + obj.offset(0))), j_(0) { }

  // Prefix ++
  iterator_T<D,RefType,YaleRef>& operator++() {
    size_t prev_j = j_++;
    if (j_ >= shape(1)) {
      j_ = 0;
      ++i_;

      // Do a binary search to find the beginning of the slice
      p_ = offset(0) > 0 ? y.find_pos_for_insertion(i_,j_) : ija(i_);
    } else {
      // If the last j was actually stored in this row of the matrix, need to advance p.

      if (!y.real_row_empty(i_ + offset(0)) && ija(p_) <= prev_j + offset(1)) ++p_;  // this test is the same as real_ndnz_exists
    }

    return *this;
  }

  iterator_T<D,RefType,YaleRef> operator++(int dummy) const {
    iterator_T<D,RefType,YaleRef> iter(*this);
    return ++iter;
  }

  virtual bool operator!=(const iterator_T<D,RefType,YaleRef>& rhs) const {
    return (i_ != rhs.i_ || j_ != rhs.j_);
  }

  virtual bool operator==(const iterator_T<D,RefType,YaleRef>& rhs) const {
    return !(*this != rhs);
  }

  bool operator<(const iterator_T<D,RefType,YaleRef>& rhs) const {
    if (i_ > rhs.i_) return false;
    if (i_ < rhs.i_) return true;
    return j_ < rhs.j_;
  }

  bool operator>(const iterator_T<D,RefType,YaleRef>& rhs) const {
    if (i_ < rhs.i_) return false;
    if (i_ > rhs.i_) return true;
    return j_ > rhs.j_;
  }

  virtual bool diag() const { return i_ + offset(0) == j_ + offset(1); }

  // De-reference
  RefType& operator*() {
    if (diag())                                                                return y.a( i_ + offset(0) );
    else if (p_ >= ija(i_+offset(0)+1))                                        return y.const_default_obj();
    else if (!y.real_row_empty(i_ + offset(0)) && ija(p_) == j_ + offset(1))   return y.a( p_ );
    else                                                                       return y.const_default_obj();
  }

  RefType& operator*() const {
    if (diag())                                                                return y.a( i_ + offset(0) );
    else if (p_ >= ija(i_+offset(0)+1))                                        return y.const_default_obj();
    else if (!y.real_row_empty(i_ + offset(0)) && ija(p_) == j_ + offset(1))   return y.a( p_ );
    else                                                                       return y.const_default_obj();
  }

  virtual size_t j() const { return j_; }
};



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
protected:
  stored_diagonal_iterator_T<D,RefType,YaleRef>     d_iter;
  stored_nondiagonal_iterator_T<D,RefType,YaleRef> nd_iter;
  bool d; // which iterator is the currently valid one

public:
  ordered_iterator_T(YaleRef& obj, size_t ii = 0)
  : basic_iterator_T<D,RefType,YaleRef>(obj, ii),
    d_iter(obj, ii),
    nd_iter(obj, ii),
    d(nd_iter > d_iter)
  {
    if (ii == 0) {
      std::cerr << "d:  " << d_iter.i() << ", " << d_iter.j() << std::endl;
      std::cerr << "nd: " << nd_iter.i() << ", " << nd_iter.j() << std::endl;
      std::cerr << "dominant: " << (d ? "d" : "nd") << std::endl;
    }
  }

  virtual size_t j() const { return d ? d_iter.j() : nd_iter.j(); }
  virtual size_t i() const { return d ? d_iter.i() : nd_iter.i(); }
  virtual size_t real_j() const { return d ? d_iter.real_j() : nd_iter.real_j(); }
  virtual size_t real_i() const { return d ? d_iter.real_i() : nd_iter.real_i(); }

  ordered_iterator_T<D,RefType,YaleRef>& operator++() {
    // FIXME: This can be sped up by only checking when necessary for nd_iter > d_iter. I believe
    // it only needs to be done once per row, and maybe never depending upon slice shape. Right?
    if (d) {
      ++d_iter;
    } else {
      ++nd_iter;
    }
    std::cerr << "nd_iter i=" << nd_iter.i() << ", j=" << nd_iter.j() << '\t';
    std::cerr << " d_iter i=" <<  d_iter.i() << ", j=" <<  d_iter.j() << '\t';
    d = nd_iter > d_iter; // || nd_iter.i() >= shape(0);
    std::cerr << "new dominant: " << (d ? "d" : "nd") << std::endl;
    std::cerr << "i=" << i() << ", j=" << j() << std::endl;

    return *this;
  }

  ordered_iterator_T<D,RefType,YaleRef> operator++(int dummy) const {
    ordered_iterator_T<D,RefType,YaleRef> iter(*this);
    return ++iter;
  }

  template <typename E, typename ERefType = typename std::conditional<std::is_const<RefType>::value, const E, E>::type>
  bool operator==(const ordered_iterator_T<E,ERefType>& rhs) const {
    return d ? rhs == d_iter : rhs == nd_iter;
  }

  template <typename E, typename ERefType = typename std::conditional<std::is_const<RefType>::value, const E, E>::type>
  bool operator==(const stored_diagonal_iterator_T<E,ERefType>& rhs) const {
    return i() == rhs.i() && j() == rhs.j();
  }

  template <typename E, typename ERefType = typename std::conditional<std::is_const<RefType>::value, const E, E>::type>
  bool operator==(const stored_nondiagonal_iterator_T<E,ERefType>& rhs) const {
    return i() == rhs.i() && j() == rhs.j();
  }

  template <typename E, typename ERefType = typename std::conditional<std::is_const<RefType>::value, const E, E>::type>
  bool operator!=(const ordered_iterator_T<E,ERefType>& rhs) const {
    return d ? rhs != d_iter : rhs != nd_iter;
  }

  template <typename E, typename ERefType = typename std::conditional<std::is_const<RefType>::value, const E, E>::type>
  bool operator!=(const stored_diagonal_iterator_T<E,ERefType>& rhs) const {
    return i() != rhs.i() || j() != rhs.j();
  }

  template <typename E, typename ERefType = typename std::conditional<std::is_const<RefType>::value, const E, E>::type>
  bool operator!=(const stored_nondiagonal_iterator_T<E,ERefType>& rhs) const {
    return i() != rhs.i() || j() != rhs.j();
  }

  template <typename E, typename ERefType = typename std::conditional<std::is_const<RefType>::value, const E, E>::type>
  bool operator<(const stored_nondiagonal_iterator_T<E,ERefType>& rhs) const {
    if (i() > rhs.i()) return false;
    else if (i() == rhs.i()) return j() < rhs.j();
    else return i() < rhs.i();
  }

  template <typename E, typename ERefType = typename std::conditional<std::is_const<RefType>::value, const E, E>::type>
  bool operator<(const ordered_iterator_T<E,ERefType>& rhs) const {
    return d ? rhs > d_iter : rhs > nd_iter;
  }

  template <typename E, typename ERefType = typename std::conditional<std::is_const<RefType>::value, const E, E>::type>
  bool operator>(const stored_nondiagonal_iterator_T<E,ERefType>& rhs) const {
    if (i() < rhs.i()) return false;
    else if (i() == rhs.i()) return j() > rhs.j();
    else return i() > rhs.i();
  }

  template <typename E, typename ERefType = typename std::conditional<std::is_const<RefType>::value, const E, E>::type>
  bool operator>(const stored_diagonal_iterator_T<E,ERefType>& rhs) const {
    if (i() < rhs.i()) return false;
    else if (i() == rhs.i()) return j() > rhs.j();
    else return i() > rhs.i();
  }

  template <typename E, typename ERefType = typename std::conditional<std::is_const<RefType>::value, const E, E>::type>
  bool operator<(const stored_diagonal_iterator_T<E,ERefType>& rhs) const {
    if (i() > rhs.i()) return false;
    else if (i() == rhs.i()) return j() < rhs.j();
    else return i() < rhs.i();
  }

  template <typename E, typename ERefType = typename std::conditional<std::is_const<RefType>::value, const E, E>::type>
  bool operator>(const ordered_iterator_T<E,ERefType>& rhs) const {
    return d ? rhs < d_iter : rhs < nd_iter;
  }


  // De-reference the iterator
  RefType& operator*()       {   return d ? *d_iter : *nd_iter; }
  RefType& operator*() const {   return d ? *d_iter : *nd_iter; }

//    virtual const RefType operator*() const {
//      return d ? *d_iter : *nd_iter;
//    }
};


} // end of namespace yale_storage

} // namespace nm

#endif