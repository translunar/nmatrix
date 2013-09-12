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


  /*
   * Update the row positions -- use to ensure a row stays valid after an insert operation. Also
   * used to initialize a row iterator at a different row index.
   */
  void update() {
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

  /*
   * Indicate to the row iterator that p_first and p_last have moved by some amount. Only
   * defined for row_iterator, not const_row_iterator. This is a lightweight form of update().
   */
  //template <typename = typename std::enable_if<!std::is_const<RefType>::value>::type>
  void shift(int amount) {
    p_first += amount;
    p_last  += amount;
  }


  /*
   * Enlarge the row by amount by moving p_last over. This is a lightweight form of update().
   */
  //template <typename = typename std::enable_if<!std::is_const<RefType>::value>::type>
  void adjust_length(int amount) {
    p_last  += amount;
  }

public:
/*  typedef row_stored_iterator_T<D,RefType,YaleRef>                  row_stored_iterator;
  typedef row_stored_nd_iterator_T<D,RefType,YaleRef>               row_stored_nd_iterator;
  typedef row_stored_iterator_T<D,const RefType,const YaleRef>      const_row_stored_iterator;
  typedef row_stored_nd_iterator_T<D,const RefType,const YaleRef>   const_row_stored_nd_iterator;*/
  typedef row_stored_iterator_T<D,RefType,YaleRef, row_iterator_T<D,RefType,YaleRef> > row_stored_iterator;
  typedef row_stored_nd_iterator_T<D,RefType,YaleRef, row_iterator_T<D,RefType,YaleRef> > row_stored_nd_iterator;
  template <typename E, typename ERefType, typename EYaleRef> friend class row_iterator_T;
  friend class row_stored_iterator_T<D,RefType,YaleRef, row_iterator_T<D,RefType,YaleRef> >;
  friend class row_stored_nd_iterator_T<D,RefType,YaleRef, row_iterator_T<D,RefType,YaleRef> >;//row_stored_iterator;
  friend class row_stored_iterator_T<D,RefType,YaleRef, const row_iterator_T<D,RefType,YaleRef> >;
  friend class row_stored_nd_iterator_T<D,RefType,YaleRef, const row_iterator_T<D,RefType,YaleRef> >;//row_stored_iterator;
  friend class nm::YaleStorage<D>;

  //friend row_stored_nd_iterator;

  inline size_t ija(size_t pp) const { return y.ija(pp); }
  inline size_t& ija(size_t pp)      { return y.ija(pp); }
  inline RefType& a(size_t p) const  { return y.a_p()[p]; }
  inline RefType& a(size_t p)        { return y.a_p()[p]; }



  row_iterator_T(YaleRef& obj, size_t ii = 0)
  : y(obj), i_(ii)
  {
    update();
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
    update();
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

  row_stored_nd_iterator_T<D,RefType,YaleRef> ndfind(size_t j) {
    if (j == 0) return ndbegin();
    size_t p = y.real_find_left_boundary_pos(p_first, p_last, j + y.offset(1));
    return row_stored_nd_iterator_T<D,RefType,YaleRef>(*this, p);
  }

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

  /*
   * Remove an entry from an already found non-diagonal position. Adjust this row appropriately so we can continue to
   * use it.
   */
  //template <typename = typename std::enable_if<!std::is_const<RefType>::value>::type>
  row_stored_nd_iterator erase(row_stored_nd_iterator position) {
    size_t sz = y.size();
    if (y.capacity() / nm::yale_storage::GROWTH_CONSTANT <= sz - 1) {
      y.update_resize_move(position, i() + offset(0), -1);
    } else {
      y.move_left(position, 1);
    }
    adjust_length(-1);
    return row_stored_nd_iterator(*this, position.p()-1);
  }

  /*
   * Remove an entry from the matrix at the already-located position. If diagonal, just sets to default; otherwise,
   * actually removes the entry.
   */
  //template <typename = typename std::enable_if<!std::is_const<RefType>::value>::type>
  row_stored_nd_iterator erase(const row_stored_iterator& jt) {
    if (jt.diag()) {
      *jt = y.const_default_obj(); // diagonal is the easy case -- no movement.
      return row_stored_nd_iterator(*this, jt.p());
    } else {
      return erase(row_stored_nd_iterator(*this, jt.p()));
    }
  }

  //template <typename = typename std::enable_if<!std::is_const<RefType>::value>::type>
  template <typename T = typename std::conditional<std::is_const<RefType>::value,void,row_stored_nd_iterator>::type>
  row_stored_nd_iterator insert(row_stored_iterator position, size_t jj, const D& val) {
    if (position.diag()) {
      *position = val;  // simply replace existing, regardless of whether it's 0 or not
      ++position;
      return row_stored_nd_iterator(*this, position.p());
    } else {
      row_stored_nd_iterator jt(*this, position.p());
      return insert(jt, jj, val);
    }
  }

  /*
   * Insert an element in column j, using position's p() as the location to insert the new column. i and j will be the
   * coordinates. This also does a replace if column j is already present.
   *
   * Returns true if a new entry was added and false if an entry was replaced.
   *
   * Pre-conditions:
   *   - position.p() must be between ija(real_i) and ija(real_i+1), inclusive, where real_i = i + offset(0)
   *   - real_i and real_j must not be equal
   */
  //template <typename = typename std::enable_if<!std::is_const<RefType>::value>::type>
  row_stored_nd_iterator insert(row_stored_nd_iterator position, size_t jj, const D& val) {
    size_t sz = y.size();
    if (!position.end() && position.j() == jj) {
      std::cerr << "insert: *position = val at " << i_ << "," << jj << "\tp=" << position.p() << std::endl;
      *position = val;      // replace existing
    } else {
      if (sz + 1 > y.capacity()) {
        std::cerr << "insert: update_resize_move " << i_ << "," << jj << "\tp=" << position.p() << std::endl;
        y.update_resize_move(position, real_i(), 1);
      } else {
        std::cerr << "insert: move_right at " << i_ << "," << jj << "\tp=" << position.p() << std::endl;
        y.move_right(position, 1);
        y.update_real_row_sizes_from(real_i(), 1);
      }
      ija(position.p()) = jj + y.offset(1);    // set column ID
      a(position.p())   = val;
      adjust_length(1);
    }

    return position++;
  }

  //template <typename = typename std::enable_if<!std::is_const<RefType>::value>::type>
  row_stored_nd_iterator insert(size_t j, const D& val) {
    return insert(ndfind(j), j, val);
  }


  /*
   * Determines a plan for inserting a single row. Returns an integer giving the amount of the row change.
   */
  int single_row_insertion_plan(row_stored_nd_iterator position, size_t jj, size_t length, D const* v, size_t v_size, const size_t& v_offset) {
    int nd_change;
    size_t m = v_offset;
    for (size_t jc = jj; jc < jj + length; ++jc, ++m) {
      if (m >= v_size) m %= v_size; // reset v position.

      if (jc + y.offset(1) != real_i()) { // diagonal    -- no nd_change here
        if (position.j() != jc) { // not present -- do we need to add it?
          if (v[m] != y.const_default_obj()) nd_change++;
        } else {  // position.j() == jc
          if (v[m] == y.const_default_obj()) nd_change--;
          ++position; // move iterator forward.
        }
      }

    }
    return nd_change;
  }

  /*
   * Determine a plan for inserting a single row -- finds the position first. Returns the position and
   * the change amount. Don't use this one if you can help it because it requires a binary search of
   * the row.
   */
  std::pair<int,size_t> single_row_insertion_plan(size_t jj, size_t length, D const* v, size_t v_size, const size_t& v_offset) {
    std::pair<int,size_t> result;
    row_stored_nd_iterator pos = ndfind(jj);
    result.first = single_row_insertion_plan(pos, jj, length, v, v_size, v_offset);
    result.second = pos.p();
    return result;
  }

  /*
   * Insert elements into a single row. Returns an iterator to the end of the insertion range.
   */
  row_stored_nd_iterator insert(row_stored_nd_iterator position, size_t jj, size_t length, D const* v, size_t v_size, size_t& v_offset) {
    int nd_change = single_row_insertion_plan(position, jj, length, v, v_size);

    // First record the position, just in case our iterator becomes invalid.
    size_t pp = position.p();

    // Resize the array as necessary, or move entries after the insertion point to make room.
    size_t sz = y.size();
    if (sz + nd_change > y.capacity()) y.update_resize_move(position, real_i(), nd_change);
    if (nd_change < 0)                 y.move_left(position, -nd_change);
    else if (nd_change > 0)            y.move_right(position, nd_change);
 // else no change!

    for (size_t jc = jj; jc < jj + length; ++jc, ++v_offset, ++pp) {
      if (v_offset >= v_size) v_offset %= v_size; // reset v position.

      if (jc + y.offset(1) == real_i()) {
        y.a(real_i())   = v[v_offset];  // modify diagonal
      } else {
        y.ija(pp)       = jc;           // modify non-diagonal
        y.a(pp)         = v[v_offset];
      }
    }

    // Update this row.
    adjust_length(nd_change);

    return row_stored_nd_iterator(*this, pp);
  }

  /*
   * For when we don't need to worry about the offset, does the same thing as the insert above.
   */
  row_stored_nd_iterator insert(const row_stored_nd_iterator& position, size_t jj, size_t length, D const* v, size_t v_size) {
    size_t v_offset = 0;
    return insert(position, jj, length, v, v_size, v_offset);
  }


  /*
   * Merges elements offered for insertion with existing elements in the row.
   */
  row_stored_nd_iterator insert(size_t jj, size_t length, D const* v, size_t v_size, size_t& v_offset) {
    return insert(ndfind(jj), jj, length, v, v_size, v_offset);
  }

  /*
   * Merges elements offered for insertion with existing elements in the row.
   */
  row_stored_nd_iterator insert(size_t jj, size_t length, D const* v, size_t v_size) {
    size_t v_offset = 0;
    return insert(ndfind(jj), jj, length, v, v_size, v_offset);
  }


};

} } // end of nm::yale_storage namespace

#endif // YALE_ITERATORS_ROW_H