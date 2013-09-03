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
// == yale.h
//
// "new yale" storage format for 2D matrices (like yale, but with
// the diagonal pulled out for O(1) access).
//
// Specifications:
// * dtype and index dtype must necessarily differ
//      * index dtype is defined by whatever unsigned type can store
//        max(rows,cols)
//      * that means vector ija stores only index dtype, but a stores
//        dtype
// * vectors must be able to grow as necessary
//      * maximum size is rows*cols+1

#ifndef YALE_H
#define YALE_H

/*
 * Standard Includes
 */

#include <limits> // for std::numeric_limits<T>::max()

/*
 * Project Includes
 */

#include "types.h"
#include "data/data.h"
#include "common.h"
#include "nmatrix.h"

extern "C" {

  /*
   * Macros
   */

  #define NM_YALE_MINIMUM(sptr)               (((YALE_STORAGE*)(sptr))->shape[0]*2 + 1) // arbitrarily defined

  #ifndef NM_CHECK_ALLOC
   #define NM_CHECK_ALLOC(x) if (!x) rb_raise(rb_eNoMemError, "insufficient memory");
  #endif

  /*
   * Types
   */


  /*
   * Data
   */


  /*
   * Functions
   */

  ///////////////
  // Lifecycle //
  ///////////////

  YALE_STORAGE* nm_yale_storage_create(nm::dtype_t dtype, size_t* shape, size_t dim, size_t init_capacity);
  YALE_STORAGE* nm_yale_storage_create_from_old_yale(nm::dtype_t dtype, size_t* shape, char* ia, char* ja, char* a, nm::dtype_t from_dtype);
  YALE_STORAGE*	nm_yale_storage_create_merged(const YALE_STORAGE* merge_template, const YALE_STORAGE* other);
  void          nm_yale_storage_delete(STORAGE* s);
  void          nm_yale_storage_delete_ref(STORAGE* s);
  void					nm_yale_storage_init(YALE_STORAGE* s, void* default_val);
  void					nm_yale_storage_mark(void*);

  ///////////////
  // Accessors //
  ///////////////

  VALUE nm_yale_each_with_indices(VALUE nmatrix);
  VALUE nm_yale_each_stored_with_indices(VALUE nmatrix);
  void* nm_yale_storage_get(STORAGE* s, SLICE* slice);
  void*	nm_yale_storage_ref(STORAGE* s, SLICE* slice);
  void  nm_yale_storage_set(VALUE left, SLICE* slice, VALUE right);

  //char  nm_yale_storage_vector_insert(YALE_STORAGE* s, size_t pos, size_t* js, void* vals, size_t n, bool struct_only, nm::dtype_t dtype, nm::itype_t itype);
  //void  nm_yale_storage_increment_ia_after(YALE_STORAGE* s, size_t ija_size, size_t i, size_t n);

  size_t  nm_yale_storage_get_size(const YALE_STORAGE* storage);
  VALUE   nm_yale_default_value(VALUE self);
  VALUE   nm_yale_map_stored(VALUE self);
  VALUE   nm_yale_map_merged_stored(VALUE left, VALUE right, VALUE init);

  ///////////
  // Tests //
  ///////////

  bool nm_yale_storage_eqeq(const STORAGE* left, const STORAGE* right);

  //////////
  // Math //
  //////////

  STORAGE* nm_yale_storage_matrix_multiply(const STORAGE_PAIR& casted_storage, size_t* resulting_shape, bool vector);

  /////////////
  // Utility //
  /////////////



  /////////////////////////
  // Copying and Casting //
  /////////////////////////

  STORAGE*      nm_yale_storage_cast_copy(const STORAGE* rhs, nm::dtype_t new_dtype, void*);
  STORAGE*      nm_yale_storage_copy_transposed(const STORAGE* rhs_base);



  void nm_init_yale_functions(void);

  VALUE nm_vector_set(int argc, VALUE* argv, VALUE self);


} // end of extern "C" block

namespace nm { namespace yale_storage {
  typedef size_t IType;

  /*
   * Constants
   */
  const float GROWTH_CONSTANT = 1.5;


  /*
   * Templated Functions
   */

  int binary_search(YALE_STORAGE* s, IType left, IType right, IType key);

  /*
   * Clear out the D portion of the A vector (clearing the diagonal and setting
   * the zero value).
   *
   * Note: This sets a literal 0 value. If your dtype is RUBYOBJ (a Ruby object),
   * it'll actually be INT2FIX(0) instead of a string of NULLs. You can actually
   * set a default for Ruby objects other than zero -- you generally want it to
   * be Qfalse, Qnil, or INT2FIX(0). The last is the default.
   */
  template <typename DType>
  inline void clear_diagonal_and_zero(YALE_STORAGE* s, void* init_val) {
    DType* a = reinterpret_cast<DType*>(s->a);

    // Clear out the diagonal + one extra entry
    if (init_val) {
      for (size_t i = 0; i <= s->shape[0]; ++i) // insert Ruby zeros, falses, or whatever else.
        a[i] = *reinterpret_cast<DType*>(init_val);
    } else {
      for (size_t i = 0; i <= s->shape[0]; ++i) // insert zeros.
        a[i] = 0;
    }
  }

  template <typename DType>
  void init(YALE_STORAGE* s, void* init_val);

  size_t  get_size(const YALE_STORAGE* storage);

  IType binary_search_left_boundary(const YALE_STORAGE* s, IType left, IType right, IType bound);

} // end of namespace yale_storage

// namespace nm

/*
 * This class is basically an intermediary for YALE_STORAGE objects which enables us to treat it like a C++ object. It
 * keeps the src pointer as its s, along with other relevant slice information.
 *
 * It's useful for creating iterators and such. It isn't responsible for allocating or freeing its YALE_STORAGE* pointers.
 */
template <typename D>
class YaleStorage {
  typedef size_t I;
public:
  YaleStorage(const YALE_STORAGE* storage)
   : s(reinterpret_cast<YALE_STORAGE*>(storage->src)),
     slice(storage != storage->src),
     slice_shape(storage->shape),
     slice_offset(storage->offset)
  { }

  YaleStorage(const STORAGE* storage)
   : s(reinterpret_cast<YALE_STORAGE*>(storage->src)),
     slice(storage != storage->src),
     slice_shape(storage->shape),
     slice_offset(storage->offset)
  { }


  bool is_ref() const { return slice; }

  inline D* default_obj_ptr() { return &(a(s->shape[0])); }
  inline D& default_obj() { return a(s->shape[0]); }
  inline D default_obj() const { return a(s->shape[0]); }
  inline D const_default_obj() const { return a(s->shape[0]); }

  inline I* ija_p()     const { return reinterpret_cast<I*>(s->ija); }
  inline I  ija(long p) const { return ija_p()[p]; }
  inline I& ija(long p)       { return ija_p()[p]; }
  inline D* a_p()       const { return reinterpret_cast<D*>(s->a); }
  inline D  a(long p)   const { return a_p()[p]; }

  bool real_row_empty(size_t i) const { return ija(i+1) - ija(i) == 0 ? true : false; }

  inline size_t* shape_p()        const { return slice_shape;      }
  inline size_t  shape(uint8_t d) const { return slice_shape[d];   }
  inline size_t* real_shape_p() const { return s->shape;           }
  inline size_t  real_shape(uint8_t d) const { return s->shape[d]; }
  inline size_t* offset_p()     const { return slice_offset;       }
  inline size_t  offset(uint8_t d) const { return slice_offset[d]; }

  // Binary search between left and right in IJA for column ID real_j. Returns left if not found.
  long real_find_pos(long left, long right, size_t real_j) {
    if (left > right) return -1;

    size_t mid   = (left + right) / 2;
    size_t mid_j = ija(mid);

    if (mid_j == real_j)      return mid;
    else if (mid_j > real_j)  return real_find_pos(left, mid - 1, real_j);
    else                      return real_find_pos(mid + 1, right, real_j);
  }

  // Binary search between left and right in IJA for column ID real_j. Essentially finds where the slice should begin,
  // with no guarantee that there's anything in there.
  long real_find_left_boundary_pos(size_t left, size_t right, size_t real_j) {
    if (left > right) return right;
    if (ija(left) >= real_j) return left;

    size_t mid   = (left + right) / 2;
    size_t mid_j = ija(mid);

    if (mid_j == real_j)      return mid;
    else if (mid_j > real_j)  return real_find_left_boundary_pos(left, mid, real_j);
    else                      return real_find_left_boundary_pos(mid + 1, right, real_j);
  }

  // Binary search for coordinates i,j in the slice. If not found, return -1.
  long find_pos(size_t i, size_t j) {
    size_t left   = ija(i + slice_offset[0]);
    size_t right  = ija(i + slice_offset[0] + 1) - 1;
    return real_find_pos(left, right, j + slice_offset[1]);
  }

  // Binary search for coordinates i,j in the slice, and return the first position >= j in row i.
  size_t find_pos_for_insertion(size_t i, size_t j) {
    size_t left   = ija(i + slice_offset[0]);
    size_t right  = ija(i + slice_offset[0] + 1) - 1;
    return real_find_left_boundary_pos(left, right, j + slice_offset[1]);
  }


  /*
   * Iterator base class (pure virtual).
   */
  class basic_iterator {
    friend class YaleStorage<D>;
  protected:
    YaleStorage<D>* y;
    size_t i_;
    I p;

    inline size_t offset(size_t d) const { return y->offset(d); }
    inline size_t shape(size_t d) const { return y->shape(d); }
    inline size_t real_shape(size_t d) const { return y->real_shape(d); }
    inline I ija(size_t pp) const { return y->ija(pp); }
    inline I& ija(size_t pp) { return y->ija(pp); }

    virtual bool diag() const {
      return p < std::min(y->real_shape(0), y->real_shape(1));
    }
    virtual bool done_with_diag() const {
      return p == std::min(y->real_shape(0), y->real_shape(1));
    }
    virtual bool nondiag() const {
      return p > std::min(y->real_shape(0), y->real_shape(1));
    }

  public:
    basic_iterator(YaleStorage<D>* obj, size_t ii = 0, I pp = 0) : y(obj), i_(ii), p(pp) { }

    virtual inline size_t i() const { return i_; }
    virtual size_t j() const = 0;

    virtual inline VALUE rb_i() const { return LONG2NUM(i()); }
    virtual inline VALUE rb_j() const { return LONG2NUM(j()); }

    virtual size_t real_i() const { return offset(0) + i(); }
    virtual size_t real_j() const { return offset(1) + j(); }
    virtual bool real_ndnz_exists() const { return !y->real_row_empty(real_i()) && ija(p) == real_j(); }

    virtual D operator*() const = 0;

    // Ruby VALUE de-reference
    virtual VALUE operator~() const {
      if (typeid(D) == typeid(RubyObject)) return *(*this);
      else return RubyObject(*(*this)).rval;
    }
  };


  /*
   * Iterate across the stored diagonal.
   */
  class stored_diagonal_iterator : public basic_iterator {
    using basic_iterator::i_;
    using basic_iterator::p;
    friend class YaleStorage<D>;
  public:
    stored_diagonal_iterator(YaleStorage<D>* obj, size_t d = 0)
    : basic_iterator(obj,                // y
                     std::max(obj->offset(0), obj->offset(1)) + d - obj->offset(0), // i_
                     std::max(obj->offset(0), obj->offset(1)) + d) // p
    {
      // p can range from max(y->offset(0), y->offset(1)) to min(y->real_shape(0), y->real_shape(1))
    }


    inline size_t d() const {
      return p - std::max(offset(0), offset(1));
    }

    stored_diagonal_iterator& operator++() {
      i_ = ++p - offset(0);
      return *this;
    }

    stored_diagonal_iterator operator++(int dummy) const {
      stored_diagonal_iterator iter(*this);
      return ++iter;
    }

    virtual inline size_t j() const {
      return i_ + offset(0) - offset(1);
    }


    virtual bool operator!=(const stored_diagonal_iterator& rhs) const { return d() != rhs.d(); }
    virtual bool operator==(const stored_diagonal_iterator& rhs) const { return !(*this != rhs); }
    virtual bool operator<(const stored_diagonal_iterator& rhs) const {  return d() < rhs.d(); }

    virtual bool operator<=(const stored_diagonal_iterator& rhs) const {
      return d() <= rhs.d();
    }

    virtual bool operator>(const stored_diagonal_iterator& rhs) const {
      return d() > rhs.d();
    }

    virtual bool operator>=(const stored_diagonal_iterator& rhs) const {
      return d() >= rhs.d();
    }
  };

  /*
   * Iterate across the stored non-diagonals.
   */
  class stored_nondiagonal_iterator : public basic_iterator {
    using basic_iterator::i_;
    using basic_iterator::p;
    friend class YaleStorage<D>;
  protected:

    virtual bool in_valid_nonempty_real_row() const {
      return i_ < shape(0) && ija(i_ + offset(0)) < ija(i_ + offset(0)+1);
    }

    virtual bool in_valid_empty_real_row() const {
      return i_ < shape(0) && ija(i_ + offset(0)) == ija(i_ + offset(0)+1);
    }

    // Key loop for forward row iteration in the non-diagonal portion of the matrix. Called during construction and by
    // the ++ operators.
    void advance_next_nonempty_row() {
      if (in_valid_nonempty_real_row())
        p = this->y->find_pos_for_insertion(i_, 0);

      while (in_valid_empty_real_row() || j() >= shape(1)) {
        ++i_;

        if (in_valid_nonempty_real_row()) {
          p = this->y->find_pos_for_insertion(i_, 0); // find the beginning of this row
        } else if (i_ >= shape(0)) {
          p = ija(real_shape(0));         // find the end of the matrix
          break;
        }
      }
    }

    // Key loop for forward column iteration in the non-diagonal portion of the matrix. Called by the ++operator.
    bool advance_next_column() {
      if (i_ >= shape(0) || in_valid_empty_real_row())    return false;
      if (p < ija(i_ + offset(0)+1)-1) ++p; // advance to next column
      if (j() < shape(1)) return true;         // see if we found a valid column
      return false;                               // nope.
    }

  public:
    stored_nondiagonal_iterator(YaleStorage<D>* obj, bool end, size_t ii = 0)
    : basic_iterator(obj,
                     end ? obj->real_shape(0) : ii,
                     end ? obj->ija(ii + obj->offset(0) + 1) : std::max(obj->offset(0), obj->offset(1)))
    {
      if (begin) advance_next_nonempty_row();
    }

    stored_nondiagonal_iterator& operator++() {
      while (i_ < shape(0) && !advance_next_column()) { // if advancing to the next column fails,
        advance_next_nonempty_row();                       // then go to the next row.
      }
      return *this;
    }

    stored_nondiagonal_iterator operator++(int dummy) const {
      stored_nondiagonal_iterator iter(*this);
      return ++iter;
    }

    virtual inline size_t j() const {
      return ija(p) - offset(1);
    }

  };



  /*
   * Iterate across a matrix in storage order.
   *
   * Note: It is not recommended that this iterator be compared to an iterator from another matrix, as storage order
   * determines iteration -- and storage order will differ for slices which are positioned differently relative to the
   * original matrix.
   */
  class stored_iterator : public basic_iterator {
    friend class YaleStorage<D>;
    using basic_iterator::i_;
    using basic_iterator::p;
    using basic_iterator::y;
  protected:
    virtual bool diag() const { return iter->diag(); }
    virtual bool nondiag() const { return iter->nondiag(); }

    basic_iterator* iter;

  public:
    stored_iterator(YaleStorage<D>* obj, bool begin = true) : basic_iterator(obj, 0, 0) {
      if (begin) {
        iter = new stored_diagonal_iterator(obj);

        // if we're past the diagonal already, delete it and create a nondiagonal iterator
        if (!iter.diag()) {
          delete iter;
          iter = new stored_nondiagonal_iterator(obj, true);
        }
      } else {
        iter = new stored_nondiagonal_iterator(obj, false);
      }
    }

    ~stored_iterator() {
      delete iter;
    }

    stored_iterator(const stored_iterator& rhs) {
      y = rhs.y;
      if (rhs.diag()) {
        iter = new stored_diagonal_iterator(*(stored_diagonal_iterator*)iter);
      } else {
        iter = new stored_nondiagonal_iterator(*(stored_nondiagonal_iterator*)iter);
      }
    }

    virtual size_t j() const { return iter->j(); }
    virtual size_t i() const { return iter->i(); }
    virtual size_t real_j() const { return iter->real_j(); }
    virtual size_t real_i() const { return iter->real_i(); }

    stored_iterator& operator++() {
      if (diag()) {
        ++(*iter);
        if (!iter.diag()) {
          delete iter;
          iter = new stored_nondiagonal_iterator(y);
        }
      } else {
        ++(*iter);
      }
      return *this;
    }

    stored_iterator operator++(int dummy) const {
      stored_iterator iter(*this);
      return ++iter;
    }

    virtual bool operator==(const stored_iterator& rhs) const {
      return *this == *(rhs->iter);
    }

    virtual bool operator!=(const stored_iterator& rhs) const {
      return *this != *(rhs->iter);
    }


    // De-reference the iterator
    virtual D& operator*() {
      return &(**iter);
    }

    virtual D operator*() const {
      return **iter;
    }
  };

  /*
   * Iterator for traversing matrix class as if it were dense (visits each entry in order).
   */
  class iterator : public basic_iterator {
    friend class YaleStorage<D>;
    using basic_iterator::i_;
    using basic_iterator::p;
    using basic_iterator::y;
  protected:
    size_t j_; // These are relative to the slice.

  public:
    // Create an iterator. May select the row since this is O(1).
    iterator(YaleStorage<D>* obj, size_t ii = 0) : basic_iterator(obj, ii, obj->ija(ii + obj->offset(0))), j_(0) { }

    // Prefix ++
    iterator& operator++() {
      size_t prev_j = j_++;
      if (j_ >= y->shape(1)) {
        j_ = 0;
        ++i_;

        // Do a binary search to find the beginning of the slice
        p = offset(0) > 0 ? y->find_pos_for_insertion(i_,j_) : ija(i_);
      } else {
        // If the last j was actually stored in this row of the matrix, need to advance p.

        if (!y->real_row_empty(i_ + offset(0)) && ija(p) <= prev_j + offset(1)) ++p;  // this test is the same as real_ndnz_exists
      }

      return *this;
    }

    iterator operator++(int dummy) const {
      iterator iter(*this);
      return ++iter;
    }

    virtual bool operator!=(const iterator& rhs) const {
      return (i_ != rhs.i_ || j_ != rhs.j_);
    }

    virtual bool operator==(const iterator& rhs) const {
      return !(*this != rhs);
    }

    bool operator<(const iterator& rhs) const {
      if (i_ > rhs.i_) return false;
      if (i_ < rhs.i_) return true;
      return j_ < rhs.j_;
    }

    bool operator>(const iterator& rhs) const {
      if (i_ < rhs.i_) return false;
      if (i_ > rhs.i_) return true;
      return j_ > rhs.j_;
    }

    virtual bool real_diag() const { return i_ + offset(0) == j_ + offset(1); }

    // De-reference
    virtual D operator*() const {
      if (real_diag())                                                          return y->a( i_ + offset(0) );
      else if (p >= ija(i_+offset(0)+1))                                        return y->const_default_obj();
      else if (!y->real_row_empty(i_ + offset(0)) && ija(p) == j_ + offset(1))  return y->a( p );
      else                                                                      return y->const_default_obj();
    }

    virtual size_t j() const { return j_; }
  };


  /*
   * The trickiest of all the iterators for Yale. We only want to visit the stored indices, but we want to visit them
   * in matrix order.
   */
  class ordered_iterator : public basic_iterator {
    friend class YaleStorage<D>;
  protected:
    bool d; // which iterator is the currently valid one
    stored_diagonal_iterator*     d_iter;
    stored_nondiagonal_iterator* nd_iter;

  public:
    ordered_iterator(YaleStorage<D>* obj, size_t ii = 0) : basic_iterator(obj, ii) {
      d_iter  = new stored_diagonal_iterator(obj);
      nd_iter = new stored_nondiagonal_iterator(obj, false, ii);
      d = *d_iter < *nd_iter;
    }

    ordered_iterator(const ordered_iterator& rhs) {
      d_iter  = new stored_diagonal_iterator(*d_iter);
      nd_iter = new stored_nondiagonal_iterator(*nd_iter);
      d       = rhs.d;
    }

    ~ordered_iterator() {
      delete  d_iter;
      delete nd_iter;
    }

    virtual size_t j() const { return d ? d_iter->j() : nd_iter->j(); }
    virtual size_t i() const { return d ? d_iter->i() : nd_iter->i(); }
    virtual size_t real_j() const { return d ? d_iter->real_j() : nd_iter->real_j(); }
    virtual size_t real_i() const { return d ? d_iter->real_i() : nd_iter->real_i(); }

    ordered_iterator& operator++() {
      if (d)    ++(*d_iter);
      else      ++(*nd_iter);
      d = *d_iter < *nd_iter;
      return *this;
    }

    ordered_iterator operator++(int dummy) const {
      ordered_iterator iter(*this);
      return ++iter;
    }

    virtual bool operator==(const ordered_iterator& rhs) const {
      return d ? rhs == *d_iter : rhs == *nd_iter;
    }

    virtual bool operator!=(const stored_iterator& rhs) const {
      return d ? rhs != *d_iter : rhs != *nd_iter;
    }


    // De-reference the iterator
    virtual D& operator*() {
      return d ? &(**d_iter) : &(**nd_iter);
    }

    virtual D operator*() const {
      return d ? **d_iter : **nd_iter;
    }
  };

  // Variety of iterator begin and end functions.
  iterator begin(size_t row = 0)                      {      return iterator(this, row);               }
  iterator row_end(size_t row)                        {      return begin(row+1);                      }
  iterator end()                                      {      return iterator(this, shape(0));          }
  stored_diagonal_iterator sdbegin(size_t d = 0)      {      return stored_diagonal_iterator(this, d); }
  stored_diagonal_iterator sdend()                    {      return stored_diagonal_iterator(this, std::min(real_shape(0), real_shape(1))); }
  stored_nondiagonal_iterator sndbegin(size_t row = 0){      return stored_nondiagonal_iterator(this, false, row); }
  stored_nondiagonal_iterator sndrow_end(size_t row)  {      return sndbegin(row+1);                   }
  stored_nondiagonal_iterator sndend()                {      return stored_nondiagonal_iterator(this, true); }
  stored_iterator sbegin()                            {      return stored_iterator(this, true);       }
  stored_iterator send()                              {      return stored_iterator(this, false);      }
  ordered_iterator obegin(size_t row = 0)             {      return ordered_iterator(this, row);       }
  ordered_iterator oend()                             {      return ordered_iterator(this, shape(0));  }
  ordered_iterator orow_end(size_t row)               {      return obegin(row+1);                     }

protected:
  YALE_STORAGE* s;
  bool          slice;
  size_t*       slice_shape;
  size_t*       slice_offset;
};

} // end of namespace nm

#endif // YALE_H
