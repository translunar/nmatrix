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
#include <stdexcept>

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
  VALUE nm_yale_each_ordered_stored_with_indices(VALUE nmatrix);
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
  inline const D& default_obj() const { return a(s->shape[0]); }
  inline const D& const_default_obj() const { return a(s->shape[0]); }

  inline I* ija_p()       const       { return reinterpret_cast<I*>(s->ija); }
  inline const I& ija(size_t p) const { return ija_p()[p]; }
  inline I& ija(size_t p)             { return ija_p()[p]; }
  inline D* a_p()         const       { return reinterpret_cast<D*>(s->a); }
  inline const D& a(size_t p) const   { return a_p()[p]; }
  inline D& a(size_t p)               { return a_p()[p]; }

  bool real_row_empty(size_t i) const { return ija(i+1) - ija(i) == 0 ? true : false; }

  inline size_t* shape_p()        const { return slice_shape;      }
  inline size_t  shape(uint8_t d) const { return slice_shape[d];   }
  inline size_t* real_shape_p() const { return s->shape;           }
  inline size_t  real_shape(uint8_t d) const { return s->shape[d]; }
  inline size_t* offset_p()     const { return slice_offset;       }
  inline size_t  offset(uint8_t d) const { return slice_offset[d]; }
  inline size_t  capacity() const { return s->capacity;            }
  inline size_t  size() const { return ija(real_shape(0));         }

  /*
   * This is the guaranteed maximum size of the IJA/A arrays of the matrix given its shape.
   */
  inline size_t real_max_size() const {
    size_t result = real_shape(0) * real_shape(1) + 1;
    if (real_shape(0) > real_shape(1))
      result += real_shape(0) - real_shape(1);

    return result;
  }

  // Binary search between left and right in IJA for column ID real_j. Returns left if not found.
  size_t real_find_pos(long left, long right, size_t real_j, bool& found) const {
    if (left > right) {
      found = false;
      return left;
    }

    size_t mid   = (left + right) / 2;
    size_t mid_j = ija(mid);

    if (mid_j == real_j) {
      found = true;
      return mid;
    } else if (mid_j > real_j)  return real_find_pos(left, mid - 1, real_j, found);
    else                        return real_find_pos(mid + 1, right, real_j, found);
  }

  // Binary search between left and right in IJA for column ID real_j. Essentially finds where the slice should begin,
  // with no guarantee that there's anything in there.
  size_t real_find_left_boundary_pos(size_t left, size_t right, size_t real_j) const {
    if (left > right) return right;
    if (ija(left) >= real_j) return left;

    size_t mid   = (left + right) / 2;
    size_t mid_j = ija(mid);

    if (mid_j == real_j)      return mid;
    else if (mid_j > real_j)  return real_find_left_boundary_pos(left, mid, real_j);
    else                      return real_find_left_boundary_pos(mid + 1, right, real_j);
  }

  // Binary search for coordinates i,j in the slice. If not found, return -1.
  std::pair<size_t,bool> find_pos(const std::pair<size_t,size_t>& ij) const {
    size_t left   = ija(ij.first + offset(0));
    size_t right  = ija(ij.first + offset(0) + 1) - 1;

    std::pair<size_t, bool> result;
    result.first = real_find_pos(left, right, ij.second + offset(1), result.second);
    return result;
  }

  // Binary search for coordinates i,j in the slice, and return the first position >= j in row i.
  size_t find_pos_for_insertion(size_t i, size_t j) const {
    size_t left   = ija(i + offset(0));
    size_t right  = ija(i + offset(0) + 1) - 1;
    return real_find_left_boundary_pos(left, right, j + offset(1));
  }


  /*
   * Iterator base class (pure virtual).
   */
  class basic_iterator {
    friend class YaleStorage<D>;
  protected:
    YaleStorage<D>* y;
    size_t i_;
    I p_;

    inline size_t offset(size_t d) const { return y->offset(d); }
    inline size_t shape(size_t d) const { return y->shape(d); }
    inline size_t real_shape(size_t d) const { return y->real_shape(d); }
    inline I ija(size_t pp) const { return y->ija(pp); }
    inline I& ija(size_t pp) { return y->ija(pp); }

    virtual bool diag() const {
      return p_ < std::min(y->real_shape(0), y->real_shape(1));
    }
    virtual bool done_with_diag() const {
      return p_ == std::min(y->real_shape(0), y->real_shape(1));
    }
    virtual bool nondiag() const {
      return p_ > std::min(y->real_shape(0), y->real_shape(1));
    }

  public:
    basic_iterator(YaleStorage<D>* obj, size_t ii = 0, I pp = 0) : y(obj), i_(ii), p_(pp) { }

    virtual inline size_t i() const { return i_; }
    virtual size_t j() const = 0;

    virtual inline VALUE rb_i() const { return LONG2NUM(i()); }
    virtual inline VALUE rb_j() const { return LONG2NUM(j()); }

    virtual size_t real_i() const { return offset(0) + i(); }
    virtual size_t real_j() const { return offset(1) + j(); }
    virtual size_t p() const { return p_; }
    virtual bool real_ndnz_exists() const { return !y->real_row_empty(real_i()) && ija(p_) == real_j(); }

    virtual const D& operator*() const = 0;


    // Ruby VALUE de-reference
    virtual VALUE operator~() const {
      if (typeid(D) == typeid(RubyObject)) return (**this); // FIXME: return rval instead, faster;
      else return RubyObject(*(*this)).rval;
    }

    virtual bool operator==(const std::pair<size_t,size_t>& ij) {
      if (p() >= ija(real_shape(0))) return false;
      else return i() == ij.first && j() == ij.second;
    }

    virtual bool operator==(const basic_iterator& rhs) const {
      return i() == rhs.i() && j() == rhs.j();
    }
    virtual bool operator!=(const basic_iterator& rhs) const {
      return i() != rhs.i() || j() != rhs.j();
    }
  };


  /*
   * Iterate across the stored diagonal.
   */
  class stored_diagonal_iterator : public basic_iterator {
    using basic_iterator::p_;
    using basic_iterator::y;
    friend class YaleStorage<D>;
  public:
    stored_diagonal_iterator(YaleStorage<D>* obj, size_t d = 0)
    : basic_iterator(obj,                // y
                     std::max(obj->offset(0), obj->offset(1)) + d - obj->offset(0), // i_
                     std::max(obj->offset(0), obj->offset(1)) + d) // p_
    {
//      std::cerr << "sdbegin: d=" << d << ", p_=" << p_ << ", i()=" << i() << ", j()=" << j() << std::endl;
      // p_ can range from max(y->offset(0), y->offset(1)) to min(y->real_shape(0), y->real_shape(1))
    }


    inline size_t d() const {
      return p_ - std::max(offset(0), offset(1));
    }

    stored_diagonal_iterator& operator++() {
      ++p_;
      return *this;
    }

    stored_diagonal_iterator operator++(int dummy) const {
      stored_diagonal_iterator iter(*this);
      return ++iter;
    }

    virtual inline size_t i() const {
      return p_ - offset(0);
    }

    virtual inline size_t j() const {
      return p_ - offset(1);
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

    D& operator*() {
      return y->a(p_);
    }

    const D& operator*() const {
      return y->a(p_);
    }
  };

  /*
   * Iterate across the stored non-diagonals.
   */
  class stored_nondiagonal_iterator : public basic_iterator {
    using basic_iterator::i_;
    using basic_iterator::p_;
    using basic_iterator::y;
    friend class YaleStorage<D>;
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
        p_ = y->find_pos_for_insertion(i_, 0);
        //std::cerr << "A: " << p_ << std::endl;
      }

      // If the row is entirely empty, we know we need to go to the next row.
      // Even if the row isn't empty, an invalid p needs to be advanced.
      while (is_valid_empty_real_row() || ija(p_) < offset(1) || ija(p_) - offset(1) >= shape(1)) {
        ++i_;

        // If we're in a valid row, we can try looking for p again.
        if (is_valid_nonempty_real_row()) {
          p_ = y->find_pos_for_insertion(i_, 0);
          //std::cerr << "B1: " << p_ << std::endl;
        } else {
          p_ = ija(i_ + offset(0) + 1); // beginning of next valid row
          //std::cerr << "B2: " << p_ << std::endl;
        }

        if (p_ >= y->size()) { // abort -- empty matrix
          p_ = y->size();
          i_ = y->shape(0);
          break;
        }
      }

      //std::cerr << "advance: i = " << i_ << ", p_ = " << p_ << std::endl;
    }


  public:
    stored_nondiagonal_iterator(YaleStorage<D>* obj, size_t ii = 0)
    : basic_iterator(obj, ii, obj->ija(ii + obj->offset(0)))
    {
      if (ii < shape(0))
        advance_to_first_valid_entry();
    }


    stored_nondiagonal_iterator(YaleStorage<D>* obj, size_t ii, size_t pp) : basic_iterator(obj, ii, pp) { }


    // Pre-condition: p >= real_row_begin
    bool find_valid_p_for_row() {
      size_t real_row_begin = ija(i_ + offset(0)),
             real_row_end   = ija(i_ + offset(0) + 1);
      // if (p_ < real_row_begin) return false; // This is a pre-condition
      if (real_row_begin == real_row_end) return false;
      if (p_ >= real_row_end) return false;

      if (ija(p_) < offset(1)) {
        size_t next_p = y->find_pos_for_insertion(i_, 0);
        if (p_ == next_p) return false;
        else p_ = next_p;

        if (p_ >= real_row_end) return false;
        if (ija(p_) < offset(1)) return false;
        if (j() < shape(1)) return true; // binary search worked!
        return false; // out of range. not in row.
      } else if (j() < shape(1)) return true; // within the necessary range.
      else
        return false; // out of range. not in row.
    }


    stored_nondiagonal_iterator& operator++() {

      size_t real_row_begin = ija(i_ + offset(0)),
             real_row_end   = ija(i_ + offset(0) + 1);
      ++p_;

      if (p_ >= y->size()) {
        i_ = shape(0);
        return *this;
      }

      while (i_ < shape(0) && !find_valid_p_for_row()) { // skip forward
        ++i_;
      }

      return *this;
    }

    stored_nondiagonal_iterator operator++(int dummy) const {
      stored_nondiagonal_iterator iter(*this);
      return ++iter;
    }

    virtual bool operator==(const stored_nondiagonal_iterator& rhs) const {
      if (i_ != rhs.i_) return false;
      if (i_ > shape(0) && rhs.i_ > rhs.shape(0)) return true;   // handles sndend()
      else return j() == rhs.j();
    }

    virtual bool operator!=(const stored_nondiagonal_iterator& rhs) const {
      if (i_ > shape(0)) {
        if (rhs.i_ > rhs.shape(0)) return false;  // all sndend iterators are ==
        else                        return true;   // one is sndend and one isn't
      } else if (rhs.i_ > rhs.shape(0)) return true;

      return (i_ != rhs.i_ || j() != rhs.j());
    }

    virtual bool operator<(const stored_diagonal_iterator& rhs) const {
      if (i_ < rhs.i()) return true;
      else if (i_ == rhs.i()) return j() < rhs.j();
      else return false;
    }

    virtual bool operator>(const stored_diagonal_iterator& rhs) const {
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

    D& operator*() {
      return y->a(p_);
    }

    const D& operator*() const {
      return y->a(p_);
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
    using basic_iterator::p_;
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
        if (!diag()) {
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
    virtual size_t p() const { return iter->p(); }
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
      // Need to keep these up to date for easier copy construction.
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

    virtual const D& operator*() const {
      return **iter;
    }
  };

  /*
   * Iterator for traversing matrix class as if it were dense (visits each entry in order).
   */
  class iterator : public basic_iterator {
    friend class YaleStorage<D>;
    using basic_iterator::i_;
    using basic_iterator::p_;
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
        p_ = offset(0) > 0 ? y->find_pos_for_insertion(i_,j_) : ija(i_);
      } else {
        // If the last j was actually stored in this row of the matrix, need to advance p.

        if (!y->real_row_empty(i_ + offset(0)) && ija(p_) <= prev_j + offset(1)) ++p_;  // this test is the same as real_ndnz_exists
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

    virtual bool diag() const { return i_ + offset(0) == j_ + offset(1); }

    // De-reference
    virtual const D& operator*() const {
      if (diag())                                                                return y->a( i_ + offset(0) );
      else if (p_ >= ija(i_+offset(0)+1))                                        return y->const_default_obj();
      else if (!y->real_row_empty(i_ + offset(0)) && ija(p_) == j_ + offset(1))  return y->a( p_ );
      else                                                                       return y->const_default_obj();
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
    stored_diagonal_iterator     d_iter;
    stored_nondiagonal_iterator nd_iter;
    bool d; // which iterator is the currently valid one

  public:
    ordered_iterator(YaleStorage<D>* obj, size_t ii = 0)
    : basic_iterator(obj, ii),
      d_iter(obj, ii),
      nd_iter(obj, ii),
      d(nd_iter > d_iter)
    {
/*      std::cerr << "d:  " << d_iter.i() << ", " << d_iter.j() << std::endl;
      std::cerr << "nd: " << nd_iter.i() << ", " << nd_iter.j() << std::endl;
      std::cerr << "dominant: " << (d ? "d" : "nd") << std::endl; */

    }

    virtual size_t j() const { return d ? d_iter.j() : nd_iter.j(); }
    virtual size_t i() const { return d ? d_iter.i() : nd_iter.i(); }
    virtual size_t real_j() const { return d ? d_iter.real_j() : nd_iter.real_j(); }
    virtual size_t real_i() const { return d ? d_iter.real_i() : nd_iter.real_i(); }

    ordered_iterator& operator++() {
      // FIXME: This can be sped up by only checking when necessary for nd_iter > d_iter. I believe
      // it only needs to be done once per row, and maybe never depending upon slice shape. Right?
      //std::cerr << "++" << std::endl;
      if (d)    ++d_iter;
      else      ++nd_iter;
      d = nd_iter > d_iter;
      return *this;
    }

    ordered_iterator operator++(int dummy) const {
      ordered_iterator iter(*this);
      return ++iter;
    }

    virtual bool operator==(const ordered_iterator& rhs) const {
      return d ? rhs == d_iter : rhs == nd_iter;
    }

    virtual bool operator==(const stored_diagonal_iterator& rhs) const {
      return i() == rhs.i() && j() == rhs.j();
    }

    virtual bool operator==(const stored_nondiagonal_iterator& rhs) const {
      return i() == rhs.i() && j() == rhs.j();
    }

    virtual bool operator!=(const ordered_iterator& rhs) const {
      return d ? rhs != d_iter : rhs != nd_iter;
    }

    virtual bool operator!=(const stored_diagonal_iterator& rhs) const {
      return i() != rhs.i() || j() != rhs.j();
    }

    virtual bool operator!=(const stored_nondiagonal_iterator& rhs) const {
      return i() != rhs.i() || j() != rhs.j();
    }


    // De-reference the iterator
    virtual D& operator*() {
      return d ? *d_iter : *nd_iter;
    }

    virtual const D& operator*() const {
      return d ? *d_iter : *nd_iter;
    }
  };

  // Variety of iterator begin and end functions.
  iterator begin(size_t row = 0)                      {      return iterator(this, row);               }
  iterator row_end(size_t row)                        {      return begin(row+1);                      }
  iterator end()                                      {      return iterator(this, shape(0));          }
  stored_diagonal_iterator sdbegin(size_t d = 0)      {      return stored_diagonal_iterator(this, d); }
  stored_diagonal_iterator sdend()                    {
    return stored_diagonal_iterator(this, std::min( shape(0) + offset(0), shape(1) + offset(1) ) - std::max(offset(0), offset(1)) );
  }
  stored_nondiagonal_iterator sndbegin(size_t row = 0){      return stored_nondiagonal_iterator(this, row); }
  stored_nondiagonal_iterator sndrow_end(size_t row)  {      return sndbegin(row+1);                   }
  stored_nondiagonal_iterator sndend()                {      return stored_nondiagonal_iterator(this, shape(0)); }
  stored_iterator sbegin()                            {      return stored_iterator(this, true);       }
  stored_iterator send()                              {      return stored_iterator(this, false);      }
  ordered_iterator obegin(size_t row = 0)             {      return ordered_iterator(this, row);       }
  ordered_iterator oend()                             {      return ordered_iterator(this, shape(0));  }
  ordered_iterator orow_end(size_t row)               {      return obegin(row+1);                     }


  /*
   * Returns the iterator for i,j or snd_end() if not found.
   */
  stored_nondiagonal_iterator find(const std::pair<size_t,size_t>& ij) {
    std::pair<size_t,bool> find_pos_result = find_pos(ij);
    if (!find_pos_result.second) return sndend();
    else return stored_nondiagonal_iterator(this, ij.first, find_pos_result.first);
  }

  /*
   * Returns a stored_nondiagonal_iterator pointing to the location where some coords i,j should go, or returns their
   * location if present.
   */
  stored_nondiagonal_iterator lower_bound(const std::pair<size_t,size_t>& ij) {
    return stored_nondiagonal_iterator(this, ij.first, find_pos_for_insertion(ij.first, ij.second));
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
  bool insert(stored_nondiagonal_iterator position, size_t i, size_t j, const D& val) {
    return insert(position, std::make_pair(i,j), val);
  }

  /*
   * See the above insert.
   */
  bool insert(stored_nondiagonal_iterator position, const std::pair<size_t,size_t>& ij, const D& val) {
    size_t  i = ij.first,
            j = ij.second,
           sz = size();

    if (position != ij) {
      *position = val; // replace
      return false;
    } else if (sz + 1 > capacity()) {
      update_resize_move(position, ij.first+offset(0), 1);
    } else {
      move_right(position, 1);
      update_real_row_sizes_from(ij.first+offset(0), 1);
    }
    ija(position.p()) = j + offset(1); // set the column ID
    a(position.p())   = val;           // set the value
    return true;
  }

  /*
   * Insert n elements v in columns j, using position as a guide. i gives the starting row. If at any time a value in j
   * decreases,
   */
  /*bool insert(stored_iterator position, size_t n, size_t i, size_t* j, DType* v) {

  } */

  /*
   * A pseudo-insert operation, since the diagonal portion of the A array is constant size.
   */
  stored_diagonal_iterator insert(stored_diagonal_iterator position, const D& val) {
    *position = val;
    return position;
  }

  iterator insert(iterator position, size_t j, const D& val) {
    if (position.real_i() == position.real_j()) {
      s->a(position.real_i()) = val;
      return position;
    } else {
      return insert(stored_nondiagonal_iterator(position), position.i(), j, val);
    }
  }

  // Simple insertion/getting of an element -- happens when [] is called.
  inline D& operator[](const std::pair<size_t,size_t>& ij) {
    if (ij.first > shape(0) || ij.second > shape(1)) rb_raise(rb_eRangeError, "element access out of range at %u, %u", ij.first, ij.second);
    if (ij.first + offset(0) == ij.second + offset(1)) return a(ij.first + offset(0));
    stored_nondiagonal_iterator iter = lower_bound(ij);
    if (iter != ij) { // if not found, insert the default
      insert(iter, ij, const_default_obj());
    }
    // we can now safely return a reference
    return *iter;
  }

  /*
   * Attempt to return a reference to some location i,j. Not Ruby-safe; will throw out_of_range if not found.
   */
  inline D& at(const std::pair<size_t,size_t>& ij) {
    if (ij.first > shape(0) || ij.second > shape(1)) throw std::out_of_range("i,j out of bounds");
    if (ij.first + offset(0) == ij.second + offset(1)) return a(ij.first + offset(0));
    stored_nondiagonal_iterator iter = find(ij);
    if (iter != ij) throw std::out_of_range("i,j not found in matrix");
    return *iter;
  }

  // See above.
  inline const D& at(const std::pair<size_t,size_t>& ij) const {
    if (ij.first > shape(0) || ij.second > shape(1)) throw std::out_of_range("i,j out of bounds");
    if (ij.first + offset(0) == ij.second + offset(1)) return a(ij.first + offset(0));
    stored_nondiagonal_iterator iter = find(ij);
    if (iter != ij) throw std::out_of_range("i,j not found in matrix");
    return *iter;
  }

protected:
  /*
   * Update row sizes starting with row i
   */
  void update_real_row_sizes_from(size_t real_i, int change) {
    for (; real_i <= real_shape(0); ++real_i) {
      ija(real_i) += change;
    }
  }

  /*
   * Move elements in the IJA and A arrays by n (to the right).
   * Does not update row sizes.
   */
  void move_right(stored_nondiagonal_iterator position, size_t n) {
    size_t sz = size();
    for (size_t m = 0; m < sz - position.p(); ++m) {
      ija(sz+n-1-m) = ija(sz-1-m);
      a(sz+n-1-m)   = a(sz-1-m);
    }
  }

  /*
   * Like move_right, but also involving a resize. This updates row sizes as well.
   */
  void update_resize_move(stored_nondiagonal_iterator position, size_t real_i, int n) {
    size_t sz      = size(); // current size of the storage vectors
    size_t new_cap = capacity() * nm::yale_storage::GROWTH_CONSTANT;
    size_t max_cap = real_max_size();

    if (new_cap > max_cap) {
      new_cap = max_cap;
      if (sz + n > max_cap)
        rb_raise(rb_eStandardError, "insertion size exceeded maximum yale matrix size");

    }

    if (new_cap < sz + n) new_cap = sz + n;

    IType* new_ija      = ALLOC_N( I,     new_cap );
    D* new_a            = ALLOC_N( D,     new_cap );

    // Copy unchanged row pointers first.
    for (size_t m = 0; m <= real_i; ++m) {
      new_ija[m]        = ija(m);
      new_a[m]          = a(m);
    }

    // Now update row pointers following the changed row as we copy the additional values.
    for (size_t m = real_i + 1; m < real_shape(0); ++m) {
      new_ija[m]        = ija(m) + n;
      new_a[m]          = a(m);
    }

    // Copy all remaining prior to insertion/removal site
    for (size_t m = real_shape(0); m < position.p(); ++m) {
      new_ija[m]        = ija(m);
      new_a[m]          = a(m);
    }

    // Copy all subsequent to insertion/removal site
    for (size_t m = position.p(); m < sz; ++m) {
      new_ija[m+n]      = ija(m);
      new_a[m+n]        = a(m);
    }

    s->capacity = new_cap;

    xfree(s->ija);
    xfree(s->a);

    s->ija      = new_ija;
    s->a        = reinterpret_cast<void*>(new_a);
  }

  /*
   * Move elements in the IJA and A arrays by n (to the left). Here position gives
   * the location to move to, and they should come from n to the right.
   */
  void move_left(stored_nondiagonal_iterator position, size_t n) {
    size_t sz = size();
    for (size_t m = sz; m > position.p() + n; --m) {   // work backwards
      ija(m-n)      = ija(m);
      a(m-n)        = a(m);
    }
  }

  YALE_STORAGE* s;
  bool          slice;
  size_t*       slice_shape;
  size_t*       slice_offset;
};

} // end of namespace nm

#endif // YALE_H
