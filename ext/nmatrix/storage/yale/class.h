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
// == class.h
//
// Object-oriented interface for Yale.
//

#ifndef YALE_CLASS_H
# define YALE_CLASS_H

namespace nm {


namespace yale_storage {
  /*
   * Constants
   */
  const float GROWTH_CONSTANT = 1.5;

}
/*
 * This class is basically an intermediary for YALE_STORAGE objects which enables us to treat it like a C++ object. It
 * keeps the src pointer as its s, along with other relevant slice information.
 *
 * It's useful for creating iterators and such. It isn't responsible for allocating or freeing its YALE_STORAGE* pointers.
 */

template <typename D>
class YaleStorage {
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

  /* Allows us to do YaleStorage<uint8>::dtype() to get an nm::dtype_t */
  static nm::dtype_t dtype() {
    return nm::ctype_to_dtype_enum<D>::value_type;
  }


  bool is_ref() const { return slice; }

  inline D* default_obj_ptr() { return &(a(s->shape[0])); }
  inline D& default_obj() { return a(s->shape[0]); }
  inline const D& default_obj() const { return a(s->shape[0]); }
  inline const D& const_default_obj() const { return a(s->shape[0]); }

  inline size_t* ija_p()       const       { return reinterpret_cast<size_t*>(s->ija); }
  inline const size_t& ija(size_t p) const { return ija_p()[p]; }
  inline size_t& ija(size_t p)             { return ija_p()[p]; }
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
   * Given a size-2 array of size_t, representing the shape, determine
   * the maximum size of YaleStorage arrays.
   */
  static size_t max_size(const size_t* shape) {
    size_t result = shape[0] * shape[1] + 1;
    if (shape[0] > shape[1])
      result += shape[0] - shape[1];
    return result;
  }

  /*
   * Minimum size of Yale Storage arrays given some shape.
   */
  static size_t min_size(const size_t* shape) {
    return shape[0]*2 + 1;
  }

  /*
   * This is the guaranteed maximum size of the IJA/A arrays of the matrix given its shape.
   */
  inline size_t real_max_size() const {
    return YaleStorage<D>::max_size(real_shape_p());
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
    std::cerr << "rflbp: left=" << left << "\tright=" << right << "\treal_j=" << real_j << std::endl;
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
    std::cerr << "fpfi: i=" << i << "\tleft=" << left << "\tright=" << right << "\treal_j=" << j + offset(1) << std::endl;

    // Check that the right search point is valid. rflbp will check to make sure the left is valid relative to left.
    if (right > ija(real_shape(0))) {
      std::cerr << "\tright now set to real_shape(0) = " << real_shape(0) << std::endl;
      right = ija(real_shape(0));
    }
    size_t result = real_find_left_boundary_pos(left, right, j + offset(1));
    std::cerr << "\t" << result << std::endl;
    return result;
  }

  typedef yale_storage::basic_iterator_T<D,D,YaleStorage<D> >              basic_iterator;
  typedef yale_storage::basic_iterator_T<D,const D,const YaleStorage<D> >  const_basic_iterator;

  typedef yale_storage::stored_diagonal_iterator_T<D,D,YaleStorage<D> >              stored_diagonal_iterator;
  typedef yale_storage::stored_diagonal_iterator_T<D,const D,const YaleStorage<D> >  const_stored_diagonal_iterator;

  typedef yale_storage::stored_nondiagonal_iterator_T<D,const D,const YaleStorage<D> >   const_stored_nondiagonal_iterator;
  typedef yale_storage::stored_nondiagonal_iterator_T<D,D,YaleStorage<D>>                stored_nondiagonal_iterator;

  typedef yale_storage::stored_iterator_T<D,D,YaleStorage<D> >             stored_iterator;
  typedef yale_storage::stored_iterator_T<D,const D,const YaleStorage<D> > const_stored_iterator;

  typedef yale_storage::iterator_T<D,D,YaleStorage<D> >                iterator;
  typedef yale_storage::iterator_T<D,const D,const YaleStorage<D> >    const_iterator;

  typedef yale_storage::ordered_iterator_T<D,D,YaleStorage<D> >              ordered_iterator;
  typedef yale_storage::ordered_iterator_T<D,const D,const YaleStorage<D> >  const_ordered_iterator;


  // Variety of iterator begin and end functions.
  iterator begin(size_t row = 0)                      {      return iterator(*this, row);               }
  iterator row_end(size_t row)                        {      return begin(row+1);                      }
  iterator end()                                      {      return iterator(*this, shape(0));          }
  const_iterator cbegin(size_t row = 0) const         {      return const_iterator(*this, row);         }
  const_iterator crow_end(size_t row) const           {      return cbegin(row+1);                     }
  const_iterator cend() const                         {      return const_iterator(*this, shape(0));    }

  stored_diagonal_iterator sdbegin(size_t d = 0)      {      return stored_diagonal_iterator(*this, d); }
  stored_diagonal_iterator sdend()                    {
    return stored_diagonal_iterator(*this, std::min( shape(0) + offset(0), shape(1) + offset(1) ) - std::max(offset(0), offset(1)) );
  }
  const_stored_diagonal_iterator csdbegin(size_t d = 0) const { return const_stored_diagonal_iterator(*this, d); }
  const_stored_diagonal_iterator csdend() const        {
    return const_stored_diagonal_iterator(*this, std::min( shape(0) + offset(0), shape(1) + offset(1) ) - std::max(offset(0), offset(1)) );
  }

  stored_nondiagonal_iterator sndbegin(size_t row = 0){      return stored_nondiagonal_iterator(*this, row); }
  stored_nondiagonal_iterator sndrow_end(size_t row)  {      return sndbegin(row+1);                   }
  stored_nondiagonal_iterator sndend()                {      return stored_nondiagonal_iterator(*this, shape(0)); }
  const_stored_nondiagonal_iterator csndbegin(size_t row = 0) const { return const_stored_nondiagonal_iterator(*this, row); }
  const_stored_nondiagonal_iterator csndrow_end(size_t row) const {   return csndbegin(row+1);                   }
  const_stored_nondiagonal_iterator csndend() const               {   return const_stored_nondiagonal_iterator(*this, shape(0)); }

  stored_iterator sbegin()                            {      return stored_iterator(*this, 0);       }
  stored_iterator send()                              {      return stored_iterator(*this, shape(0));      }
  const_stored_iterator csbegin() const               {      return const_stored_iterator(*this, 0);       }
  const_stored_iterator csend() const                 {      return const_stored_iterator(*this, shape(0));      }

  ordered_iterator obegin(size_t row = 0)             {      return ordered_iterator(*this, row);       }
  ordered_iterator oend()                             {      return ordered_iterator(*this, shape(0));  }
  ordered_iterator orow_end(size_t row)               {      return obegin(row+1);                     }
  const_ordered_iterator cobegin(size_t row = 0) const{      return const_ordered_iterator(*this, row); }
  const_ordered_iterator corow_end(size_t row) const  {      return cobegin(row+1);                    }
  const_ordered_iterator coend() const                {      return const_ordered_iterator(*this, shape(0)); }

  /*
   * Get a count of the ndnz in the slice as if it were its own matrix.
   */
  size_t count_copy_ndnz() const {
    if (!slice) return s->ndnz; // easy way -- not a slice.
    size_t count = 0;
    for (const_stored_nondiagonal_iterator iter = csndbegin(); iter != csndend(); ++iter) {
      if (iter.i() != iter.j() && *iter == const_default_obj()) ++count;
    }
    return count;
  }

  /*
   * Returns the iterator for i,j or snd_end() if not found.
   */
  stored_nondiagonal_iterator find(const std::pair<size_t,size_t>& ij) {
    std::pair<size_t,bool> find_pos_result = find_pos(ij);
    if (!find_pos_result.second) return sndend();
    else return stored_nondiagonal_iterator(*this, ij.first, find_pos_result.first);
  }

  /*
   * Returns a stored_nondiagonal_iterator pointing to the location where some coords i,j should go, or returns their
   * location if present.
   */
  stored_nondiagonal_iterator lower_bound(const std::pair<size_t,size_t>& ij) {
    return stored_nondiagonal_iterator(*this, ij.first, find_pos_for_insertion(ij.first, ij.second));
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


  /*
   * Returns a pointer to the location of some entry in the matrix.
   *
   * This is needed for backwards compatibility. We don't really want anyone
   * to modify the contents of that pointer, because it might be the ZERO location.
   *
   * TODO: Change all storage_get functions to return a VALUE once we've put list and
   * dense in OO mode. ???
   */
  inline D* get_single_p(SLICE* slice) {
    size_t real_i = offset(0) + slice->coords[0],
           real_j = offset(1) + slice->coords[1];

    if (real_i == real_j)
      return &(a(real_i));

    if (ija(real_i) == ija(real_i+1))
      return default_obj_ptr(); // zero pointer

    // binary search for a column's location
    std::pair<size_t,bool> p = find_pos(std::make_pair(slice->coords[0], slice->coords[1]));
    if (p.second)
      return &(a(p.first));
                       // not found: return default
    return default_obj_ptr(); // zero pointer
  }


  /*
   * Allocate a reference pointing to s. Note that even if +this+ is a reference,
   * we can create a reference within it.
   *
   * Note: Make sure you xfree() the result of this call. You can't just cast it
   * directly into a YaleStorage<D> class.
   */
  YALE_STORAGE* alloc_ref(SLICE* slice) {
    YALE_STORAGE* ns  = ALLOC( YALE_STORAGE );

    ns->dim           = s->dim;
    ns->offset        = ALLOC_N(size_t, ns->dim);
    ns->shape         = ALLOC_N(size_t, ns->dim);

    for (size_t d = 0; d < ns->dim; ++d) {
      ns->offset[d]   = slice->coords[d]  + offset(d);
      ns->shape[d]    = slice->lengths[d];
    }

    ns->dtype         = s->dtype;
    ns->a             = a_p();
    ns->ija           = ija_p();

    ns->src           = s;
    s->count++;

    ns->ndnz          = 0;
    ns->capacity      = 0;

    return ns;
  }


  /*
   * Allocates and initializes the basic struct (but not IJA or A vectors).
   */
  static YALE_STORAGE* alloc(size_t* shape, size_t dim = 2) {
    YALE_STORAGE* s = ALLOC( YALE_STORAGE );

    s->ndnz         = 0;
    s->dtype        = dtype();
    s->shape        = shape;
    s->offset       = ALLOC_N(size_t, dim);
    for (size_t d = 0; d < dim; ++d)
      s->offset[d]  = 0;
    s->dim          = dim;
    s->src          = reinterpret_cast<STORAGE*>(s);
    s->count        = 1;

    return s;
  }


  static YALE_STORAGE* create(size_t* shape, size_t dim, size_t reserve) {
    if (dim != 2) {
      rb_raise(rb_eNotImpError, "yale can only support 2D matrices");
    }

    YALE_STORAGE* s = alloc( shape, dim );
    size_t max_sz   = YaleStorage<D>::max_size(shape),
           min_sz   = YaleStorage<D>::min_size(shape);

    if (reserve < min_sz) {
      s->capacity = min_sz;
    } else if (reserve > max_sz) {
      s->capacity = max_sz;
    } else {
      s->capacity = reserve;
    }

    s->ija = ALLOC_N( size_t, s->capacity );
    s->a   = ALLOC_N( D,      s->capacity );

    return s;
  }


  /*
   * Clear out the D portion of the A vector (clearing the diagonal and setting
   * the zero value).
   */
  static void clear_diagonal_and_zero(YALE_STORAGE& s, D* init_val = NULL) {
    D* a  = reinterpret_cast<D*>(s.a);

    // Clear out the diagonal + one extra entry
    if (init_val) {
      for (size_t i = 0; i <= s.shape[0]; ++i)
        a[i] = *init_val;
    } else {
      for (size_t i = 0; i <= s.shape[0]; ++i)
        a[i] = 0;
    }
  }


  /*
   * Empty the matrix by initializing the IJA vector and setting the diagonal to 0.
   *
   * Called when most YALE_STORAGE objects are created.
   *
   * Can't go inside of class YaleStorage because YaleStorage creation requires that
   * IJA already be initialized.
   */
  static void init(YALE_STORAGE& s, D* init_val) {
    size_t IA_INIT = s.shape[0] + 1;
    for (size_t m = 0; m < IA_INIT; ++m) {
      s.ija[m] = IA_INIT;
    }

    clear_diagonal_and_zero(s, init_val);
  }


  /*
   * Make a full matrix structure copy (entries remain uninitialized). Remember to xfree()!
   */
  template <typename E>
  YALE_STORAGE* alloc_struct_copy(size_t new_capacity) const {
    nm::dtype_t new_dtype = nm::ctype_to_dtype_enum<E>::value_type;
    YALE_STORAGE* lhs     = ALLOC( YALE_STORAGE );
    lhs->dim              = s->dim;
    lhs->shape            = ALLOC_N( size_t, lhs->dim );
    lhs->offset           = ALLOC_N( size_t, lhs->dim );
    memcpy(lhs->shape, shape_p(), lhs->dim * sizeof(size_t));
    lhs->offset[0]        = 0;
    lhs->offset[1]        = 0;

    lhs->capacity         = new_capacity;
    lhs->dtype            = new_dtype;
    lhs->ndnz             = count_copy_ndnz();
    lhs->ija              = ALLOC_N( size_t, new_capacity );
    lhs->a                = ALLOC_N( E,      new_capacity );
    lhs->src              = lhs;
    lhs->count            = 1;

    // Now copy the IJA contents
    if (slice) {
      rb_raise(rb_eNotImpError, "cannot copy struct due to different offsets");
    } else {
      for (size_t m = 0; m < size(); ++m) {
        lhs->ija[m] = ija(m); // copy indices
      }
    }
    return lhs;
  }


  /*
   * Copy this slice (or the full matrix if it isn't a slice) into a new matrix which is already allocated, ns.
   */
  template <typename E>
  void copy(YALE_STORAGE& ns) const {
    nm::dtype_t new_dtype = nm::ctype_to_dtype_enum<E>::value_type;
    // get the default value for initialization (we'll re-use val for other copies after this)
    E val = static_cast<E>(default_obj());

    // initialize the matrix structure and clear the diagonal so we don't have to
    // keep track of unwritten entries.
    YaleStorage<E>::init(ns, &val);

    E* ns_a    = reinterpret_cast<E*>(ns.a);
    size_t sz  = shape(0) + 1; // current used size of ns

    // FIXME: Set this up so if THIS and NS have a shared diagonal, we use stored_diagonal and stored_nondiagonal instead. Faster.
    for (size_t i = 0; i < shape(0); ++i) {
      std::cerr << "copy row i = " << i << std::endl;

      const_ordered_iterator iter = cobegin(i);
      const_ordered_iterator next = cobegin(i+1);
      std::cerr << std::endl;
      std::cerr << "\titer.i,j=" << iter.i() << "," << iter.j() << "\tdenseloc=" << iter.dense_location() << std::endl;
      std::cerr << "\tnext.i,j=" << next.i() << "," << next.j() << "\tdenseloc=" << next.dense_location() << std::endl;
      std::cerr << std::boolalpha << "\titer < next ? " << (iter < next) << std::endl;

      for (const_ordered_iterator iter = cobegin(i); iter.i() == i && iter < next; ++iter) {
        std::cerr << "\t****\t\t\tcopy: iter != next_iter" << std::endl;
        std::cerr  << "\t\t\t\t\t\ti,j=" << iter.i() << "," << iter.j();
        std::cerr << "   i,j=" << next.i() << "," << next.j() << std::endl;

        if (i == iter.j()) {  // set diagonal in ns
          std::cerr << "\t\t****\t\t\tcopy: diag i=" << i << std::endl;
          ns_a[i]       = static_cast<E>(*iter);
        } else {
          std::cerr << "\t\t****\t\t\tcopy: nd   i=" << i << ", j=" << iter.j() << std::endl;
          ns.ija[sz]    = iter.j();
          ns_a[sz]      = static_cast<E>(*iter);

          ++sz;
        }
      }

      ns.ija[i+1] = sz;
    }

    //ns.ija[shape(0)] = sz;                // indicate end of last row
    ns.ndnz          = sz - shape(0) - 1; // update ndnz count
  }


  /*
   * Allocate a casted copy of this matrix/reference. Remember to xfree() the result!
   */
  template <typename E>
  YALE_STORAGE* alloc_copy() const {
    nm::dtype_t new_dtype = nm::ctype_to_dtype_enum<E>::value_type;

    YALE_STORAGE* lhs;
    if (slice) {
      size_t* shape     = ALLOC_N(size_t, 2);
      shape[0]          = this->shape(0);
      shape[1]          = this->shape(1);
      size_t ndnz       = count_copy_ndnz();
      size_t reserve    = this->shape(0) + ndnz + 1;

      std::cerr << "reserve = " << reserve << std::endl;

      lhs               = YaleStorage<E>::create(shape, 2, reserve);

      if (lhs->capacity < reserve)
        rb_raise(nm_eStorageTypeError, "conversion failed; capacity of %ld requested, max allowable is %ld", reserve, lhs->capacity);

      // Fill lhs with what's in our current matrix.
      copy<E>(*lhs);
    } else {
      // Copy the structure and setup the IJA structure.
      lhs               = alloc_struct_copy<E>(s->capacity);

      E* la = reinterpret_cast<E*>(lhs->a);
      for (size_t m = 0; m < size(); ++m) {
        la[m] = static_cast<E>(a(m));
      }

    }

    return lhs;

  }

  template <typename E>
  bool operator==(const YaleStorage<E>& rhs) const {

    typename YaleStorage<D>::const_ordered_iterator l = cobegin();
    typename YaleStorage<E>::const_ordered_iterator r = rhs.cobegin();

    while (l != coend() || r != rhs.coend()) {
      std::cerr << "looping:\t";
      if (l != coend()) std::cerr << "l != coend() (l ij=" << l.i() << "," << l.j() << ")\t";
      if (r != rhs.coend()) std::cerr << "r != coend() (r ij=" << r.i() << "," << r.j() << ")\t";
      std::cerr << "coend() ij=" << coend().i() << "," << coend().j() << std::endl;

      if (r == rhs.coend() || (l != coend() && r > l)) {
        if (*l != rhs.const_default_obj()) return false;
        ++l;
      } else if (l == coend() || (r != rhs.coend() && l > r)) {
        if (const_default_obj() != *r) return false;
        ++r;
      } else {
        if (*l != *r) return false;
        ++l;
        ++r;
      }
    }

    return true;
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

    size_t* new_ija     = ALLOC_N( size_t,new_cap );
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

} // end of nm namespace

#endif // YALE_CLASS_H