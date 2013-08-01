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
// == list.c
//
// List-of-lists n-dimensional matrix storage. Uses singly-linked
// lists.

/*
 * Standard Includes
 */

#include <ruby.h>
#include <algorithm> // std::min
#include <iostream>

/*
 * Project Includes
 */

#include "types.h"

#include "data/data.h"

#include "common.h"
#include "list.h"

#include "util/math.h"
#include "util/sl_list.h"

/*
 * Macros
 */

/*
 * Global Variables
 */

namespace nm { namespace list_storage {

/*
 * Forward Declarations
 */

template <typename LDType, typename RDType>
static LIST_STORAGE* cast_copy(const LIST_STORAGE* rhs, dtype_t new_dtype);

template <typename LDType, typename RDType>
static bool eqeq_r(const LIST_STORAGE* left, const size_t* l_offsets, const LIST_STORAGE* right, const size_t* r_offsets, const LIST* l, const LIST* r, size_t recursions, const void* l_init_, const void* r_init_);

template <typename SDType, typename TDType>
static bool eqeq_empty_r(const LIST_STORAGE* s, const size_t* offsets, const LIST* l, int recursions, const void* t_init);

} // end of namespace list_storage

extern "C" {

/*
 * Functions
 */


////////////////
// Lifecycle //
///////////////

/*
 * Creates a list-of-lists(-of-lists-of-lists-etc) storage framework for a
 * matrix.
 *
 * Note: The pointers you pass in for shape and init_val become property of our
 * new storage. You don't need to free them, and you shouldn't re-use them.
 */
LIST_STORAGE* nm_list_storage_create(dtype_t dtype, size_t* shape, size_t dim, void* init_val) {
  LIST_STORAGE* s = ALLOC( LIST_STORAGE );

  s->dim   = dim;
  s->shape = shape;
  s->dtype = dtype;

  s->offset = ALLOC_N(size_t, s->dim);
  memset(s->offset, 0, s->dim * sizeof(size_t));

  s->rows  = list::create();
  s->default_val = init_val;
  s->count = 1;
  s->src = s;

  return s;
}

/*
 * Documentation goes here.
 */
void nm_list_storage_delete(STORAGE* s) {
  if (s) {
    LIST_STORAGE* storage = (LIST_STORAGE*)s;
    if (storage->count-- == 1) {
      list::del( storage->rows, storage->dim - 1 );

      free(storage->shape);
      free(storage->offset);
      free(storage->default_val);
      free(s);
    }
  }
}

/*
 * Documentation goes here.
 */
void nm_list_storage_delete_ref(STORAGE* s) {
  if (s) {
    LIST_STORAGE* storage = (LIST_STORAGE*)s;

    nm_list_storage_delete( reinterpret_cast<STORAGE*>(storage->src ) );
    free(storage->shape);
    free(storage->offset);
    free(s);
  }
}

/*
 * Documentation goes here.
 */
void nm_list_storage_mark(void* storage_base) {
  LIST_STORAGE* storage = (LIST_STORAGE*)storage_base;

  if (storage && storage->dtype == RUBYOBJ) {
    rb_gc_mark(*((VALUE*)(storage->default_val)));
    list::mark(storage->rows, storage->dim - 1);
  }
}

///////////////
// Accessors //
///////////////

/*
 * Documentation goes here.
 */
NODE* list_storage_get_single_node(LIST_STORAGE* s, SLICE* slice)
{
  size_t r;
  LIST*  l = s->rows;
  NODE*  n;

  for (r = 0; r < s->dim; r++) {
    n = list::find(l, s->offset[r] + slice->coords[r]);
    if (n)  l = reinterpret_cast<LIST*>(n->val);
    else return NULL;
  }

  return n;
}


/*
 * Recursive helper function for each_with_indices, based on nm_list_storage_count_elements_r.
 * Handles empty/non-existent sublists.
 */
static void each_empty_with_indices_r(LIST_STORAGE* s, size_t recursions, VALUE& stack) {
  size_t dim   = s->dim;
  long   max   = s->shape[dim-recursions-1];
  VALUE empty  = s->dtype == nm::RUBYOBJ ? *reinterpret_cast<VALUE*>(s->default_val) : rubyobj_from_cval(s->default_val, s->dtype).rval;

  if (recursions) {
    for (long index = 0; index < max; ++index) {
      // Don't do an unshift/shift here -- we'll let that be handled in the lowest-level iteration (recursions == 0)
      rb_ary_push(stack, LONG2NUM(index));
      each_empty_with_indices_r(s, recursions-1, stack);
      rb_ary_pop(stack);
    }
  } else {
    rb_ary_unshift(stack, empty);
    for (long index = 0; index < max; ++index) {
      rb_ary_push(stack, LONG2NUM(index));
      rb_yield_splat(stack);
      rb_ary_pop(stack);
    }
    rb_ary_shift(stack);
  }
}

/*
 * Recursive helper function for each_with_indices, based on nm_list_storage_count_elements_r.
 */
static void each_with_indices_r(LIST_STORAGE* s, const LIST* l, size_t recursions, VALUE& stack) {
  NODE*  curr  = l->first;
  size_t dim   = s->dim;
  long   max   = s->shape[dim-recursions-1];

  if (recursions) {
    for (long index = 0; index < max; ++index) {
      rb_ary_push(stack, LONG2NUM(index));
      if (!curr || index < curr->key) {
        each_empty_with_indices_r(s, recursions-1, stack);
      } else {
        each_with_indices_r(s, reinterpret_cast<const LIST*>(curr->val), recursions-1, stack);
        curr = curr->next;
      }
      rb_ary_pop(stack);
    }
  } else {
    for (long index = 0; index < max; ++index) {

      rb_ary_push(stack, LONG2NUM(index));

      if (!curr || index < curr->key) {
        if (s->dtype == nm::RUBYOBJ)
          rb_ary_unshift(stack, *reinterpret_cast<VALUE*>(s->default_val));
        else
          rb_ary_unshift(stack, rubyobj_from_cval(s->default_val, s->dtype).rval);

      } else { // index == curr->key
        if (s->dtype == nm::RUBYOBJ)
          rb_ary_unshift(stack, *reinterpret_cast<VALUE*>(curr->val));
        else
          rb_ary_unshift(stack, rubyobj_from_cval(curr->val, s->dtype).rval);

        curr = curr->next;
      }
      rb_yield_splat(stack);

      rb_ary_shift(stack);
      rb_ary_pop(stack);
    }
  }

}


/*
 * Recursive helper function for each_stored_with_indices, based on nm_list_storage_count_elements_r.
 */
static void each_stored_with_indices_r(LIST_STORAGE* s, const LIST* l, size_t recursions, VALUE& stack) {
  NODE* curr = l->first;

  if (recursions) {
    while (curr) {
      rb_ary_push(stack, LONG2NUM(static_cast<long>(curr->key)));
      each_stored_with_indices_r(s, reinterpret_cast<const LIST*>(curr->val), recursions-1, stack);
      rb_ary_pop(stack);
      curr = curr->next;
    }
  } else {
    while (curr) {
      rb_ary_push(stack, LONG2NUM(static_cast<long>(curr->key))); // add index to end

      // add value to beginning
      if (s->dtype == nm::RUBYOBJ) {
        rb_ary_unshift(stack, *reinterpret_cast<VALUE*>(curr->val));
      } else {
        rb_ary_unshift(stack, rubyobj_from_cval(curr->val, s->dtype).rval);
      }
      // yield to the whole stack (value, i, j, k, ...)
      rb_yield_splat(stack);

      // remove the value
      rb_ary_shift(stack);

      // remove the index from the end
      rb_ary_pop(stack);

      curr = curr->next;
    }
  }
}


/*
 * Recursive helper for map_merged_stored_r which handles the case where one list is empty and the other is not.
 */
static void map_empty_stored_r(LIST_STORAGE* result, const LIST_STORAGE* s, const size_t* offsets, LIST* x, const LIST* l, size_t recursions, bool rev, const VALUE& t_init, const VALUE& init) {
  NODE *curr  = l->first,
       *xcurr = NULL;

  // For reference matrices, make sure we start in the correct place.
  size_t offset = offsets[s->dim - recursions - 1];
  while (curr && curr->key < offset) {  curr = curr->next;  }

  if (recursions) {
    while (curr) {
      LIST* val = nm::list::create();
      map_empty_stored_r(result, s, offsets, val, reinterpret_cast<const LIST*>(curr->val), recursions-1, rev, t_init, init);

      if (!val->first) nm::list::del(val, 0);
      else nm::list::insert_helper(x, xcurr, curr->key - offset, val);

      curr = curr->next;
    }
  } else {
    while (curr) {
      VALUE val, s_val = rubyobj_from_cval(curr->val, s->dtype).rval;
      if (rev) val = rb_yield_values(2, t_init, s_val);
      else     val = rb_yield_values(2, s_val, t_init);

      if (rb_funcall(val, rb_intern("!="), 1, init) == Qtrue)
        xcurr = nm::list::insert_helper(x, xcurr, curr->key - offset, val);

      curr = curr->next;
    }
  }

}


/*
 * Recursive helper function for nm_list_map_merged_stored
 */
static void map_merged_stored_r(LIST_STORAGE* result, const LIST_STORAGE* left, const size_t* l_offsets, const LIST_STORAGE* right, const size_t* r_offsets, LIST* x, const LIST* l, const LIST* r, size_t recursions, const VALUE& l_init, const VALUE& r_init, const VALUE& init) {
  NODE *lcurr = l->first,
       *rcurr = r->first,
       *xcurr = x->first;

  size_t l_offset = l_offsets[left->dim - recursions - 1];
  size_t r_offset = r_offsets[right->dim - recursions - 1];

  // For reference matrices, make sure we start in the correct place.
  while (lcurr && lcurr->key < l_offset) {  lcurr = lcurr->next;  }
  while (rcurr && rcurr->key < r_offset) {  rcurr = rcurr->next;  }

  if (recursions) {
    while (lcurr || rcurr) {
      size_t key;
      LIST*  val = nm::list::create();

      if (!rcurr || (lcurr && (lcurr->key - l_offset < rcurr->key - r_offset))) {
        map_empty_stored_r(result, left, l_offsets, val, reinterpret_cast<const LIST*>(lcurr->val), recursions-1, false, r_init, init);
        key   = lcurr->key - l_offset;
        lcurr = lcurr->next;
      } else if (!lcurr || (rcurr && (rcurr->key - r_offset < lcurr->key - l_offset))) {
        map_empty_stored_r(result, right, r_offsets, val, reinterpret_cast<const LIST*>(rcurr->val), recursions-1, true, l_init, init);
        key   = rcurr->key - r_offset;
        rcurr = rcurr->next;
      } else { // == and both present
        map_merged_stored_r(result, left, l_offsets, right, r_offsets, val, reinterpret_cast<const LIST*>(lcurr->val), reinterpret_cast<const LIST*>(rcurr->val), recursions-1, l_init, r_init, init);
        key   = lcurr->key - l_offset;
        lcurr = lcurr->next;
        rcurr = rcurr->next;
      }

      if (!val->first) nm::list::del(val, 0); // empty list -- don't insert
      else xcurr = nm::list::insert_helper(x, xcurr, key, val);

    }
  } else {
    while (lcurr || rcurr) {
      size_t key;
      VALUE  val;

      if (!rcurr || (lcurr && (lcurr->key - l_offset < rcurr->key - r_offset))) {
        val   = rb_yield_values(2, rubyobj_from_cval(lcurr->val, left->dtype).rval, r_init);
        key   = lcurr->key - l_offset;
        lcurr = lcurr->next;
      } else if (!lcurr || (rcurr && (rcurr->key - r_offset < lcurr->key - l_offset))) {
        val   = rb_yield_values(2, l_init, rubyobj_from_cval(rcurr->val, right->dtype).rval);
        key   = rcurr->key - r_offset;
        rcurr = rcurr->next;
      } else { // == and both present
        val   = rb_yield_values(2, rubyobj_from_cval(lcurr->val, left->dtype).rval, rubyobj_from_cval(rcurr->val, right->dtype).rval);
        key   = lcurr->key - l_offset;
        lcurr = lcurr->next;
        rcurr = rcurr->next;
      }
      if (rb_funcall(val, rb_intern("!="), 1, init) == Qtrue)
        xcurr = nm::list::insert_helper(x, xcurr, key, val);

    }
  }
}



/*
 * Each/each-stored iterator, brings along the indices.
 */
VALUE nm_list_each_with_indices(VALUE nmatrix, bool stored) {

  // If we don't have a block, return an enumerator.
  RETURN_SIZED_ENUMERATOR(nmatrix, 0, 0, 0);

  LIST_STORAGE* s = NM_STORAGE_LIST(nmatrix);

  VALUE stack = rb_ary_new();

  if (stored) each_stored_with_indices_r(s, s->rows, s->dim - 1, stack);
  else        each_with_indices_r(s, s->rows, s->dim - 1, stack);

  return nmatrix;
}


/*
 * map merged stored iterator. Always returns a matrix containing RubyObjects which probably needs to be casted.
 */
VALUE nm_list_map_merged_stored(int argc, VALUE* argv, VALUE left) {

  VALUE right, init;
  bool scalar = false;

  rb_scan_args(argc, argv, "11", &right, &init);

  LIST_STORAGE *s   = NM_STORAGE_LIST(left),
               *t;
  size_t *s_offsets = s->offset,
         *t_offsets;

  // For each matrix, if it's a reference, we want to deal directly with the original (now that we have the offsets)
  if (s->src != s) s = reinterpret_cast<LIST_STORAGE*>(s->src);

  // right might be a scalar, in which case this is a scalar operation.
  if (TYPE(right) != T_DATA || (RDATA(right)->dfree != (RUBY_DATA_FUNC)nm_delete && RDATA(right)->dfree != (RUBY_DATA_FUNC)nm_delete_ref)) {
    nm::dtype_t r_dtype = nm_dtype_guess(right);

    size_t* shape       = ALLOC_N(size_t, s->dim);
    memcpy(shape, s->shape, s->dim);
    void *scalar_init   = rubyobj_to_cval(right, r_dtype); // make a copy of right

    t                   = reinterpret_cast<LIST_STORAGE*>(nm_list_storage_create(r_dtype, shape, s->dim, scalar_init));
    t_offsets           = t->offset;
    scalar              = true;
  } else {
    t                   = NM_STORAGE_LIST(right); // element-wise, not scalar.
    t_offsets           = t->offset;
    if (t->src != t) t  = reinterpret_cast<LIST_STORAGE*>(t->src);
  }

  //if (!rb_block_given_p()) {
  //  rb_raise(rb_eNotImpError, "RETURN_SIZED_ENUMERATOR probably won't work for a map_merged since no merged object is created");
  //}
  // If we don't have a block, return an enumerator.
  RETURN_SIZED_ENUMERATOR(left, 0, 0, 0); // FIXME: Test this. Probably won't work. Enable above code instead.

  // Figure out a default value if one wasn't provided by the user.
  VALUE s_init = rubyobj_from_cval(s->default_val, s->dtype).rval,
        t_init = rubyobj_from_cval(t->default_val, t->dtype).rval;
  if (init == Qnil) init = rb_yield_values(2, s_init, t_init);

	// Allocate a new shape array for the resulting matrix.
	size_t* shape = ALLOC_N(size_t, s->dim);
	memcpy(shape, s->shape, sizeof(size_t) * s->dim);
  void* init_val = ALLOC(VALUE);
  memcpy(init_val, &init, sizeof(VALUE));

  NMATRIX* result = nm_create(nm::LIST_STORE, nm_list_storage_create(nm::RUBYOBJ, shape, s->dim, init_val));
  LIST_STORAGE* r = reinterpret_cast<LIST_STORAGE*>(result->storage);

  map_merged_stored_r(r, s, s_offsets, t, t_offsets, r->rows, s->rows, t->rows, s->dim - 1, s_init, t_init, init);

  // If we are working with a scalar operation
  if (scalar) nm_list_storage_delete(t);

  return Data_Wrap_Struct(CLASS_OF(left), nm_list_storage_mark, nm_delete, result);
}



static LIST* slice_copy(const LIST_STORAGE *src, LIST *src_rows, size_t *coords, size_t *lengths, size_t n) {
  NODE *src_node;
  LIST *dst_rows = NULL;
  void *val = NULL;
  int key;
  
  dst_rows = list::create();
  src_node = src_rows->first;

  while (src_node) {
    key = src_node->key - (src->offset[n] + coords[n]);
    
    if (key >= 0 && (size_t)key < lengths[n]) {
      if (src->dim - n > 1) {
        val = slice_copy(src,  
          reinterpret_cast<LIST*>(src_node->val), 
          coords,
          lengths,
          n + 1);  

        if (val) 
          list::insert_with_copy(dst_rows, key, val, sizeof(LIST));          
        
      }
      else {
        list::insert_with_copy(dst_rows, key, src_node->val, DTYPE_SIZES[src->dtype]);
      }
    }

    src_node = src_node->next;
  }

  return dst_rows;
}

/*
 * Documentation goes here.
 */
void* nm_list_storage_get(STORAGE* storage, SLICE* slice) {
  LIST_STORAGE* s = (LIST_STORAGE*)storage;
  LIST_STORAGE* ns = NULL;
  NODE* n;

  if (slice->single) {
    n = list_storage_get_single_node(s, slice); 
    return (n ? n->val : s->default_val);
  } 
  else {
    void *init_val = ALLOC_N(char, DTYPE_SIZES[s->dtype]);
    memcpy(init_val, s->default_val, DTYPE_SIZES[s->dtype]);

    size_t *shape = ALLOC_N(size_t, s->dim);
    memcpy(shape, slice->lengths, sizeof(size_t) * s->dim);

    ns = nm_list_storage_create(s->dtype, shape, s->dim, init_val);
    
    ns->rows = slice_copy(s, s->rows, slice->coords, slice->lengths, 0);
    return ns;
  }
}

/*
 * Get the contents of some set of coordinates. Note: Does not make a copy!
 * Don't free!
 */
void* nm_list_storage_ref(STORAGE* storage, SLICE* slice) {
  LIST_STORAGE* s = (LIST_STORAGE*)storage;
  LIST_STORAGE* ns = NULL;
  NODE* n;

  //TODO: It needs a refactoring.
  if (slice->single) {
    n = list_storage_get_single_node(s, slice); 
    return (n ? n->val : s->default_val);
  } 
  else {
    ns = ALLOC( LIST_STORAGE );
    
    ns->dim = s->dim;
    ns->dtype = s->dtype;
    ns->offset     = ALLOC_N(size_t, ns->dim);
    ns->shape      = ALLOC_N(size_t, ns->dim);

    for (size_t i = 0; i < ns->dim; ++i) {
      ns->offset[i] = slice->coords[i] + s->offset[i];
      ns->shape[i]  = slice->lengths[i];
    }

    ns->rows = s->rows;
    ns->default_val = s->default_val;
    
    s->src->count++;
    ns->src = s->src;
    
    return ns;
  }
}

/*
 * Documentation goes here.
 *
 * TODO: Allow this function to accept an entire row and not just one value -- for slicing
 */
void* nm_list_storage_insert(STORAGE* storage, SLICE* slice, void* val) {
  LIST_STORAGE* s = (LIST_STORAGE*)storage;
  // Pretend dims = 2
  // Then coords is going to be size 2
  // So we need to find out if some key already exists
  size_t r;
  NODE*  n;
  LIST*  l = s->rows;

  // drill down into the structure
  for (r = s->dim; r > 1; --r) {
    n = list::insert(l, false, s->offset[s->dim - r] + slice->coords[s->dim - r], list::create());
    l = reinterpret_cast<LIST*>(n->val);
  }

  n = list::insert(l, true, s->offset[s->dim - r] + slice->coords[s->dim - r], val);
  return n->val;
}

/*
 * Remove an item from list storage.
 */
void* nm_list_storage_remove(STORAGE* storage, SLICE* slice) {
  LIST_STORAGE* s = (LIST_STORAGE*)storage;
  void* rm = NULL;

  // This returns a boolean, which will indicate whether s->rows is empty.
  // We can safely ignore it, since we never want to delete s->rows until
  // it's time to destroy the LIST_STORAGE object.
  list::remove_recursive(s->rows, slice->coords, s->offset, 0, s->dim, rm);

  return rm;
}

///////////
// Tests //
///////////

/*
 * Comparison of contents for list storage.
 */
bool nm_list_storage_eqeq(const STORAGE* left, const STORAGE* right) {
	NAMED_LR_DTYPE_TEMPLATE_TABLE(ttable, nm::list_storage::eqeq_r, bool, const LIST_STORAGE* left, const size_t* l_offsets, const LIST_STORAGE* right, const size_t* r_offsets, const LIST* l, const LIST* r, size_t recursions, const void* l_init_, const void* r_init_)

  const LIST_STORAGE* casted_left  = reinterpret_cast<const LIST_STORAGE*>(left);
  const size_t* l_offsets          = casted_left->offset;
  if (casted_left->src != casted_left)
    casted_left                    = reinterpret_cast<const LIST_STORAGE*>(casted_left->src);

  const LIST_STORAGE* casted_right = reinterpret_cast<const LIST_STORAGE*>(right);
  const size_t* r_offsets          = casted_right->offset;
  if (casted_right->src != casted_right)
    casted_right                   = reinterpret_cast<const LIST_STORAGE*>(casted_right->src);

	return ttable[left->dtype][right->dtype](casted_left, l_offsets, casted_right, r_offsets, casted_left->rows, casted_right->rows, casted_left->dim - 1, casted_left->default_val, casted_right->default_val);
}

//////////
// Math //
//////////


/*
 * List storage matrix multiplication.
 */
STORAGE* nm_list_storage_matrix_multiply(const STORAGE_PAIR& casted_storage, size_t* resulting_shape, bool vector) {
  free(resulting_shape);
  rb_raise(rb_eNotImpError, "multiplication not implemented for list-of-list matrices");
  return NULL;
  //DTYPE_TEMPLATE_TABLE(dense_storage::matrix_multiply, NMATRIX*, STORAGE_PAIR, size_t*, bool);

  //return ttable[reinterpret_cast<DENSE_STORAGE*>(casted_storage.left)->dtype](casted_storage, resulting_shape, vector);
}


/*
 * List storage to Hash conversion. Uses Hashes with default values, so you can continue to pretend
 * it's a sparse matrix.
 */
VALUE nm_list_storage_to_hash(const LIST_STORAGE* s, const dtype_t dtype) {

  // Get the default value for the list storage.
  VALUE default_value = rubyobj_from_cval(s->default_val, dtype).rval;

  // Recursively copy each dimension of the matrix into a nested hash.
  return nm_list_copy_to_hash(s->rows, dtype, s->dim - 1, default_value);
}

/////////////
// Utility //
/////////////

/*
 * Recursively count the non-zero elements in a list storage object.
 */
size_t nm_list_storage_count_elements_r(const LIST* l, size_t recursions) {
  size_t count = 0;
  NODE* curr = l->first;
  
  if (recursions) {
    while (curr) {
      count += nm_list_storage_count_elements_r(reinterpret_cast<const LIST*>(curr->val), recursions - 1);
      curr   = curr->next;
    }
    
  } else {
    while (curr) {
      ++count;
      curr = curr->next;
    }
  }
  
  return count;
}

/*
 * Count non-diagonal non-zero elements.
 */
size_t nm_list_storage_count_nd_elements(const LIST_STORAGE* s) {
  NODE *i_curr, *j_curr;
  size_t count = 0;
  
  if (s->dim != 2) {
  	rb_raise(rb_eNotImpError, "non-diagonal element counting only defined for dim = 2");
  }

  for (i_curr = s->rows->first; i_curr; i_curr = i_curr->next) {
    int i = i_curr->key - s->offset[0];
    if (i < 0 || i >= (int)s->shape[0]) continue;

    for (j_curr = ((LIST*)(i_curr->val))->first; j_curr; j_curr = j_curr->next) {
      int j = j_curr->key - s->offset[1];
      if (j < 0 || j >= (int)s->shape[1]) continue;

      if (i != j)  	++count;
    }
  }
  
  return count;
}

/////////////////////////
// Copying and Casting //
/////////////////////////
//
/*
 * List storage copy constructor C access.
 */

LIST_STORAGE* nm_list_storage_copy(const LIST_STORAGE* rhs)
{
  size_t *shape = ALLOC_N(size_t, rhs->dim);
  memcpy(shape, rhs->shape, sizeof(size_t) * rhs->dim);
  
  void *init_val = ALLOC_N(char, DTYPE_SIZES[rhs->dtype]);
  memcpy(init_val, rhs->default_val, DTYPE_SIZES[rhs->dtype]);

  LIST_STORAGE* lhs = nm_list_storage_create(rhs->dtype, shape, rhs->dim, init_val);
  
  lhs->rows = slice_copy(rhs, rhs->rows, lhs->offset, lhs->shape, 0);

  return lhs;
}

/*
 * List storage copy constructor C access with casting.
 */
STORAGE* nm_list_storage_cast_copy(const STORAGE* rhs, dtype_t new_dtype) {
  NAMED_LR_DTYPE_TEMPLATE_TABLE(ttable, nm::list_storage::cast_copy, LIST_STORAGE*, const LIST_STORAGE* rhs, dtype_t new_dtype);

  return (STORAGE*)ttable[new_dtype][rhs->dtype]((LIST_STORAGE*)rhs, new_dtype);
}


/*
 * List storage copy constructor for transposing.
 */
STORAGE* nm_list_storage_copy_transposed(const STORAGE* rhs_base) {
  rb_raise(rb_eNotImpError, "list storage transpose not yet implemented");
  return NULL;
}


} // end of extern "C" block


/////////////////////////
// Templated Functions //
/////////////////////////

namespace list_storage {

/*
 * List storage copy constructor for changing dtypes.
 */
template <typename LDType, typename RDType>
static LIST_STORAGE* cast_copy(const LIST_STORAGE* rhs, dtype_t new_dtype) {

  // allocate and copy shape
  size_t* shape = ALLOC_N(size_t, rhs->dim);
  memcpy(shape, rhs->shape, rhs->dim * sizeof(size_t));

  // copy default value
  LDType* default_val = ALLOC_N(LDType, 1);
  *default_val = *reinterpret_cast<RDType*>(rhs->default_val);

  LIST_STORAGE* lhs = nm_list_storage_create(new_dtype, shape, rhs->dim, default_val);
  //lhs->rows         = list::create();

  // TODO: Needs optimization. When matrix is reference it is copped twice.
  if (rhs->src == rhs) 
    list::cast_copy_contents<LDType, RDType>(lhs->rows, rhs->rows, rhs->dim - 1);
  else {
    LIST_STORAGE *tmp = nm_list_storage_copy(rhs);
    list::cast_copy_contents<LDType, RDType>(lhs->rows, tmp->rows, rhs->dim - 1);
    nm_list_storage_delete(tmp);
  }

  return lhs;
}


/*
 * Recursive helper function for eqeq. Note that we use SDType and TDType instead of L and R because this function
 * is a re-labeling. That is, it can be called in order L,R or order R,L; and we don't want to get confused. So we
 * use S and T to denote first and second passed in.
 */
template <typename SDType, typename TDType>
static bool eqeq_empty_r(const LIST_STORAGE* s, const size_t* offsets, const LIST* l, int recursions, const void* t_init) {
  NODE* curr  = l->first;

  // For reference matrices, make sure we start in the correct place.
  while (curr && curr->key < offsets[s->dim - recursions - 1]) {  curr = curr->next;  }

  if (recursions) {
    while (curr) {
      if (!eqeq_empty_r<SDType,TDType>(s, offsets, reinterpret_cast<const LIST*>(curr->val), recursions-1, t_init)) return false;
      curr = curr->next;
    }
  } else {
    while (curr) {
      if (*reinterpret_cast<SDType*>(curr->val) != *reinterpret_cast<const TDType*>(t_init)) return false;
      curr = curr->next;
    }
  }
  return true;
}


/*
 * Do these two list matrices of the same dtype have exactly the same contents (accounting for default_vals)?
 *
 * This function is recursive.
 */
template <typename LDType, typename RDType>
static bool eqeq_r(const LIST_STORAGE* left, const size_t* l_offsets, const LIST_STORAGE* right, const size_t* r_offsets, const LIST* l, const LIST* r, size_t recursions, const void* l_init_, const void* r_init_) {
  const LDType* l_init = reinterpret_cast<const LDType*>(l_init_);
  const RDType* r_init = reinterpret_cast<const RDType*>(r_init_);

  bool result;

  bool same_init = *l_init == *r_init;

  NODE *lcurr = l->first,
       *rcurr = r->first;

  size_t l_offset = l_offsets[left->dim - recursions - 1];
  size_t r_offset = r_offsets[right->dim - recursions - 1];

  // For reference matrices, make sure we start in the correct place.
  while (lcurr && lcurr->key < l_offset) {  lcurr = lcurr->next;  }
  while (rcurr && rcurr->key < r_offset) {  rcurr = rcurr->next;  }

  if (recursions) {
    while (lcurr || rcurr) {

      if (!rcurr || (lcurr && (lcurr->key - l_offset < rcurr->key - r_offset))) {
        if (!eqeq_empty_r<LDType,RDType>(left, l_offsets, reinterpret_cast<const LIST*>(lcurr->val), recursions-1, r_init_)) return false;
        lcurr   = lcurr->next;
      } else if (!lcurr || (rcurr && (rcurr->key - r_offset < lcurr->key - l_offset))) {
        if (!eqeq_empty_r<RDType,LDType>(right, r_offsets, reinterpret_cast<const LIST*>(rcurr->val), recursions-1, l_init_)) return false;
        rcurr   = rcurr->next;
      } else { // keys are == and both present
        if (!eqeq_r<LDType,RDType>(left, l_offsets, right, r_offsets, reinterpret_cast<const LIST*>(lcurr->val), reinterpret_cast<const LIST*>(rcurr->val), recursions-1, l_init_, r_init_)) return false;
        lcurr   = lcurr->next;
        rcurr   = rcurr->next;
      }
    }
  } else {
    while (lcurr || rcurr) {
      if (!rcurr || (lcurr && (lcurr->key - l_offset < rcurr->key - r_offset))) {
        if (*reinterpret_cast<LDType*>(lcurr->val) != *r_init) return false;
        lcurr         = lcurr->next;
      } else if (!lcurr || (rcurr && (rcurr->key - r_offset < lcurr->key - l_offset))) {
        if (*reinterpret_cast<RDType*>(rcurr->val) != *l_init) return false;
        rcurr         = rcurr->next;
      } else { // keys == and both left and right nodes present
        if (*reinterpret_cast<LDType*>(lcurr->val) != *reinterpret_cast<RDType*>(rcurr->val)) return false;
        lcurr         = lcurr->next;
        rcurr         = rcurr->next;
      }
    }
  }

  return true;
}


}} // end of namespace nm::list_storage

extern "C" {
  /*
   * call-seq:
   *     __list_to_hash__ -> Hash
   *
   * Create a Ruby Hash from a list NMatrix.
   *
   * This is an internal C function which handles list stype only.
   */
  VALUE nm_to_hash(VALUE self) {
    return nm_list_storage_to_hash(NM_STORAGE_LIST(self), NM_DTYPE(self));
  }

    /*
     * call-seq:
     *     __list_default_value__ -> ...
     *
     * Get the default_val property from a list matrix.
     */
    VALUE nm_list_default_value(VALUE self) {
      return rubyobj_from_cval(NM_DEFAULT_VAL(self), NM_DTYPE(self)).rval;
    }
} // end of extern "C" block
