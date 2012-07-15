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
// SciRuby is Copyright (c) 2010 - 2012, Ruby Science Foundation
// NMatrix is Copyright (c) 2012, Ruby Science Foundation
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
// == dense.c
//
// Dense n-dimensional matrix storage.

#ifndef DENSE_C
#define DENSE_C

#include <ruby.h>

#include "nmatrix.h"
extern bool (*ElemEqEq[NM_TYPES][2])(const void*, const void*, const int, const int);
extern const int nm_sizeof[NM_TYPES];

/* Calculate the number of elements in the dense storage structure, based on shape and rank */
size_t count_dense_storage_elements(const DENSE_STORAGE* s) {
  size_t i;
  size_t count = 1;
  for (i = 0; i < s->rank; ++i) count *= s->shape[i];
  return count;
}


// Is this dense matrix symmetric about the diagonal?
bool dense_is_symmetric(const DENSE_STORAGE* mat, int lda, bool hermitian) {
  unsigned int i, j;

  for (i = 0; i < mat->shape[0]; ++i) {
    for (j = i+1; j < mat->shape[1]; ++j) {
      if ( !(ElemEqEq[mat->dtype][(int8_t)(hermitian)]( (char*)(mat->elements) + (i*lda+j)*nm_sizeof[mat->dtype],
                                                       (char*)(mat->elements) + (j*lda+i)*nm_sizeof[mat->dtype],
                                                       1,
                                                       nm_sizeof[mat->dtype] )) )
        return false;
    }
  }
  return true;
}

/* Calculate element's number of dense matrix */
size_t dense_storage_pos(const DENSE_STORAGE* s, const size_t* coords) {
  size_t i;
  size_t pos = 0;

  for (i = 0; i < s->rank; i++)
    pos += (coords[i] + s->offset[i]) * (s->strides[i]);

  return pos;
}

/* Strides store offsets in straightforward array of element for each dimension. */
size_t* dense_calc_strides(size_t* shape, size_t rank) {
  size_t i, j;
  size_t* strides = calloc(sizeof(*shape), rank);

  if (!strides)
    rb_raise(rb_eNoMemError, "Memory error");

  for (i = 0; i < rank; i++) {
    strides[i] = 1;
    for (j = i + 1; j < rank; j++) {
      strides[i] *= shape[j];
    }
  }
  
  return strides;
}

/* The recursive slicing for N-dimension matrix */
void dense_slice_with_copping(DENSE_STORAGE *dest, DENSE_STORAGE *src,
    size_t* lens,
    size_t psrc, size_t pdest,
    size_t n) {

  size_t i;

  if (src->rank - n > 1) {
    for (i=0; i < lens[n]; i++) {
    dense_slice_with_copping(dest, src, lens,
        psrc + src->strides[n]*i, pdest + dest->strides[n]*i,
        n + 1);
    }
  }
  else {
    memcpy((char*)dest->elements + pdest*nm_sizeof[dest->dtype], 
        (char*)src->elements + psrc*nm_sizeof[src->dtype], 
        src->shape[n]*nm_sizeof[dest->dtype]);
  }

}

/* Get slice or one elements with copping */
void* dense_storage_get(DENSE_STORAGE* s, SLICE* slice) {
   DENSE_STORAGE *ns;
   size_t count;

  if (slice->is_one_el)
    return (char*)(s->elements) + dense_storage_pos(s, slice->coords) * nm_sizeof[s->dtype];
  else { // Make references  
    ns = ALLOC( DENSE_STORAGE );

    ns->rank       = s->rank;
    ns->shape      = slice->lens;
    ns->dtype      = s->dtype;
    ns->offset     = calloc(sizeof(size_t),ns->rank);
    ns->strides    = dense_calc_strides(ns->shape, ns->rank);
    ns->count      = 1;
    ns->src        = ns;

    count         = count_dense_storage_elements(s);
    ns->elements = ALLOC_N(char, nm_sizeof[ns->dtype]*count);

    dense_slice_with_copping(ns, s, slice->lens, dense_storage_pos(s, slice->coords), 0, 0);
    return ns;
  }
}

/* Get slice or one elements by refs*/
void* dense_storage_ref(DENSE_STORAGE* s, SLICE* slice) {
  DENSE_STORAGE *ns;

  if (slice->is_one_el)
    return (char*)(s->elements) + dense_storage_pos(s, slice->coords) * nm_sizeof[s->dtype];
  else { // Make references  
    ns = ALLOC( DENSE_STORAGE );

    ns->rank      = s->rank;
    ns->dtype     = s->dtype;

    ns->offset    = calloc(sizeof(*ns->offset), ns->rank);
    ns->shape     = calloc(sizeof(*ns->offset), ns->rank);
    
    memcpy(ns->offset, slice->coords, sizeof(*ns->offset)*ns->rank);
    memcpy(ns->shape, slice->lens, sizeof(*ns->shape)*ns->rank);
  
    ns->strides   = s->strides;
    ns->elements  = s->elements;
    
    s->count++;
    ns->src = (void*)s;

    return ns;
  }
}

/* Does not free passed-in value! Different from list_storage_insert. */
void dense_storage_set(DENSE_STORAGE* s, SLICE* slice, void* val) {
  memcpy((char*)(s->elements) + dense_storage_pos(s, slice->coords) * nm_sizeof[s->dtype], val, nm_sizeof[s->dtype]); 
}

bool dense_is_ref(const DENSE_STORAGE* s) {
  if (s->src == s)
    return false;

  return true;
}

DENSE_STORAGE* copy_dense_storage(const DENSE_STORAGE* rhs) {
  DENSE_STORAGE* lhs;
  size_t count = count_dense_storage_elements(rhs), p;
  size_t* shape = ALLOC_N(size_t, rhs->rank);
  if (!shape) return NULL;

  // copy shape array
  for (p = 0; p < rhs->rank; ++p)
    shape[p] = rhs->shape[p];

  lhs = create_dense_storage(rhs->dtype, shape, rhs->rank, NULL, 0);

  if (lhs && count) // ensure that allocation worked before copying
    if (dense_is_ref(rhs)) 
      dense_slice_with_copping(lhs, rhs->src, rhs->shape, 0, 0, 0); // slice all matrix
    else
      memcpy(lhs->elements, rhs->elements, nm_sizeof[rhs->dtype] * count); 

  return lhs;
}


DENSE_STORAGE* cast_copy_dense_storage(const DENSE_STORAGE* rhs, int8_t new_dtype) {
  DENSE_STORAGE *lhs, *tmp;
  size_t count, p;
  size_t* shape;


  if (new_dtype == rhs->dtype)
    return copy_dense_storage(rhs); 
  
  count = count_dense_storage_elements(rhs);
  shape = ALLOC_N(size_t, rhs->rank);
  if (!shape) return NULL;

  // copy shape array
  for (p = 0; p < rhs->rank; ++p) shape[p] = rhs->shape[p];

  lhs = create_dense_storage(new_dtype, shape, rhs->rank, NULL, 0);
  if (lhs && count) // ensure that allocation worked before copying
    if (dense_is_ref(rhs)) {
      tmp = copy_dense_storage(rhs);
      SetFuncs[lhs->dtype][tmp->dtype](count, lhs->elements, nm_sizeof[lhs->dtype], tmp->elements, nm_sizeof[tmp->dtype]);
      delete_dense_storage(tmp);
    }
    else
      SetFuncs[lhs->dtype][rhs->dtype](count, lhs->elements, nm_sizeof[lhs->dtype], rhs->elements, nm_sizeof[rhs->dtype]);


  return lhs;
}

// Do these two dense matrices of the same dtype have exactly the same contents?
bool dense_storage_eqeq(const DENSE_STORAGE* left, const DENSE_STORAGE* right) {
  DENSE_STORAGE *a, *b;

  /* FIXME: Very strange behavior! The GC calls directly the method with non-initialized data. */
  if (left->rank != right->rank)
    return false;

  a = (dense_is_ref(left) ? copy_dense_storage(left) : left); 
  b = (dense_is_ref(right) ? copy_dense_storage(right) : right); 


  return ElemEqEq[a->dtype][0](a->elements, b->elements, count_dense_storage_elements(a), nm_sizeof[b->dtype]);
}


// Copy a set of default values into dense
static inline void cast_copy_dense_list_default(void* lhs, void* default_val, int8_t l_dtype, int8_t r_dtype, size_t* pos, const size_t* shape, size_t rank, size_t max_elements, size_t recursions) {
  size_t i;

  for (i = 0; i < shape[rank-1-recursions]; ++i, ++(*pos)) {
    //fprintf(stderr, "default: pos = %u, dim = %u\t", *pos, shape[rank-1-recursions]);

    if (recursions == 0) { cast_copy_value_single((char*)lhs + (*pos)*nm_sizeof[l_dtype], default_val, l_dtype, r_dtype); fprintf(stderr, "zero\n"); }
    else                 { cast_copy_dense_list_default(lhs, default_val, l_dtype, r_dtype, pos, shape, rank, max_elements, recursions-1); fprintf(stderr, "column of zeros\n"); }
  }
  --(*pos);
}


// Copy list contents into dense recursively
static void cast_copy_dense_list_contents(void* lhs, const LIST* rhs, void* default_val, int8_t l_dtype, int8_t r_dtype, size_t* pos, const size_t* shape, size_t rank, size_t max_elements, size_t recursions) {
  NODE *curr = rhs->first;
  int last_key = -1;
  size_t i = 0;

  for (i = 0; i < shape[rank-1-recursions]; ++i, ++(*pos)) {

    if (!curr || (curr->key > (size_t)(last_key+1))) {
      //fprintf(stderr, "pos = %u, dim = %u, curr->key XX, last_key+1 = %d\t", *pos, shape[rank-1-recursions], last_key+1);
      if (recursions == 0) cast_copy_value_single((char*)lhs + (*pos)*nm_sizeof[l_dtype], default_val, l_dtype, r_dtype); //fprintf(stderr, "zero\n"); }
      else                 cast_copy_dense_list_default(lhs, default_val, l_dtype, r_dtype, pos, shape, rank, max_elements, recursions-1); //fprintf(stderr, "column of zeros\n"); }

      ++last_key;
    } else {
      //fprintf(stderr, "pos = %u, dim = %u, curr->key = %u, last_key+1 = %d\t", *pos, shape[rank-1-recursions], curr->key, last_key+1);
      if (recursions == 0) cast_copy_value_single((char*)lhs + (*pos)*nm_sizeof[l_dtype], curr->val, l_dtype, r_dtype); //fprintf(stderr, "value\n"); }
      else                 cast_copy_dense_list_contents(lhs, curr->val, default_val, l_dtype, r_dtype, pos, shape, rank, max_elements, recursions-1); //fprintf(stderr, "column of values\n"); }

      last_key = curr->key;
      curr     = curr->next;
    }
  }
  --(*pos);
}


// Convert (by creating a copy) from list storage to dense storage.
DENSE_STORAGE* scast_copy_dense_list(const LIST_STORAGE* rhs, int8_t l_dtype) {
  DENSE_STORAGE* lhs;
  size_t pos   = 0; // position in lhs->elements

  // allocate and copy shape
  size_t* shape = ALLOC_N(size_t, rhs->rank);
  memcpy(shape, rhs->shape, rhs->rank * sizeof(size_t));

  lhs = create_dense_storage(l_dtype, shape, rhs->rank, NULL, 0);

  // recursively copy the contents
  cast_copy_dense_list_contents(lhs->elements, rhs->rows, rhs->default_val, l_dtype, rhs->dtype, &pos, shape, lhs->rank, count_storage_max_elements((STORAGE*)rhs), rhs->rank-1);

  return lhs;
}


DENSE_STORAGE* scast_copy_dense_yale(const YALE_STORAGE* rhs, int8_t l_dtype) {
  DENSE_STORAGE* lhs;
  y_size_t i, j, // position in lhs->elements
           ija, ija_next, jj; // position in rhs->elements
  y_size_t pos = 0;          // position in dense to write to
  void* R_ZERO = (char*)(rhs->a) + rhs->shape[0] * nm_sizeof[rhs->dtype]; // determine zero representation

  // allocate and set shape
  size_t* shape = ALLOC_N(size_t, rhs->rank);
  memcpy(shape, rhs->shape, rhs->rank * sizeof(size_t));

  lhs = create_dense_storage(l_dtype, shape, rhs->rank, NULL, 0);

  // Walk through rows. For each entry we set in dense, increment pos.
  for (i = 0; i < rhs->shape[0]; ++i) {

    // get boundaries of this row, store in ija and ija_next
    YaleGetIJA(ija,      rhs, i);
    YaleGetIJA(ija_next, rhs, i+1);

    if (ija == ija_next) { // row is empty?

      for (j = 0; j < rhs->shape[1]; ++j) {  // write zeros in each column

        // Fill in zeros (except for diagonal)
        if (i == j) cast_copy_value_single((char*)(lhs->elements) + pos*nm_sizeof[l_dtype], (char*)(rhs->a) + i*nm_sizeof[rhs->dtype], l_dtype, rhs->dtype);
        else        cast_copy_value_single((char*)(lhs->elements) + pos*nm_sizeof[l_dtype], R_ZERO, l_dtype, rhs->dtype);

        ++pos; // move to next dense position
      }

    } else {
      // row contains entries: write those in each column, interspersed with zeros
      YaleGetIJA(jj, rhs, ija);

      for (j = 0; j < rhs->shape[1]; ++j) {
        if (i == j) {

          cast_copy_value_single((char*)(lhs->elements) + pos*nm_sizeof[l_dtype], (char*)(rhs->a) + i*nm_sizeof[rhs->dtype], l_dtype, rhs->dtype);

        } else if (j == jj) {

          // copy from rhs
          cast_copy_value_single((char*)(lhs->elements) + pos*nm_sizeof[l_dtype], (char*)(rhs->a) + ija*nm_sizeof[rhs->dtype], l_dtype, rhs->dtype);

          // get next
          ++ija;

          // increment to next column ID (or go off the end)
          if (ija < ija_next) YaleGetIJA(jj, rhs, ija);
          else jj = rhs->shape[1];

        } else { // j < jj

          // insert zero
          cast_copy_value_single((char*)(lhs->elements) + pos*nm_sizeof[l_dtype], R_ZERO, l_dtype, rhs->dtype);
        }
        ++pos; // move to next dense position
      }
    }
  }

  return lhs;
}




// Note that elements and elements_length are for initial value(s) passed in. If they are the correct length, they will
// be used directly. If not, they will be concatenated over and over again into a new elements array. If elements is NULL,
// the new elements array will not be initialized.
DENSE_STORAGE* create_dense_storage(int8_t dtype, size_t* shape, size_t rank, void* elements, size_t elements_length) {
  DENSE_STORAGE* s;
  size_t count, i, copy_length = elements_length;

  s = ALLOC( DENSE_STORAGE );
  //if (!(s = malloc(sizeof(DENSE_STORAGE)))) return NULL;

  s->rank       = rank;
  s->shape      = shape;
  s->dtype      = dtype;
  s->offset     = calloc(sizeof(size_t),rank);
  s->strides    = dense_calc_strides(shape, rank);
  s->count      = 1;
  s->src        = s;

  //fprintf(stderr, "create_dense_storage: %p\n", s);

  count         = count_dense_storage_elements(s);
  //fprintf(stderr, "count_dense_storage_elements: %d\n", count);

  if (elements_length == count) s->elements = elements;
  else {
    s->elements = ALLOC_N(char, nm_sizeof[dtype]*count);

    if (elements_length > 0) {
      // repeat elements over and over again until the end of the matrix
      for (i = 0; i < count; i += elements_length) {
        if (i + elements_length > count) copy_length = count - i;
        memcpy((char*)(s->elements)+i*nm_sizeof[dtype], (char*)(elements)+(i % elements_length)*nm_sizeof[dtype], copy_length*nm_sizeof[dtype]);
      }

      // get rid of the init_val
      free(elements);
    }
  }

  return s;
}

void delete_dense_storage(DENSE_STORAGE* s) {
  if (s) { // sometimes Ruby passes in NULL storage for some reason (probably on copy construction failure)
    if(s->count <= 1) {
      free(s->shape);
      free(s->offset);
      free(s->strides);
      free(s->elements);
      free(s);
    }
  }
}

void delete_dense_storage_ref(DENSE_STORAGE* s) {
  if (s) { // sometimes Ruby passes in NULL storage for some reason (probably on copy construction failure)
    ((DENSE_STORAGE*)s->src)->count--;
    free(s->shape);
    free(s->offset);
    free(s);
  }
}


void mark_dense_storage(void* m) {
  size_t i;
  DENSE_STORAGE* storage;

  if (m) {
    storage = (DENSE_STORAGE*)(((NMATRIX*)m)->storage);
    if (storage && storage->dtype == NM_ROBJ)
      for (i = 0; i < count_dense_storage_elements(storage); ++i)
        rb_gc_mark(*((VALUE*)((char*)(storage->elements) + i*nm_sizeof[NM_ROBJ])));
  }
}



#endif
