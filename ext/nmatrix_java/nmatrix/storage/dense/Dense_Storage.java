public class Dense_Storage{

	private int dim;
	private int[] shape;
	private String dtype;
	private int offset;
	
	// Don't know about its significance
	private int stride;

	private int count;
	private double[] elements;

  /////////////////////////
  // Setters and Getters //
  /////////////////////////

  public void set_dim(int dim){
    this.dim = dim;
  }

  public int get_dim(){
    return this.dim;
  }

  public void set_shape(int[] shape){
    this.shape = shape;
  }

  public int[] get_shape(){
    return this.shape;
  }

  public void set_dtype(String dtype){
    this.dtype = dtype;
  }

  public String get_dtype(){
    return this.dtype;
  }

  public void set_offset(int offset){
    this.offset = offset;
  }

  public int get_offset(){
    return offset;
  }

  public void set_stride(int stride){
    this.stride = stride;
  }

  public int get_stride(){
    return this.stride;
  }

  public void set_count(int count){
    this.count = count;
  }

  public int get_count(){
    return this.count;
  }

  public void set_elements(double[] elements){
    this.elements = elements;
  }

  public double[] get_elements(){
    return this.elements;
  }

  /////////////////
  // Contructors //
  /////////////////

  public Dense_Storage(int dim, int[] shape, String dtype, int offset, int count, double[] elements){
  	// this.dim = dim;
  	this.shape = shape;
  	// this.dtype = dtype;
  	// this.offset = offset;
  	// this.count = count;
  	this.elements = elements;
  }

  // public Dense_Storage set_dense_storage(int dim, int[] shape, String dtype, int offset, int count, double[] elements){
  // 	this(dim, shape, dtype, offset, count, elements);
  // 	return this;
  // }

  // public static void ref_slice_copy_transposed(final Dense_Storage rhs, Dense_Storage lhs){

  // }

  
  // public static Dense_Storage cast_copy(final DENSE_STORAGE rhs){

  // }

  ///////////////////////
  // Utility Functions //
  ///////////////////////


  // public static boolean eqeq(final DENSE_STORAGE left, final DENSE_STORAGE right){

  // }

  
  // public static DENSE_STORAGE* matrix_multiply(final STORAGE_PAIR casted_storage, int[] resulting_shape, bool vector){

  // }

  // public static boolean is_hermitian(final Dense_Storage mat, int lda){

  // }

  // public static boolean is_symmetric(final DENSE_STORAGE mat, int lda){

  // }

  ///////////////////////////////////////////////////
  // Need to understand the significance of stride //
  ///////////////////////////////////////////////////

  /////////////
  // Utility //
  /////////////

  /*
   * Determine the linear array position (in elements of s) of some set of coordinates
   * (given by slice).
   */
  // size_t nm_dense_storage_pos(const DENSE_STORAGE* s, const size_t* coords) {
  //   size_t pos = 0;

  //   for (size_t i = 0; i < s->dim; ++i)
  //     pos += (coords[i] + s->offset[i]) * s->stride[i];

  //   return pos;

  // }

  // /*
  //  * Determine the a set of slice coordinates from linear array position (in elements
  //  * of s) of some set of coordinates (given by slice).  (Inverse of
  //  * nm_dense_storage_pos).
  //  *
  //  * The parameter coords_out should be a pre-allocated array of size equal to s->dim.
  //  */
  // void nm_dense_storage_coords(const DENSE_STORAGE* s, const size_t slice_pos, size_t* coords_out) {

  //   size_t temp_pos = slice_pos;

  //   for (size_t i = 0; i < s->dim; ++i) {
  //     coords_out[i] = (temp_pos - temp_pos % s->stride[i])/s->stride[i] - s->offset[i];
  //     temp_pos = temp_pos % s->stride[i];
  //   }
  // }

  // /*
  //  * Calculate the stride length.
  //  */
  // static size_t* stride(size_t* shape, size_t dim) {
  //   size_t i, j;
  //   size_t* stride = NM_ALLOC_N(size_t, dim);

  //   for (i = 0; i < dim; ++i) {
  //     stride[i] = 1;
  //     for (j = i+1; j < dim; ++j) {
  //       stride[i] *= shape[j];
  //     }
  //   }

  //   return stride;
  // }


  /*
   * Recursive slicing for N-dimensional matrix.
   */
  
  // public static void slice_copy(Dense_Storage dest, const Dense_Storage src, int[] lengths, int pdest, int psrc, int n) {
  //   if (src->dim - n > 1) {
  //     for (size_t i = 0; i < lengths[n]; ++i) {
  //       slice_copy<LDType,RDType>(dest, src, lengths,
  //                  pdest + dest.stride[n]*i,
  //                  psrc + src.stride[n]*i,
  //                  n + 1);
  //     }
  //   } else {
  //     for (size_t p = 0; p < dest->shape[n]; ++p) {
  //       reinterpret_cast<LDType*>(dest->elements)[p+pdest] = reinterpret_cast<RDType*>(src->elements)[p+psrc];
  //     }
  //     memcpy((char*)dest->elements + pdest*DTYPE_SIZES[dest->dtype],
  //         (char*)src->elements + psrc*DTYPE_SIZES[src->dtype],
  //         dest->shape[n]*DTYPE_SIZES[dest->dtype]); 
  //   }

  // }

  // /*
  //  * Recursive function, sets multiple values in a matrix from a single source value. Same basic pattern as slice_copy.
  //  */

  // public static void slice_set(Dense_Storage dest, int[]* lengths, int pdest, int rank, D final v, int v_size, int v_offset) {
  //   if (dest.dim - rank > 1) {
  //     for (int i = 0; i < lengths[rank]; ++i) {
  //       slice_set<D>(dest, lengths, pdest + dest.stride[rank] * i, rank + 1, v, v_size, v_offset);
  //     }
  //   } else {
  //     for (size_t p = 0; p < lengths[rank]; ++p, ++v_offset) {
  //       if (v_offset >= v_size) v_offset %= v_size;

  //       D* elem = reinterpret_cast<D*>(dest->elements);
  //       elem[p + pdest] = v[v_offset];
  //     }
  //   }
  // }

  
}