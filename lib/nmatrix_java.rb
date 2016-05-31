require '../ext/nmatrix_java/vendor/commons-math3-3.6.1.jar'
require '../ext/nmatrix_java/target/nmatrix.jar'
java_import 'JNMatrix'
java_import 'Dtype'
java_import 'Stype'
java_import 'Simple'
class NMatrix
	@JNMatrix
	@shape
	@dtype
	@elements
	@q
	
	attr_accessor :shape , :dtype, :elements, :s
	# def NMatrix.to_s
	# 	return elements
	# end


	def initialize(shape,elements,args)
		@dtype = args[:dtype] if args[:dtype]
		@stype = args[:stype] if args[:stype]
		@shape = shape
		@elements = elements
		@s={}
		@s['elements'] = elements
		# Java enums are accessible from Ruby code as constants:
		@q = Simple.new(2);
		@j= JNMatrix.new( 2, [2,3,4,5] , "FLOAT32", "DENSE_STORE" )
	end
	#  rb_define_method(cNMatrix, "dtype", (METHOD)nm_dtype, 0);
	#  rb_define_method(cNMatrix, "stype", (METHOD)nm_stype, 0);
	#  rb_define_method(cNMatrix, "cast_full",  (METHOD)nm_cast, 3);
	#  rb_define_method(cNMatrix, "default_value", (METHOD)nm_default_value, 0);

		
	def entries
		return @s['elements']
	end

	def dtype
		return @dtype
  end

	# static VALUE nm_stype(VALUE self) {
	#   NM_CONSERVATIVE(nm_register_value(&self));
	#   VALUE stype = ID2SYM(rb_intern(STYPE_NAMES[NM_STYPE(self)]));
	#   NM_CONSERVATIVE(nm_unregister_value(&self));
	#   return stype;
	# }
  def stype
  	@stype = :dense;
  end

  def cast_full

  end

  def default_value
  	
  end

	#  rb_define_protected_method(cNMatrix, "__list_default_value__", (METHOD)nm_list_default_value, 0);
	#  rb_define_protected_method(cNMatrix, "__yale_default_value__", (METHOD)nm_yale_default_value, 0);


	#  rb_define_method(cNMatrix, "[]", (METHOD)nm_mref, -1);
	#  rb_define_method(cNMatrix, "slice", (METHOD)nm_mget, -1);
	#  rb_define_method(cNMatrix, "[]=", (METHOD)nm_mset, -1);
	#  rb_define_method(cNMatrix, "is_ref?", (METHOD)nm_is_ref, 0);
	#  rb_define_method(cNMatrix, "dimensions", (METHOD)nm_dim, 0);
	#  rb_define_method(cNMatrix, "effective_dimensions", (METHOD)nm_effective_dim, 0);

	#  rb_define_protected_method(cNMatrix, "__list_to_hash__", (METHOD)nm_to_hash, 0); // handles list and dense, which are n-dimensional

	#  rb_define_method(cNMatrix, "shape", (METHOD)nm_shape, 0);
	#  rb_define_method(cNMatrix, "supershape", (METHOD)nm_supershape, 0);
	#  rb_define_method(cNMatrix, "offset", (METHOD)nm_offset, 0);
	#  rb_define_method(cNMatrix, "det_exact", (METHOD)nm_det_exact, 0);
	#  rb_define_method(cNMatrix, "complex_conjugate!", (METHOD)nm_complex_conjugate_bang, 0);

	#  rb_define_protected_method(cNMatrix, "reshape_bang", (METHOD)nm_reshape_bang, 1);

	#  // Iterators public methods
	#  rb_define_method(cNMatrix, "each_with_indices", (METHOD)nm_each_with_indices, 0);
	#  rb_define_method(cNMatrix, "each_stored_with_indices", (METHOD)nm_each_stored_with_indices, 0);
	#  rb_define_method(cNMatrix, "map_stored", (METHOD)nm_map_stored, 0);
	#  rb_define_method(cNMatrix, "each_ordered_stored_with_indices", (METHOD)nm_each_ordered_stored_with_indices, 0);

	def each_with_indices
	 # NM_CONSERVATIVE(nm_register_value(&nmatrix));
	 #  VALUE to_return = Qnil;

	 #  switch(NM_STYPE(nmatrix)) {
	 #  case nm::YALE_STORE:
	 #    to_return = nm_yale_each_with_indices(nmatrix);
	 #    break;
	 #  case nm::DENSE_STORE:
	 #    to_return = nm_dense_each_with_indices(nmatrix);
	 #    break;
	 #  case nm::LIST_STORE:
	 #    to_return = nm_list_each_with_indices(nmatrix, false);
	 #    break;
	 #  default:
	 #    NM_CONSERVATIVE(nm_unregister_value(&nmatrix));
	 #    rb_raise(nm_eDataTypeError, "Not a proper storage type");
	 #  }

	 #  NM_CONSERVATIVE(nm_unregister_value(&nmatrix));
	 #  return to_return;
	end


	def each_stored_with_indices
		# NM_CONSERVATIVE(nm_register_value(&nmatrix));
  # 	VALUE to_return = Qnil;

	 #  switch(NM_STYPE(nmatrix)) {
	 #  case nm::YALE_STORE:
	 #    to_return = nm_yale_each_stored_with_indices(nmatrix);
	 #    break;
	 #  case nm::DENSE_STORE:
	 #    to_return = nm_dense_each_with_indices(nmatrix);
	 #    break;
	 #  case nm::LIST_STORE:
	 #    to_return = nm_list_each_with_indices(nmatrix, true);
	 #    break;
	 #  default:
	 #    NM_CONSERVATIVE(nm_unregister_value(&nmatrix));
	 #    rb_raise(nm_eDataTypeError, "Not a proper storage type");
	 #  }

	 #  NM_CONSERVATIVE(nm_unregister_value(&nmatrix));
	 #  return to_return;

	end

	def map_stored
		
	end

	def each_ordered_stored_with_indeces
		
	end


	#  // Iterators protected methods
	#  rb_define_protected_method(cNMatrix, "__dense_each__", (METHOD)nm_dense_each, 0);
	#  rb_define_protected_method(cNMatrix, "__dense_map__", (METHOD)nm_dense_map, 0);
	#  rb_define_protected_method(cNMatrix, "__dense_map_pair__", (METHOD)nm_dense_map_pair, 1);
	#  rb_define_protected_method(cNMatrix, "__list_map_merged_stored__", (METHOD)nm_list_map_merged_stored, 2);
	#  rb_define_protected_method(cNMatrix, "__list_map_stored__", (METHOD)nm_list_map_stored, 1);
	#  rb_define_protected_method(cNMatrix, "__yale_map_merged_stored__", (METHOD)nm_yale_map_merged_stored, 2);
	#  rb_define_protected_method(cNMatrix, "__yale_map_stored__", (METHOD)nm_yale_map_stored, 0);
	#  rb_define_protected_method(cNMatrix, "__yale_stored_diagonal_each_with_indices__", (METHOD)nm_yale_stored_diagonal_each_with_indices, 0);
	#  rb_define_protected_method(cNMatrix, "__yale_stored_nondiagonal_each_with_indices__", (METHOD)nm_yale_stored_nondiagonal_each_with_indices, 0);

	protected
	# for (size_t i = 0; i < nm_storage_count_max_elements(s); ++i) {
	#      nm_dense_storage_coords(sliced_dummy, i, temp_coords);
	#      sliced_index = nm_dense_storage_pos(s, temp_coords);
	#      VALUE v = nm::rubyobj_from_cval((char*)(s->elements) + sliced_index*DTYPE_SIZES[NM_DTYPE(nmatrix)], NM_DTYPE(nmatrix)).rval;
	#      rb_yield( v ); // yield to the copy we made
 #    }

 	def __dense_each__
 		@s['elements']
 	end

	def __dense_map__
		
	end

	def __dense_map_pair__

	end

	public

		# rb_define_method(cNMatrix, "[]", (METHOD)nm_mref, -1);	end
	 #  rb_define_method(cNMatrix, "slice", (METHOD)nm_mget, -1);
	 #  rb_define_method(cNMatrix, "[]=", (METHOD)nm_mset, -1);
	 #  rb_define_method(cNMatrix, "is_ref?", (METHOD)nm_is_ref, 0);
	 #  rb_define_method(cNMatrix, "dimensions", (METHOD)nm_dim, 0);
	 #  rb_define_method(cNMatrix, "effective_dimensions", (METHOD)nm_effective_dim, 0);

	def [] *args
		# @s['elements'][args]
		0
	end

	def []=

	end

	def slice

	end

	def dimensions

	end

	def effective_dimensions

	end

	#  rb_define_method(cNMatrix, "==",    (METHOD)nm_eqeq,        1);

	#  rb_define_method(cNMatrix, "+",      (METHOD)nm_ew_add,      1);
	#  rb_define_method(cNMatrix, "-",      (METHOD)nm_ew_subtract,  1);
	#  rb_define_method(cNMatrix, "*",      (METHOD)nm_ew_multiply,  1);
	#  rb_define_method(cNMatrix, "/",      (METHOD)nm_ew_divide,    1);
	#  rb_define_method(cNMatrix, "**",    (METHOD)nm_ew_power,    1);
	#  rb_define_method(cNMatrix, "%",     (METHOD)nm_ew_mod,      1);

	def ==
		
	end

	def +
		
	end

	def -
		
	end

	def *
		
	end

	def /
		
	end

	def **
		
	end

	def %
		
	end

	#  rb_define_method(cNMatrix, "atan2", (METHOD)nm_noncom_ew_atan2, -1);
	#  rb_define_method(cNMatrix, "ldexp", (METHOD)nm_noncom_ew_ldexp, -1);
	#  rb_define_method(cNMatrix, "hypot", (METHOD)nm_noncom_ew_hypot, -1);

	def atan2
		
	end

	def ldexp

	end

	def hypot

	end
	#  rb_define_method(cNMatrix, "sin",   (METHOD)nm_unary_sin,   0);
	#  rb_define_method(cNMatrix, "cos",   (METHOD)nm_unary_cos,   0);
	#  rb_define_method(cNMatrix, "tan",   (METHOD)nm_unary_tan,   0);
	#  rb_define_method(cNMatrix, "asin",  (METHOD)nm_unary_asin,  0);
	#  rb_define_method(cNMatrix, "acos",  (METHOD)nm_unary_acos,  0);
	#  rb_define_method(cNMatrix, "atan",  (METHOD)nm_unary_atan,  0);
	#  rb_define_method(cNMatrix, "sinh",  (METHOD)nm_unary_sinh,  0);
	#  rb_define_method(cNMatrix, "cosh",  (METHOD)nm_unary_cosh,  0);
	#  rb_define_method(cNMatrix, "tanh",  (METHOD)nm_unary_tanh,  0);
	#  rb_define_method(cNMatrix, "asinh", (METHOD)nm_unary_asinh, 0);
	#  rb_define_method(cNMatrix, "acosh", (METHOD)nm_unary_acosh, 0);
	#  rb_define_method(cNMatrix, "atanh", (METHOD)nm_unary_atanh, 0);
	#  rb_define_method(cNMatrix, "exp",   (METHOD)nm_unary_exp,   0);
	#  rb_define_method(cNMatrix, "log2",  (METHOD)nm_unary_log2,  0);
	#  rb_define_method(cNMatrix, "log10", (METHOD)nm_unary_log10, 0);
	#  rb_define_method(cNMatrix, "sqrt",  (METHOD)nm_unary_sqrt,  0);
	#  rb_define_method(cNMatrix, "erf",   (METHOD)nm_unary_erf,   0);
	#  rb_define_method(cNMatrix, "erfc",  (METHOD)nm_unary_erfc,  0);
	#  rb_define_method(cNMatrix, "cbrt",  (METHOD)nm_unary_cbrt,  0);
	#  rb_define_method(cNMatrix, "gamma", (METHOD)nm_unary_gamma, 0);
	#  rb_define_method(cNMatrix, "log",   (METHOD)nm_unary_log,  -1);
	#  rb_define_method(cNMatrix, "-@",    (METHOD)nm_unary_negate,0);
	#  rb_define_method(cNMatrix, "floor", (METHOD)nm_unary_floor, 0);
	#  rb_define_method(cNMatrix, "ceil", (METHOD)nm_unary_ceil, 0);
	#  rb_define_method(cNMatrix, "round", (METHOD)nm_unary_round, -1);

	def sin
		
	end

	def cos
		
	end

	def tan
		
	end

	def asin
		
	end

	def acos
		
	end

	def atan
		
	end

	def sinh
		
	end

	def cosh
		
	end

	def tanh
		
	end

	def asinh
		
	end

	def acosh
		
	end

	def atanh
		
	end

	def exp
		
	end

	def log2
		
	end

	def log10
		
	end

	def sqrt

	end

	def erf
		
	end

	def erfc
		
	end

	def cbrt
		
	end

	def gamma
		
	end

	def log
		
	end

	def -@
		
	end

	def floor
		
	end

	def ceil
		
	end

	def round
		
	end

#  rb_define_method(cNMatrix, "=~", (METHOD)nm_ew_eqeq, 1);
#  rb_define_method(cNMatrix, "!~", (METHOD)nm_ew_neq, 1);
#  rb_define_method(cNMatrix, "<=", (METHOD)nm_ew_leq, 1);
#  rb_define_method(cNMatrix, ">=", (METHOD)nm_ew_geq, 1);
#  rb_define_method(cNMatrix, "<", (METHOD)nm_ew_lt, 1);
#  rb_define_method(cNMatrix, ">", (METHOD)nm_ew_gt, 1);
	def =~
		
	end

	def !~

	end

	def <=

	end

	def >=
		
	end

	def <
		
	end

	def >
		
	end


end