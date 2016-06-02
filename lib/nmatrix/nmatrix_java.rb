require 'java'
require_relative '../../ext/nmatrix_java/vendor/commons-math3-3.6.1.jar'
require_relative '../../ext/nmatrix_java/target/nmatrix.jar'

java_import 'JNMatrix'
java_import 'Dtype'
java_import 'Stype'

class NMatrix
  attr_accessor :shape , :dtype, :elements, :s, :dim, :nmat


  def initialize(*args)
    # puts args.length
    # puts args
    if (args.length <= 3)
      @shape = args[0]
      if args[1].is_a?(Array)
        elements = args[1]
        hash = args[2]
      else
        elements = Array.new()
        hash = args[1]
      end
    end

    
    offset = 0

    if (!args[0].is_a?(Symbol) && !args[0].is_a?(String))
      @stype = :dense
    else
      offset = 1
      @stype = :dense
    end

    @shape = args[offset]
    @shape = [shape,shape] unless shape.is_a?(Array)

    # @dtype = interpret_dtype(argc-1-offset, argv+offset+1, stype);

    # @dtype = args[:dtype] if args[:dtype]
    @dtype_sym = nil
    @stype_sym = nil
    @default_val_num = nil
    @capacity_num = nil
    
    
    @size = (0...@shape.size).inject(1) { |x,i| x * @shape[i] }

    j=0;
    @elements = Array.new(size)
    if size > elements.length
      (0...size).each do |i|
        j=0 unless j!=elements.length
        @elements[i] = elements[j]
        j+=1
      end
    else
      @elements = elements
    end
    @dim = shape.is_a?(Array) ? shape.length : 2
    @s = @elements
    # Java enums are accessible from Ruby code as constants:
    @nmat= JNMatrix.new(@shape, @elements , "FLOAT32", "DENSE_STORE" )
  end

  def entries
    return @s
  end

  def dtype
    return @dtype
  end

  def stype
    @stype = :dense
  end

  def cast_full
    #not implemented currently
  end

  def default_value
    return nil
  end

  def __list_default_value__
    #not implemented currently
  end

  def __yale_default_value__
    #not implemented currently
  end

  def [] *args
    return xslice(args);
  end

  def slice(*args)
    return nm_xslice(args)
  end

  def []=(*args)
    @dim

    to_return = nil;

    if argc > @dim+1
      raise Exception.new("wrong number of arguments (%d for %lu)", argc, self.effective_dimensions+1);
    else
      slice = get_slice(args)


      # ttable[NM_STYPE(self)](self, slice, argv[argc-1]);

      to_return = argv[argc-1];
    end

    return to_return;
  end



  def is_ref?
    
  end

  def dimensions
    @dim
  end

  def effective_dimensions
    d = 0
    (0...@dim).each do |i|
      d+=1 unless s.shape[i] == 1
    end
    return d
  end

  def xslice(args)
    result = nil

    s = @elements

    if @dim < args.length
      raise Exception.new("wrong number of arguments (%d for %lu)", args, effective_dim(s))
    else
      result = Array.new()

      slice = get_slice(@dim, args, @shape);

      if slice[:single]
        if (@dtype == "RUBYOBJ") 
          # result = *reinterpret_cast<VALUE*>( ttable[NM_STYPE(self)](s, slice) );
        else                                
          result = slice
        end 
      else
        # NMATRIX* mat  = NM_ALLOC(NMATRIX);
        # mat->stype    = NM_STYPE(self);
        # mat->storage  = (STORAGE*)((*slice_func)( s, slice ));
        # nm_register_nmatrix(mat);
        # result        = Data_Wrap_Struct(CLASS_OF(self), nm_mark, delete_func, mat);
      end
    end

    return result
  end

  def get_slice(dim, args, shape_array)
    slice = {}
    slice[:coords]=[]
    slice[:lengths]=[]
    slice[:single] = true

    argc = args.length

    t = 0
    (0...dim).each do |r|
      v = t == argc ? nil : args[t]

      if(argc - t + r < dim && shape_array[r] ==1)
        slice[:coords][r]  = 0
        slice[:lengths][r] = 1
      elsif v.is_a?(Fixnum)
        v_ = v.to_i.to_int
        if (v_ < 0) # checking for negative indexes
          slice[:coords][r]  = shape_array[r]+v_
        else
          slice[:coords][r]  = v_
          slice[:lengths][r] = 1
        t+=1
        end
      elsif (v.is_a?(Symbol) && v.__id__ == "*")
        slice[:coords][r] = 0
        slice[:lengths][r] = shape_array[r]
        slice[:single] = false
        t+=1
      # elsif condition
        # not implemented currently
        # for range
        # if condition
          
        # elsif condition
          
        # slice[:single] = false
        # t++
        # end
      else
        raise Exception.new("expected Fixnum or Range for slice component instead of")
      end

      if (slice[:coords][r] > shape_array[r] || slice[:coords][r] + slice[:lengths][r] > shape_array[r])
        raise Exception.new("slice is larger than matrix in dimension %lu (slice component %lu)", r, t);
      end
    end

    return slice
  end

  
  protected

  def __list_to_hash__
    
  end

  public

  # def shape
    
  # end

  def supershape
    
  end

  def offset
    
  end

  def det_exact

  end

  def complex_conjugate!

  end


  protected

  def reshape_bang

  end


  public

  def each_with_indices
    to_return = nil

    case(@dtype)
    when 'DENSE_STORE'
      to_return = @s
      break;
    else
      raise Exception.new(nm_eDataTypeError, "Not a proper storage type");
    end
    to_return
  end


  def each_stored_with_indices
    to_return = nil

    case(@dtype)
    when 'DENSE_STORE'
      to_return = @s
      break;
    else
      raise Exception.new(nm_eDataTypeError, "Not a proper storage type");
    end
    to_return;
  end

  def map_stored
    
  end

  def each_ordered_stored_with_indices
    
  end


  protected

  def __dense_each__
    @s
  end

  def __dense_map__
    
  end

  def __dense_map_pair__

  end

  def __list_map_merged_stored__
    
  end

  def __list_map_stored__
    
  end

  def __yale_map_merged_stored__
    
  end

  def __yale_map_stored__
    
  end

  def __yale_stored_diagonal_each_with_indices__
    
  end

  def __yale_stored_nondiagonal_each_with_indices__
    
  end


  public

  def == (otherNmatrix)
    result = false
    if (otherNmatrix.is_a?(NMatrix))
      #check dimension
      #check shape
      if (@dim != otherNmatrix.dim)
        raise Exception.new("cannot compare matrices with different dimension")
      end
      #check shape
      (i=0...dim).each do |i|
        if (@shape[i] != otherNmatrix.shape[i])
          raise Exception.new("cannot compare matrices with different shapes");
        end
      end

      #check the entries

      result = @nmat.equals(otherNmatrix.nmat)
    end
    result
  end

  def +(otherNmatrix)
    result = nil
    if (otherNmatrix.is_a?(NMatrix))
      #check dimension
      #check shape
      if (@dim != otherNmatrix.dim)
        raise Exception.new("cannot compare matrices with different dimension")
      end
      #check shape
      (i=0...dim).each do |i|
        if (@shape[i] != otherNmatrix.shape[i])
          raise Exception.new("cannot compare matrices with different shapes");
        end
      end

      resultArray = @nmat.add(otherNmatrix.nmat).to_a
      result = NMatrix.new(shape, resultArray,  dtype: :int64)
    end
    result
  end

  def -
    @nmap.mapSubtractToSelf(d)
  end

  def *
    @nmap.mapMultiplyToSelf(d)
  end

  def /
    @nmap.mapDivideToSelf(d)
  end

  def **
    @nmap.mapToSelf(univariate_function_power)
  end

  def %
    @nmap.mapToSelf(univariate_function_mod)
  end

  def atan2
    @nmap.mapToSelf(univariate_function_atan2)
  end

  def ldexp
    @nmap.mapToSelf(univariate_function_)
  end

  def hypot
    @nmap.mapToSelf(univariate_function_)
  end

  def sin
    @nmap.mapToSelf(univariate_function_)
  end

  def cos
    @nmap.mapToSelf(univariate_function_)
  end

  def tan
    @nmap.mapToSelf(univariate_function_)
  end

  def asin
    @nmap.mapToSelf(univariate_function_)
  end

  def acos
    @nmap.mapToSelf(univariate_function_)
  end

  def atan
    @nmap.mapToSelf(univariate_function_)
  end

  def sinh
    @nmap.mapToSelf(univariate_function_)
  end

  def cosh
    @nmap.mapToSelf(univariate_function_)
  end

  def tanh
    @nmap.mapToSelf(univariate_function_)
  end

  def asinh
    @nmap.mapToSelf(univariate_function_)
  end

  def acosh
    @nmap.mapToSelf(univariate_function_)
  end

  def atanh
    @nmap.mapToSelf(univariate_function_)
  end

  def exp
    @nmap.mapToSelf(univariate_function_)
  end

  def log2
    @nmap.mapToSelf(univariate_function_)
  end

  def log10
    @nmap.mapToSelf(univariate_function_)
  end

  def sqrt
    @nmap.mapToSelf(univariate_function_)
  end

  def erf
    @nmap.mapToSelf(univariate_function_)
  end

  def erfc
    @nmap.mapToSelf(univariate_function_)
  end

  def cbrt
    @nmap.mapToSelf(univariate_function_)
  end

  def gamma
    @nmap.mapToSelf(univariate_function_)
  end

  def log
    @nmap.mapToSelf(univariate_function_)
  end

  def -@
    @nmap.mapToSelf(univariate_function_)
  end

  def floor
    @nmap.mapToSelf(univariate_function_)
  end

  def ceil
    @nmap.mapToSelf(univariate_function_)
  end

  def round
    @nmap.mapToSelf(univariate_function_)
  end

  def =~
    @nmap.mapToSelf(univariate_function_)
  end

  def !~
    @nmap.mapToSelf(univariate_function_)
  end

  def <=
    @nmap.mapToSelf(univariate_function_)
  end

  def >=
    @nmap.mapToSelf(univariate_function_)
  end

  def <
    @nmap.mapToSelf(univariate_function_)
  end

  def >
    @nmap.mapToSelf(univariate_function_)
  end

  # /////////////////////////////
  # // Helper Instance Methods //
  # /////////////////////////////

  # /////////////////////////
  # // Matrix Math Methods //
  # /////////////////////////

  def dot
    
  end

  def symmetric?(nmat)
    return is_symmetric(nmat, false)
  end

  def is_symmetric(nmat, hermitian)
    is_symmetric = false

    if (nmat.shape[0] == nmat.shape[1] and nmat.dim == 2)
      if nmat.stype == :DENSE_STORE
        if (hermitian)
          # is_symmetric = nm_dense_storage_is_hermitian((DENSE_STORAGE*)(m->storage), m->storage->shape[0]);

        else
          # is_symmetric = nmat.is_symmetric((DENSE_STORAGE*)(m->storage), m->storage->shape[0]);
        end

      else
        #TODO: Implement, at the very least, yale_is_symmetric. Model it after yale/transp.template.c.
        raise Exception.new("symmetric? and hermitian? only implemented for dense currently")
      end
    end
    return is_symmetric ? true : false
  end

  def hermitian?
    
  end

  def capacity

  end

  # // protected methods

  protected
  
  def __inverse__(inverse, bang)

    # if (@dtype != "DENSE_STORE")
    #   rb_raise(rb_eNotImpError, "needs exact determinant implementation for this matrix stype");
    #   return Qnil;
    # end

    # if (@dim != 2 || @shape[0] != @shape[1])
   #    rb_raise(nm_eShapeError, "matrices must be square to have an inverse defined");
   #    return nil
    # end

    # if (bang == true)
   #    math_inverse(@shape[0], @s, @dtype)
          
   #    return self;
    # end

    # math_inverse(NM_SHAPE0(inverse), @s 

    # return inverse
  end

  def __inverse_exact__(inverse, bang)

  end

  private

  # // private methods

  def __hessenberg__
    
  end

  # /////////////////
  # // FFI Methods //
  # /////////////////

  public

  def data_pointer
    
  end

  # /////////////
  # // Aliases //
  # /////////////

  # rb_define_alias(cNMatrix, "dim", "dimensions");
  # rb_define_alias(cNMatrix, "effective_dim", "effective_dimensions");
  # rb_define_alias(cNMatrix, "equal?", "eql?");


  def elementwise_op(op,left_val,right_val)

  end
end