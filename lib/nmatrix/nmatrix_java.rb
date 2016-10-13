require 'java'
require_relative '../../ext/nmatrix_java/vendor/commons-math3-3.6.1.jar'
require_relative '../../ext/nmatrix_java/target/nmatrix.jar'

java_import 'JNMatrix'
java_import 'Dtype'
java_import 'Stype'
java_import 'org.apache.commons.math3.analysis.function.Sin'

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

  def []=(args,arg2)
    # pos = xslice(args)
    # @nmat.setElement(pos, args[1])
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
      stride = get_stride
      if slice[:single]
        if (@dtype == "RUBYOBJ") 
          # result = *reinterpret_cast<VALUE*>( ttable[NM_STYPE(self)](s, slice) );
        else                                
          result = @s[dense_storage_get(slice,stride)]
        end 
      else
        result = dense_storage_get(slice,stride)
        # NMATRIX* mat  = NM_ALLOC(NMATRIX);
        # mat->stype    = NM_STYPE(self);
        # mat->storage  = (STORAGE*)((*slice_func)( s, slice ));
        # nm_register_nmatrix(mat);
        # result        = Data_Wrap_Struct(CLASS_OF(self), nm_mark, delete_func, mat);
      end
    end

    return result
  end
#its by ref
  def dense_storage_get(slice,stride)
    if slice[:single]
      return dense_storage_pos(slice[:coords],stride)
    else
      ns={}
      ns[:dim]        = @dim
      ns[:dtype]      = @dtype
      ns[:offset]     = []
      ns[:shape]      = []
      (0...ns[:dim]).each do |i|
        ns[:offset][i] = slice[:coords][i] 
        # + s[:offset][i]
        ns[:shape][i]  = slice[:lengths][i]
      end

      # ns[:stride]     = s[:stride];
      ns[:elements]   = @s

      # s[:src][:count] += 1
      # ns[:src] = s[:src]

      return ns;
    end
    # return dense_storage_pos(slice[:coords],stride)
  end

  def dense_storage_pos(coords,stride)
    pos = 0;
    offset = 0
    (0...@dim).each do |i|
      pos += coords[i]  * stride[i] + offset;
    end
    return pos
  end

  # def get_element
  #   for (p = 0; p < dest->shape[n]; ++p) {
  #       reinterpret_cast<LDType*>(dest->elements)[p+pdest] = reinterpret_cast<RDType*>(src->elements)[p+psrc];
  #     }
  # end

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
      elsif v.is_a?(Range)
        begin_ = v.begin
        end_ = v.end
        excl = v.exclude_end?
        slice[:coords][r] = (begin_ < 0) ? shape[r] + begin_ : begin_
      
        # Exclude last element for a...b range
        if (end_ < 0)
          slice[:lengths][r] = shape_array[r] + end_ - slice[:coords][r] + (excl ? 0 : 1)
        else
          slice[:lengths][r] = end_ - slice[:coords][r] + (excl ? 0 : 1)
        end

        slice[:single] = false
        t+=1
      else
        raise Exception.new("expected Fixnum or Range for slice component instead of")
      end

      if (slice[:coords][r] > shape_array[r] || slice[:coords][r] + slice[:lengths][r] > shape_array[r])
        raise Exception.new("slice is larger than matrix in dimension %lu (slice component %lu)", r, t);
      end
    end

    return slice
  end

  def get_stride
    stride = Array.new()
    (0...@dim).each do |i|
      stride[i] = 1;
      (i+1...dim).each do |j|
        stride[i] *= @shape[j]
      end
    end
    stride
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

  def +(other)
    result = nil
    if (other.is_a?(NMatrix))
      #check dimension
      #check shape
      if (@dim != other.dim)
        raise Exception.new("cannot add matrices with different dimension")
      end
      #check shape
      (i=0...dim).each do |i|
        if (@shape[i] != other.shape[i])
          raise Exception.new("cannot add matrices with different shapes");
        end
      end
      resultArray = @nmat.add(other.nmat).to_a
      result = NMatrix.new(shape, resultArray,  dtype: :int64)
    else
      resultArray = @nmat.mapAddToSelf(other).to_a
      result = NMatrix.new(shape, resultArray,  dtype: :int64)
    end
    result
  end

  def -(other)
    result = nil
    if (other.is_a?(NMatrix))
      #check dimension
      #check shape
      if (@dim != other.dim)
        raise Exception.new("cannot subtract matrices with different dimension")
      end
      #check shape
      (i=0...dim).each do |i|
        if (@shape[i] != other.shape[i])
          raise Exception.new("cannot subtract matrices with different shapes");
        end
      end
      resultArray = @nmat.subtract(other.nmat).to_a
      result = NMatrix.new(shape, resultArray,  dtype: :int64)
    else
      resultArray = @nmat.mapSubtractToSelf(other).to_a
      result = NMatrix.new(shape, resultArray,  dtype: :int64)
    end
    result
  end

  def *(other)
    result = nil
    if (other.is_a?(NMatrix))
      #check dimension
      #check shape
      if (@dim != other.dim)
        raise Exception.new("cannot multiply matrices with different dimension")
      end
      #check shape
      (i=0...dim).each do |i|
        if (@shape[i] != other.shape[i])
          raise Exception.new("cannot multiply matrices with different shapes");
        end
      end
      resultArray = @nmat.ebeMultiply(other.nmat).to_a
      result = NMatrix.new(shape, resultArray,  dtype: :int64)
    else
      resultArray = @nmat.mapMultiplyToSelf(other).to_a
      result = NMatrix.new(shape, resultArray,  dtype: :int64)
    end
    result
  end

  def /(other)
    result = nil
    if (other.is_a?(NMatrix))
      #check dimension
      #check shape
      if (@dim != other.dim)
        raise Exception.new("cannot divide matrices with different dimension")
      end
      #check shape
      (i=0...dim).each do |i|
        if (@shape[i] != other.shape[i])
          raise Exception.new("cannot divide matrices with different shapes");
        end
      end
      resultArray = @nmat.ebeDivide(other.nmat).to_a
      result = NMatrix.new(shape, resultArray,  dtype: :int64)
    else
      resultArray = @nmat.mapDivideToSelf(other).to_a
      result = NMatrix.new(shape, resultArray,  dtype: :int64)
    end
    result
  end

  def **
    @nmap.mapToSelf(univariate_function_power)
  end

  def %
    @nmap.mapToSelf(univariate_function_mod)
  end

  def atan2
    # resultArray = @nmat.mapAtan2ToSelf().to_a
    # result = NMatrix.new(@shape, resultArray,  dtype: :int64)
  end

  def ldexp
    resultArray = @nmat.mapLdexpToSelf().to_a
    result = NMatrix.new(@shape, resultArray,  dtype: :int64)
  end

  def hypot
    resultArray = @nmat.mapHypotToSelf().to_a
    result = NMatrix.new(@shape, resultArray,  dtype: :int64)
  end

  def sin
    resultArray = @nmat.mapSinToSelf().to_a
    result = NMatrix.new(@shape, resultArray,  dtype: :int64)
  end

  def cos
    resultArray = @nmat.mapCosToSelf().to_a
    result = NMatrix.new(@shape, resultArray,  dtype: :int64)
  end

  def tan
    resultArray = @nmat.mapTanToSelf().to_a
    result = NMatrix.new(@shape, resultArray,  dtype: :int64)
  end

  def asin
    resultArray = @nmat.mapAsinToSelf().to_a
    result = NMatrix.new(@shape, resultArray,  dtype: :int64)
  end

  def acos
    resultArray = @nmat.mapAcosToSelf().to_a
    result = NMatrix.new(@shape, resultArray,  dtype: :int64)
  end

  def atan
    resultArray = @nmat.mapAtanToSelf().to_a
    result = NMatrix.new(@shape, resultArray,  dtype: :int64)
  end

  def sinh
    resultArray = @nmat.mapSinhToSelf().to_a
    result = NMatrix.new(@shape, resultArray,  dtype: :int64)
  end

  def cosh
    resultArray = @nmat.mapCoshToSelf().to_a
    result = NMatrix.new(@shape, resultArray,  dtype: :int64)
  end

  def tanh
    resultArray = @nmat.mapTanhToSelf().to_a
    result = NMatrix.new(@shape, resultArray,  dtype: :int64)
  end

  def asinh
    resultArray = @nmat.mapAsinhToSelf().to_a
    result = NMatrix.new(@shape, resultArray,  dtype: :int64)
  end

  def acosh
    resultArray = @nmat.mapAcoshToSelf().to_a
    result = NMatrix.new(@shape, resultArray,  dtype: :int64)
  end

  def atanh
    resultArray = @nmat.mapAtanhToSelf().to_a
    result = NMatrix.new(@shape, resultArray,  dtype: :int64)
  end

  def exp
    resultArray = @nmat.mapExpToSelf().to_a
    result = NMatrix.new(@shape, resultArray,  dtype: :int64)
  end

  def log2
    resultArray = @nmat.mapLog2ToSelf().to_a
    result = NMatrix.new(@shape, resultArray,  dtype: :int64)
  end

  def log10
    resultArray = @nmat.mapLog10ToSelf().to_a
    result = NMatrix.new(@shape, resultArray,  dtype: :int64)
  end

  def sqrt
    resultArray = @nmat.mapSqrtToSelf().to_a
    result = NMatrix.new(@shape, resultArray,  dtype: :int64)
  end

  def erf
    # @nmap.mapToSelf(univariate_function_)
  end

  def erfc
    # @nmap.mapToSelf(univariate_function_)
  end

  def cbrt
    resultArray = @nmat.mapCbrtToSelf().to_a
    result = NMatrix.new(@shape, resultArray,  dtype: :int64)
  end

  def gamma
    # @nmap.mapToSelf(univariate_function_)
  end

  def log
    resultArray = @nmat.mapLogToSelf().to_a
    result = NMatrix.new(@shape, resultArray,  dtype: :int64)
  end

  def -@
    # @nmap.mapToSelf(univariate_function_)
  end

  def floor
    resultArray = @nmat.mapFloorToSelf().to_a
    result = NMatrix.new(@shape, resultArray,  dtype: :int64)
  end

  def ceil
    resultArray = @nmat.mapCeilToSelf().to_a
    result = NMatrix.new(@shape, resultArray,  dtype: :int64)
  end

  def round
    # @nmap.mapToSelf(univariate_function_)
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
