require 'java'
require_relative '../../ext/nmatrix_java/vendor/commons-math3-3.6.1.jar'
# require_relative '../../ext/nmatrix_java/target/nmatrix.jar'

# java_import 'JNMatrix'
# java_import 'Dtype'
# java_import 'Stype'
java_import 'org.apache.commons.math3.linear.ArrayRealVector'
java_import 'org.apache.commons.math3.linear.RealMatrix'
java_import 'org.apache.commons.math3.linear.MatrixUtils'
java_import 'org.apache.commons.math3.linear.LUDecomposition'
java_import 'org.apache.commons.math3.linear.QRDecomposition'
java_import 'org.apache.commons.math3.linear.CholeskyDecomposition'

class NMatrix
  include_package 'org.apache.commons.math3.analysis.function'
  attr_accessor :shape , :dtype, :elements, :s, :nmat, :twoDMat


  def initialize(*args)
    if args[-1] == :copy
      @shape = [2,2]
      @s = [0,0,0,0]
      @dim = shape.is_a?(Array) ? shape.length : 2
    else
      if (args.length <= 3)
        @shape = args[0]
        if args[1].is_a?(Array)
          elements = args[1]
          hash = args[2]
        else
          elements = Array.new()
          hash = args[1] unless args.length<1
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
      elements = args[offset+1]

      # @dtype = interpret_dtype(argc-1-offset, argv+offset+1, stype);

      # @dtype = args[:dtype] if args[:dtype]
      @dtype_sym = nil
      @stype_sym = nil
      @default_val_num = nil
      @capacity_num = nil
      
      @size = (0...@shape.size).inject(1) { |x,i| x * @shape[i] }

      j=0;
      if (elements.is_a?(ArrayRealVector))
        @s = elements
      else
        storage = Array.new(size)
        if size > elements.length
          (0...size).each do |i|
            j=0 unless j!=elements.length
            storage[i] = elements[j]
            j+=1
          end
        else
          storage = elements
        end
        @s = ArrayRealVector.new(storage.to_java Java::double)
      end
      @dim = shape.is_a?(Array) ? shape.length : 2

      if(shape.length == 2 )
        oneDArray = @s.toArray().to_a
        twoDArray = Java::double[shape[0],shape[1]].new
        index = 0
        (0...shape[0]).each do |i|
          (0...shape[1]).each do |j|
            twoDArray[i][j] = oneDArray[index]
            index+=1
          end
        end
        @twoDMat = MatrixUtils.createRealMatrix(twoDArray)
        # puts "inited"
      end
        
    end
    # @s = @elements
    # Java enums are accessible from Ruby code as constants:
    # @nmat= JNMatrix.new(@shape, @elements , "FLOAT32", "DENSE_STORE" )
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
    return xslice(args)
  end

  def slice(*args)
    return xslice(args)
  end

  def []=(*args,value)

    to_return = nil

    if args.length > @dim+1
      raise Exception.new("wrong number of arguments (%d for %lu)", args.length, effective_dim(@dim+1))
    else
      slice = get_slice(@dim, args, @shape)
      stride = get_stride(self)
      if(slice[:single])
        pos = dense_storage_pos(slice[:coords],stride)
        @s.setEntry(pos, value)
        to_return = value
      else
        raise Exception.new("not supported")
      end
    end

    return to_return
  end



  def is_ref?
    
  end

  def dim
    shape.is_a?(Array) ? shape.length : 2
  end

  alias :dimensions :dim

  def effective_dim(s)
    d = 0
    (0...@dim).each do |i|
      d+=1 unless @shape[i] == 1
    end
    return d
  end

  alias :effective_dimensions :effective_dim

  def xslice(args)
    result = nil

    s = @s.toArray().to_a

    if @dim < args.length
      raise Exception.new("wrong number of arguments (%d for %lu)", args, effective_dim(self))
    else
      result = Array.new()

      slice = get_slice(@dim, args, @shape)
      stride = get_stride(self)
      if slice[:single]
        if (@dtype == "RUBYOBJ") 
          # result = *reinterpret_cast<VALUE*>( ttable[NM_STYPE(self)](s, slice) );
        else                                
          result = @s.getEntry(dense_storage_get(slice,stride))
        end 
      else
        result = dense_storage_get(slice,stride)
      end
    end
    return result
  end
#its by ref
  
  def dense_storage_get(slice,stride)
    if slice[:single]
      return dense_storage_pos(slice[:coords],stride)
    else
      shape = @shape.dup
      (0...@dim).each do |i|
        shape[i] = slice[:lengths][i]
      end
      psrc = dense_storage_pos(slice[:coords], stride)
      src = {}
      result = NMatrix.new(:copy)
      resultShape= Array.new(dim)
      (0...dim).each do |i|
        resultShape[i]  = slice[:lengths][i]
      end
      result.shape = resultShape
      dest = {}
      src[:stride] = get_stride(self)
      src[:elements] = @s.toArray().to_a
      dest[:stride] = get_stride(result)
      dest[:shape] = shape
      dest[:elements] = []
      temp = []
      s = (slice_copy(src, dest, slice[:lengths], 0, psrc,0))
      arr = Java::double[s.length].new
      (0...s.length).each do |i|
        arr[i] = s[i]
      end
      result.s = ArrayRealVector.new(arr)
      return result
    end
  end

  def slice_copy(src, dest,lengths, pdest, psrc,n)
    # p src
    # p dest
    
    if @dim-n>1
      (0...lengths[n]).each do |i|
        slice_copy(src, dest, lengths,pdest+dest[:stride][n]*i,psrc+src[:stride][n]*i,n+1)
      end
    else
      (0...dest[:shape][n]).each do |p|
        dest[:elements][p+pdest] = src[:elements][p+psrc]
      end
    end
    dest[:elements]
  end

  def dense_storage_pos(coords,stride)
    pos = 0;
    offset = 0
    (0...@dim).each do |i|
      pos += coords[i]  * stride[i] ;
    end
    return pos + offset;
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

  def get_stride(nmatrix)
    stride = Array.new()
    (0...nmatrix.dim).each do |i|
      stride[i] = 1;
      (i+1...dim).each do |j|
        stride[i] *= nmatrix.shape[j]
      end
    end
    stride
  end

  
  protected

  def __list_to_hash__
    
  end

  public

  def shape
    @shape
  end

  def supershape
    
  end

  def offset
    
  end

  def det_exact
    # if (:stype != :dense)
    #   raise Exception.new("can only calculate exact determinant for dense matrices")
    #   return nil
    # end

    if (@dim != 2 || @shape[0] != @shape[1])
      raise Exception.new("matrices must be square to have a determinant defined")
      return nil
    end
    to_return = nil
    if (dtype == :RUBYOBJ)
      # to_return = *reinterpret_cast<VALUE*>(result);
    else
      to_return = LUDecomposition.new(twoDMat).getDeterminant()
    end

    return to_return
  end

  alias :det :det_exact

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
    @s.toArray().to_a.to_enum
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

  def ==(otherNmatrix)
    result = false
    if (otherNmatrix.is_a?(NMatrix))
      #check dimension
      #check shape
      if (@dim != otherNmatrix.dim)
        raise Exception.new("cannot compare matrices with different dimension")
      end
      #check shape
      (0...dim).each do |i|
        if (@shape[i] != otherNmatrix.shape[i])
          raise Exception.new("cannot compare matrices with different shapes");
        end
      end

      #check the entries

      result = @s.equals(otherNmatrix.s)
    end
    result
  end

  def +(other)
    result = NMatrix.new(:copy)
    result.shape = @shape
    if (other.is_a?(NMatrix))
      #check dimension
      #check shape
      if (@dim != other.dim)
        raise Exception.new("cannot add matrices with different dimension")
      end
      #check shape
      (0...dim).each do |i|
        if (@shape[i] != other.shape[i])
          raise Exception.new("cannot add matrices with different shapes");
        end
      end
      result.s = @s.add(other.s)
    else
      result.s = @s.mapAddToSelf(other)
    end
    result
  end

  def -(other)
    result = NMatrix.new(:copy)
    result.shape = @shape
    if (other.is_a?(NMatrix))
      #check dimension
      #check shape
      if (@dim != other.dim)
        raise Exception.new("cannot subtract matrices with different dimension")
      end
      #check shape
      (0...dim).each do |i|
        if (@shape[i] != other.shape[i])
          raise Exception.new("cannot subtract matrices with different shapes");
        end
      end
      result.s = @s.subtract(other.s)
    else
      result.s = @s.mapSubtractToSelf(other)
    end
    result
  end

  def *(other)
    result = NMatrix.new(:copy)
    result.shape = @shape
    if (other.is_a?(NMatrix))
      #check dimension
      #check shape
      if (@dim != other.dim)
        raise Exception.new("cannot multiply matrices with different dimension")
      end
      #check shape
      (0...dim).each do |i|
        if (@shape[i] != other.shape[i])
          raise Exception.new("cannot multiply matrices with different shapes");
        end
      end
      result.s = @s.ebeMultiply(other.s)
    else
      result.s = @s.mapMultiplyToSelf(other)
    end
    result
  end

  def /(other)
    result = NMatrix.new(:copy)
    result.shape = @shape
    if (other.is_a?(NMatrix))
      #check dimension
      #check shape
      if (@dim != other.dim)
        raise Exception.new("cannot divide matrices with different dimension")
      end
      #check shape
      (0...dim).each do |i|
        if (@shape[i] != other.shape[i])
          raise Exception.new("cannot divide matrices with different shapes");
        end
      end
      result.s = @s.ebeDivide(other.s)
    else
      result.s = @s.mapDivideToSelf(other)
    end
    result
  end

  def **
    @nmap.mapToSelf(univariate_function_power)
  end

  def %(other)
    raise Exception.new("modulus not supported in NMatrix-jruby")
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
    result = NMatrix.new(:copy)
    result.shape = @shape
    result.s = @s.mapToSelf(Sin.new())
    result
  end

  def cos
    result = NMatrix.new(:copy)
    result.shape = @shape
    result.s = @s.mapToSelf(Cos.new())
    result
  end

  def tan
    result = NMatrix.new(:copy)
    result.shape = @shape
    result.s = @s.mapToSelf(Tan.new())
    result
  end

  def asin
    result = NMatrix.new(:copy)
    result.shape = @shape
    result.s = @s.mapToSelf(Asin.new())
    result
  end

  def acos
    result = NMatrix.new(:copy)
    result.shape = @shape
    result.s = @s.mapToSelf(Acos.new())
    result
  end

  def atan
    result = NMatrix.new(:copy)
    result.shape = @shape
    result.s = @s.mapToSelf(Atan.new())
    result
  end

  def sinh
    result = NMatrix.new(:copy)
    result.shape = @shape
    result.s = @s.mapToSelf(Sinh.new())
    result
  end

  def cosh
    result = NMatrix.new(:copy)
    result.shape = @shape
    result.s = @s.mapToSelf(Cosh.new())
    result
  end

  def tanh
    result = NMatrix.new(:copy)
    result.shape = @shape
    result.s = @s.mapToSelf(Tanh.new())
    result
  end

  def asinh
    result = NMatrix.new(:copy)
    result.shape = @shape
    result.s = @s.mapToSelf(Asinh.new())
    result
  end

  def acosh
    result = NMatrix.new(:copy)
    result.shape = @shape
    result.s = @s.mapToSelf(Acosh.new())
    result
  end

  def atanh
    result = NMatrix.new(:copy)
    result.shape = @shape
    result.s = @s.mapToSelf(Atanh.new())
    result
  end

  def exp
    result = NMatrix.new(:copy)
    result.shape = @shape
    result.s = @s.mapToSelf(Exp.new())
    result
  end

  def log2
    result = NMatrix.new(:copy)
    result.shape = @shape
    result.s = @s.mapToSelf(Log.new())
    result
  end

  def log10
    result = NMatrix.new(:copy)
    result.shape = @shape
    result.s = @s.mapToSelf(Log10.new())
    result
  end

  def sqrt
    result = NMatrix.new(:copy)
    result.shape = @shape
    result.s = @s.mapToSelf(Sqrt.new())
    result
  end

  def erf
    # @nmap.mapToSelf(univariate_function_)
  end

  def erfc
    # @nmap.mapToSelf(univariate_function_)
  end

  def cbrt
    result = NMatrix.new(:copy)
    result.shape = @shape
    result.s = @s.mapToSelf(Cbrt.new())
    result
  end

  def gamma
    # @nmap.mapToSelf(univariate_function_)
  end

  def -@
    # @nmap.mapToSelf(univariate_function_)
  end

  def floor
    result = NMatrix.new(:copy)
    result.shape = @shape
    result.s = @s.mapToSelf(Floor.new())
    result
  end

  def ceil
    result = NMatrix.new(:copy)
    result.shape = @shape
    result.s = @s.mapToSelf(Ceil.new())
    result
  end

  def round
    # @nmap.mapToSelf(univariate_function_)
  end

  def =~ (other)
    lha = @s.toArray.to_a
    rha = other.s.toArray.to_a
    resultArray = Array.new(lha.length)
    if (other.is_a?(NMatrix))
      #check dimension
      if (@dim != other.dim)
        raise Exception.new("cannot compare matrices with different dimension")
        return nil
      end
      #check shape
      (0...dim).each do |i|
        if (@shape[i] != other.shape[i])
          raise Exception.new("cannot compare matrices with different shapes");
          return nil
        end
      end
      #check the entries
      (0...lha.length).each do |i|
        resultArray[i] = lha[i] =~ rha[i] ? true : false
      end
      # result = NMatrix.new(@shape, resultArray, dtype: :int64)
    end
    resultArray
  end

  def !~ (other)
    lha = @s.toArray.to_a
    rha = other.s.toArray.to_a
    resultArray = Array.new(lha.length)
    if (other.is_a?(NMatrix))
      #check dimension
      if (@dim != other.dim)
        raise Exception.new("cannot compare matrices with different dimension")
        return nil
      end
      #check shape
      (0...dim).each do |i|
        if (@shape[i] != other.shape[i])
          raise Exception.new("cannot compare matrices with different shapes");
          return nil
        end
      end
      #check the entries
      (0...lha.length).each do |i|
        resultArray[i] = lha[i] !~ rha[i] ? true : false
      end
      # result = NMatrix.new(@shape, resultArray, dtype: :int64)
    end
    resultArray
  end

  def <= (other)
    lha = @s.toArray.to_a
    rha = other.s.toArray.to_a
    resultArray = Array.new(lha.length)
    if (other.is_a?(NMatrix))
      #check dimension
      if (@dim != other.dim)
        raise Exception.new("cannot compare matrices with different dimension")
        return nil
      end
      #check shape
      (0...dim).each do |i|
        if (@shape[i] != other.shape[i])
          raise Exception.new("cannot compare matrices with different shapes");
          return nil
        end
      end
      #check the entries
      (0...lha.length).each do |i|
        resultArray[i] = lha[i] <= rha[i] ? true : false
      end
      # result = NMatrix.new(@shape, resultArray, dtype: :int64)
    end
    resultArray
  end

  def >= (other)
    lha = @s.toArray.to_a
    rha = other.s.toArray.to_a
    resultArray = Array.new(lha.length)
    if (other.is_a?(NMatrix))
      #check dimension
      if (@dim != other.dim)
        raise Exception.new("cannot compare matrices with different dimension")
        return nil
      end
      #check shape
      (0...dim).each do |i|
        if (@shape[i] != other.shape[i])
          raise Exception.new("cannot compare matrices with different shapes");
          return nil
        end
      end
      #check the entries
      (0...lha.length).each do |i|
        resultArray[i] = lha[i] >= rha[i] ? true : false
      end
      # result = NMatrix.new(@shape, resultArray, dtype: :int64)
    end
    resultArray
  end

  def < (other)
    lha = @s.toArray.to_a
    rha = other.s.toArray.to_a
    resultArray = Array.new(lha.length)
    if (other.is_a?(NMatrix))
      #check dimension
      if (@dim != other.dim)
        raise Exception.new("cannot compare matrices with different dimension")
        return nil
      end
      #check shape
      (0...dim).each do |i|
        if (@shape[i] != other.shape[i])
          raise Exception.new("cannot compare matrices with different shapes");
          return nil
        end
      end
      #check the entries
      (0...lha.length).each do |i|
        resultArray[i] = lha[i] < rha[i] ? true : false
      end
      # result = NMatrix.new(@shape, resultArray, dtype: :int64)
    end
    resultArray
  end

  def > (other)
    lha = @s.toArray.to_a
    rha = other.s.toArray.to_a
    resultArray = Array.new(lha.length)
    if (other.is_a?(NMatrix))
      #check dimension
      if (@dim != other.dim)
        raise Exception.new("cannot compare matrices with different dimension")
        return nil
      end
      #check shape
      (0...dim).each do |i|
        if (@shape[i] != other.shape[i])
          raise Exception.new("cannot compare matrices with different shapes");
          return nil
        end
      end
      #check the entries
      (0...lha.length).each do |i|
        resultArray[i] = lha[i] > rha[i] ? true : false
      end
      # result = NMatrix.new(@shape, resultArray, dtype: :int64)
    end
    resultArray
  end

  # /////////////////////////////
  # // Helper Instance Methods //
  # /////////////////////////////

  # /////////////////////////
  # // Matrix Math Methods //
  # /////////////////////////

  def get_oneDArray(shape,twoDArray)
    oneDArray = Java::double[shape[0]*shape[1]].new
    index = 0
    (0...shape[0]).each do |i|
      (0...shape[1]).each do |j|
        oneDArray[index] = twoDArray[i][j]
        index+=1
      end
    end
    oneDArray
  end

  def dot(other)
    result = nil
    if (other.is_a?(NMatrix))
      #check dimension
      if (@shape.length!=2 || other.shape.length!=2)
        raise Exception.new("please convert array to nx1 or 1xn NMatrix first")
        return nil
      end
      #check shape
      if (@shape[1] != other.shape[0])
        raise Exception.new("incompatible dimensions")
        return nil
      end
      
      result = NMatrix.new(:copy)
      result.shape = [shape[0],other.shape[1]]
      result.twoDMat = @twoDMat.multiply(other.twoDMat)
      result.s = ArrayRealVector.new(get_oneDArray(result.shape, result.twoDMat.getData()))
    else
      raise Exception.new("cannot have dot product with a scalar");
    end
    return result;
  end

  def symmetric?
    return is_symmetric(false)
  end

  def is_symmetric(hermitian)
    is_symmetric = false

    if (@shape[0] == @shape[1] and @dim == 2)
      if @stype == :dense
        if (hermitian)
          #Currently, we are not dealing with complex matrices.
          eps = 0
          is_symmetric = MatrixUtils.isSymmetric(@twoDMat, eps)
        else
          eps = 0
          is_symmetric = MatrixUtils.isSymmetric(@twoDMat, eps)
        end

      else
        #TODO: Implement, at the very least, yale_is_symmetric. Model it after yale/transp.template.c.
        raise Exception.new("symmetric? and hermitian? only implemented for dense currently")
      end
    end
    return is_symmetric ? true : false
  end

  def hermitian?
    return is_symmetric(true)
  end

  def capacity

  end

  # // protected methods

  protected
  
  def __inverse__
    # if (:stype != :dense)
    #   raise Exception.new("needs exact determinant implementation for this matrix stype")
    #   return nil
    # end
    
    if (@dim != 2 || @shape[0] != @shape[1])
      raise Exception.new("matrices must be square to have an inverse defined")
      return nil
    end
    to_return = nil
    if (dtype == :RUBYOBJ)
      # to_return = *reinterpret_cast<VALUE*>(result);
    else
      elements = @nmat.twoDMat.inverse().to_a
      to_return = NMatrix.new(@shape, elements, dtype: :int64)
    end

    return to_return
  end
  
  def __inverse_exact__
    # if (:stype != :dense)
    #   raise Exception.new("needs exact determinant implementation for this matrix stype")
    #   return nil
    # end
    
    if (@dim != 2 || @shape[0] != @shape[1])
      raise Exception.new("matrices must be square to have an inverse defined")
      return nil
    end
    to_return = nil
    if (dtype == :RUBYOBJ)
      # to_return = *reinterpret_cast<VALUE*>(result);
    else
      elements = @nmat.twoDMat.inverse().to_a
      to_return = NMatrix.new(@shape, elements, dtype: :int64)
    end

    return to_return
    
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
