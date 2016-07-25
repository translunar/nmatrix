class NMatrix
  def +(other)
    result = NMatrix.new(:copy)
    result.dim = @dim
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
      result.s = @s.copy.add(other.s)
    else
      result.s = @s.copy.mapAddToSelf(other)
    end
    result
  end

  def -(other)
    result = NMatrix.new(:copy)
    result.shape = @shape
    result.dim = @dim
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
      result.s = @s.copy.subtract(other.s)
    else
      result.s = @s.copy.mapSubtractToSelf(other)
    end
    result
  end

  def *(other)
    result = NMatrix.new(:copy)
    result.shape = @shape
    result.dim = @dim
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
      result.s = @s.copy.ebeMultiply(other.s)
    else
      result.s = @s.copy.mapMultiplyToSelf(other)
    end
    result
  end

  def /(other)
    result = NMatrix.new(:copy)
    result.shape = @shape
    result.dim = @dim
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
      result.s = @s.copy.ebeDivide(other.s)
    else
      result.s = @s.copy.mapDivideToSelf(other)
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
    result.dim = @dim
    result.s = @s.copy.mapToSelf(Sin.new())
    result
  end

  def cos
    result = NMatrix.new(:copy)
    result.shape = @shape
    result.dim = @dim
    result.s = @s.copy.mapToSelf(Cos.new())
    result
  end

  def tan
    result = NMatrix.new(:copy)
    result.shape = @shape
    result.dim = @dim
    result.s = @s.copy.mapToSelf(Tan.new())
    result
  end

  def asin
    result = NMatrix.new(:copy)
    result.shape = @shape
    result.dim = @dim
    result.s = @s.copy.mapToSelf(Asin.new())
    result
  end

  def acos
    result = NMatrix.new(:copy)
    result.shape = @shape
    result.dim = @dim
    result.s = @s.copy.mapToSelf(Acos.new())
    result
  end

  def atan
    result = NMatrix.new(:copy)
    result.shape = @shape
    result.dim = @dim
    result.s = @s.copy.mapToSelf(Atan.new())
    result
  end

  def sinh
    result = NMatrix.new(:copy)
    result.shape = @shape
    result.dim = @dim
    result.s = @s.copy.mapToSelf(Sinh.new())
    result
  end

  def cosh
    result = NMatrix.new(:copy)
    result.shape = @shape
    result.dim = @dim
    result.s = @s.copy.mapToSelf(Cosh.new())
    result
  end

  def tanh
    result = NMatrix.new(:copy)
    result.shape = @shape
    result.dim = @dim
    result.s = @s.copy.mapToSelf(Tanh.new())
    result
  end

  def asinh
    result = NMatrix.new(:copy)
    result.shape = @shape
    result.dim = @dim
    result.s = @s.copy.mapToSelf(Asinh.new())
    result
  end

  def acosh
    result = NMatrix.new(:copy)
    result.shape = @shape
    result.dim = @dim
    result.s = @s.copy.mapToSelf(Acosh.new())
    result
  end

  def atanh
    result = NMatrix.new(:copy)
    result.shape = @shape
    result.dim = @dim
    result.s = @s.copy.mapToSelf(Atanh.new())
    result
  end

  def exp
    result = NMatrix.new(:copy)
    result.shape = @shape
    result.dim = @dim
    result.s = @s.copy.mapToSelf(Exp.new())
    result
  end

  def log2
    result = NMatrix.new(:copy)
    result.shape = @shape
    result.dim = @dim
    result.s = @s.copy.mapToSelf(Log.new())
    result
  end

  def log10
    result = NMatrix.new(:copy)
    result.shape = @shape
    result.dim = @dim
    result.s = @s.copy.mapToSelf(Log10.new())
    result
  end

  def sqrt
    result = NMatrix.new(:copy)
    result.shape = @shape
    result.dim = @dim
    result.s = @s.copy.mapToSelf(Sqrt.new())
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
    result.dim = @dim
    result.s = @s.copy.mapToSelf(Cbrt.new())
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
    result.dim = @dim
    result.s = @s.copy.mapToSelf(Floor.new())
    result
  end

  def ceil
    result = NMatrix.new(:copy)
    result.shape = @shape
    result.dim = @dim
    result.s = @s.copy.mapToSelf(Ceil.new())
    result
  end

  def round
    # @nmap.mapToSelf(univariate_function_)
  end

end