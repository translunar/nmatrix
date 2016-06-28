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

end