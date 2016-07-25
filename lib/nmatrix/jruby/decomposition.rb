class NMatrix

  # discussion in https://github.com/SciRuby/nmatrix/issues/374
  
  def matrix_solve rhs
    if rhs.shape[1] > 1
      nmatrix = NMatrix.new :copy
      nmatrix.shape = rhs.shape
      result = []
      res = []
      (0...rhs.shape[1]).each do |i|
        res << self.solve(rhs.col(i)).s.toArray.to_a
      end
      index = 0
      (0...rhs.shape[0]).each do |i|
        (0...rhs.shape[1]).each do |j|
          result[index] = res[j][i]
          index+=1
        end
      end
      nmatrix.s = ArrayRealVector.new result.to_java :double
      nmatrix.twoDMat =  MatrixUtils.createRealMatrix get_twoDArray(rhs.shape, result)
      
      return nmatrix
    else
      return self.solve rhs
    end
  end

end