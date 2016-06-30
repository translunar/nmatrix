class NMatrix
  
  def matrix_solve b
    if b.shape[1] > 1
      nmatrix = NMatrix.new :copy
      nmatrix.shape = b.shape
      result = []
      (0...b.shape[1]).each do |i|
        result.concat(self.solve(b.col(i)).s.toArray.to_a)
      end
      nmatrix.s = ArrayRealVector.new result.to_java :double
      nmatrix.twoDMat =  MatrixUtils.createRealMatrix get_twoDArray(b.shape, result)
      return nmatrix
    else
      return self.solve b
    end
  end

end