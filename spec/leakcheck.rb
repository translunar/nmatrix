require "./lib/nmatrix"

n = NMatrix.new(:yale, [8,2], :int64)
m = NMatrix.new(:yale, [2,8], :int64)
100.times do
  n.dot(m)
end
GC.start