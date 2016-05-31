class Abc
	@ary
	attr_accessor :ary
	def initialize(a)
		@ary = a
	end
	def methodname
		# for i in 0 .. @ary.length-1 do
	 # 		puts  yield.to_s + @ary[i].to_s
	 # 	end
	 	@ary
	end
end

ab = Abc.new([12,2,3]);
nary = Array.new()
ab.methodname.each_with_index { |v,i| nary[i] = v }
puts nary