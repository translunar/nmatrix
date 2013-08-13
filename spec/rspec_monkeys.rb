module RSpec::Matchers::BuiltIn
  class BeWithin

    def of(expected)
      @expected = expected
      @unit     = ''
      if expected.is_a?(NMatrix)
        @tolerance = if @delta.is_a?(NMatrix)
                       @delta.clone
                     elsif @delta.is_a?(Array)
                       NMatrix.new(:dense, expected.shape, @delta, expected.dtype)
                     else
                       NMatrix.ones_like(expected) * @delta
                     end
      else
        @tolerance = @delta
      end

      self
    end

    def percent_of(expected)
      @expected  = expected
      @unit      = '%'
      @tolerance = @expected.abs * @delta / 100.0 # <- only change is to reverse abs and @delta
      self
    end
  end
end