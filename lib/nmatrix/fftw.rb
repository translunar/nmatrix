#--
# = NMatrix
#
# A linear algebra library for scientific computation in Ruby.
# NMatrix is part of SciRuby.
#
# NMatrix was originally inspired by and derived from NArray, by
# Masahiro Tanaka: http://narray.rubyforge.org
#
# == Copyright Information
#
# SciRuby is Copyright (c) 2010 - 2014, Ruby Science Foundation
# NMatrix is Copyright (c) 2012 - 2014, John Woods and the Ruby Science Foundation
#
# Please see LICENSE.txt for additional copyright notices.
#
# == Contributing
#
# By contributing source code to SciRuby, you agree to be bound by
# our Contributor Agreement:
#
# * https://github.com/SciRuby/sciruby/wiki/Contributor-Agreement
#
# == fftw.rb
#
# ruby file for the nmatrix-fftw gem. Loads the C extension and defines
# nice ruby interfaces for FFTW functions.
#++

require 'nmatrix/nmatrix.rb'
require "nmatrix_fftw.so"

class NMatrix
  module FFTW

    # Human friendly DSL for computing FFTs
    def self.compute &block
      
    end

    class Plan

      FLAG_VALUE_HASH = {
        estimate: 64,
        measure: 0,
        exhaustive: 8,
        patient: 32
      }

      FFT_DIRECTION_HASH = {
        forward: -1,
        backward: 1
      }

      VALID_OPTS = [:dim, :type, :direction, :flags]

      attr_reader :shape
      attr_reader :size
      attr_reader :type
      attr_reader :direction
      attr_reader :flags
      attr_reader :dim
      attr_reader :input
      attr_reader :output

      # Create a plan for DFT
      def initialize shape, opts={}
        verify_opts opts
        opts = {
          dim: 1,
          flags: :estimate,
          direction: :forward,
          type: :c2c
        }.merge(opts)

        @type      = opts[:type]
        @dim       = opts[:dim]
        @direction = opts[:direction]
        @shape     = shape.is_a?(Array) ? shape : [shape]
        @size      = @shape.inject(:*)
        @flags     = opts[:flags].is_a?(Array) ? opts[:flags] : [opts[:flags]]

        @plan_data = __create_plan__(@shape, @size, @dim, 
          combine_flags(@flags), FFT_DIRECTION_HASH[@direction])
      end

      # Set input for the DFT
      def set_input ip
        raise ArgumentError, "stype must be dense." if ip.stype != :dense

        case @type
        when :c2c
          raise ArgumentError, "dtype must be complex128." if ip.dtype != :complex128
        when :r2c
          raise NotImplementedError
        when :c2r
          raise NotImplementedError
        end

        @input = ip
        __set_input__(ip, @plan_data)
      end

      # Execute DFT with the set plan
      def execute
        @output = @input.clone_structure
        __execute__(@plan_data, @output)
      end
     private

      def combine_flags flgs
        temp = 0
        flgs.each do |f|
          temp |= FLAG_VALUE_HASH[f]
        end

        temp
      end

      def verify_opts opts
        unless (opts.keys- VALID_OPTS).empty?
          raise ArgumentError, "#{opts.keys- VALID_OPTS} are invalid opts."
        end
      end
    end
  end
end

