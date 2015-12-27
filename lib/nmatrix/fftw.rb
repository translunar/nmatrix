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

  def fft
    input = self.dtype == :complex128 ? self : self.cast(dtype: :complex128)
    plan = NMatrix::FFTW::Plan.new([self.size])
    plan.set_input input
    plan.execute
    plan.output
  end

  def fft2
    raise ShapeError, "Shape must be 2 (is #{self.shape})" if self.shape.size != 2
    input = self.dtype == :complex128 ? self : self.cast(dtype: :complex128)
    plan = NMatrix::FFTW::Plan.new(self.shape, dim: 2)
    plan.set_input input
    plan.execute
    plan.output
  end

  module FFTW

    # Human friendly DSL for computing FFTs
    def self.compute &block
      
    end

    class Plan

      REAL_REAL_FFT_KINDS_HASH = {
        r2hc:    0,
        hc2r:    1,
        dht:     2,
        redft00: 3,
        redft01: 4,
        redft10: 5,
        redft11: 6,
        rodft00: 7,
        rodft01: 9,
        rodft10: 8,
        rodft11: 10
      }

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

      DATA_TYPE_HASH = {
        complex_complex: 0,
        real_complex:    1,
        complex_real:    2,
        real_real:       3
      }

      VALID_OPTS = [:dim, :type, :direction, :flags, :rrkind]

      attr_reader :shape
      attr_reader :size
      attr_reader :type
      attr_reader :direction
      attr_reader :flags
      attr_reader :dim
      attr_reader :input
      attr_reader :output
      attr_reader :rrkind

      # Create a plan for DFT
      def initialize shape, opts={}
        verify_opts opts
        opts = {
          dim: 1,
          flags: :estimate,
          direction: :forward,
          type: :complex_complex
        }.merge(opts)

        @type      = opts[:type]
        @dim       = opts[:dim]
        @direction = opts[:direction]
        @shape     = shape.is_a?(Array) ? shape : [shape]
        @size      = @shape.inject(:*)
        @flags     = opts[:flags].is_a?(Array) ? opts[:flags] : [opts[:flags]]
        @rrkind    = opts[:rrkind]

        raise ArgumentError, "dim (#{@dim}) cannot be more than size of shape #{@shape.size}" if
          @dim > @shape.size

        @plan_data = __create_plan__(@shape, @size, @dim, 
          combine_flags(@flags), FFT_DIRECTION_HASH[@direction], 
          DATA_TYPE_HASH[@type], encoded_rr_kind)
      end

      # Set input for the DFT
      def set_input ip
        raise ArgumentError, "stype must be dense." if ip.stype != :dense

        case @type
        when :complex_complex, :complex_real
          raise ArgumentError, "dtype must be complex128." if ip.dtype != :complex128
        when :real_complex, :real_real
          raise ArgumentError, "dtype must be float64." if ip.dtype != :float64
        else
          raise "Invalid type #{@type}"
        end

        @input = ip
        __set_input__(ip, @plan_data, DATA_TYPE_HASH[@type])
      end

      # Execute DFT with the set plan
      def execute
        @output = 
        case @type
        when :complex_complex
          @input.clone_structure        
        when :real_complex
          NMatrix.new([@input.size/2 + 1], dtype: :complex128)
        when :complex_real, :real_real
          NMatrix.new([@input.size], dtype: :float64)
        else
          raise "Invalid type #{@type}"
        end

        __execute__(@plan_data, @output, DATA_TYPE_HASH[@type])
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

      def encoded_rr_kind
        return @rrkind.map { |e| REAL_REAL_FFT_KINDS_HASH[e] } if @rrkind
      end
    end
  end
end