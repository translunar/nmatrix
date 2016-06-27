class NMatrix

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
        end
        slice[:lengths][r] = 1
        t+=1
      elsif (v.is_a?(Symbol) && v == :*)
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
        raise(ArgumentError, "expected Fixnum or Range for slice component instead of #{v.class}")
      end

      if (slice[:coords][r] > shape_array[r] || slice[:coords][r] + slice[:lengths][r] > shape_array[r])
        raise(RangeError, "slice is larger than matrix in dimension #{r} (slice component #{t})")
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

  def xslice(args)
    result = nil

    s = @s.toArray().to_a

    if @dim < args.length
      raise(ArgumentError,"wrong number of arguments (#{args} for #{effective_dim(self)})")
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

  def dense_storage_coords(s, slice_pos, coords_out, stride, offset)  #array, int, array
    temp_pos = slice_pos;

    (0...dim).each do |i|
      coords_out[i] = (temp_pos - temp_pos % stride[i])/stride[i] - offset[i];
      temp_pos = temp_pos % stride[i]
    end

    return temp_pos
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

  def slice_set(dest, lengths, pdest, rank, v, v_size, v_offset)
    if (dim - rank > 1)
      (0...lengths[rank]).each do |i|
        slice_set(dest, lengths, pdest + dest[:stride][rank] * i, rank + 1, v, v_size, v_offset);
      end
    else
      (0...lengths[rank]).each do |p|
        v_offset %= v_size if(v_offset >= v_size)
        # elem = dest[:elements]
        # elem[p + pdest] = v[v_offset]
        @s.setEntry(p + pdest, v[v_offset])
        v_offset += 1
      end
    end
  end

  def dense_storage_set(slice, right)
    # s = NM_STORAGE_DENSE(left);

    # std::pair<NMATRIX*,bool> nm_and_free =
    #   interpret_arg_as_dense_nmatrix(right, s->dtype);

    # Map the data onto D* v.
    stride = get_stride(self)
    v_size = 1

    # if(nm_and_free.first) {
    #   t = Array.new(size)
    #   v_size = count_max_elements(t)
    # els
    if(right.is_a?(Array))    
      v_size = right.length
      v = right
      if (dtype == :RUBYOBJ)
        # nm_register_values(reinterpret_cast<VALUE*>(v), v_size)
      end

      (0...v_size).each do |m|
        v[m] = right[m]
      end
    else 
      v = [right]
      if (@dtype == :RUBYOBJ)
        # nm_register_values(reinterpret_cast<VALUE*>(v), v_size)
      end
    end
    if(slice[:single])
      # reinterpret_cast<D*>(s->elements)[nm_dense_storage_pos(s, slice->coords)] = v;
      pos = dense_storage_pos(slice[:coords],stride)
      @s.setEntry(pos, v[0])
    else
      v_offset = 0
      dest = {}
      dest[:stride] = get_stride(self)
      dest[:shape] = shape
      # dest[:elements] = @s.toArray().to_a
      dense_pos = dense_storage_pos(slice[:coords],stride)
      slice_set(dest, slice[:lengths], dense_pos, 0, v, v_size, v_offset)
    end

  end

  # def dense_storage_get(const STORAGE* storage, SLICE* slice)
  #   if slice[:single]
  #     return dense_storage_pos(slice[:coords])
  #   else
  #     new_shape = Array.new dim
  #     (0...dim).each fo |i|
  #       new_shape[i]  = slice[:lengths][i]
  #     end

  #     # DENSE_STORAGE* ns = nm_dense_storage_create(s->dtype, shape, s->dim, NULL, 0);

  #     slice_copy(ns,
  #         reinterpret_cast<const DENSE_STORAGE*>(s->src),
  #         slice->lengths,
  #         0,
  #         nm_dense_storage_pos(s, slice->coords),
  #         0);

  #     return ns

  #   end
  # end

  # def dense_storage_ref(const STORAGE* storage, SLICE* slice) {
  #   DENSE_STORAGE* s = (DENSE_STORAGE*)storage;

  #   if (slice->single)
  #     return dense_storage_pos(slice[:coords])
  #   else
  #     ns = NMatrix.new :copy
  #     ns.dim        = dim;
  #     ns.dtype      = dtype
  #     ns.shape      = Array.new dim

  #     # replace by get_slice function
  #     # (0...ns.dim).each do |i|
  #     #   ns->offset[i] = slice->coords[i] + s->offset[i];
  #     #   ns->shape[i]  = slice->lengths[i];
  #     # end
  #     return ns
  #   end
  # end
end