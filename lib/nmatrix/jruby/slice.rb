class NMatrix
  def dense_storage_get(const STORAGE* storage, SLICE* slice)
    if slice[:single]
      return dense_storage_pos(slice[:coords])
    else
      new_shape = Array.new dim
      (0...dim).each fo |i|
        new_shape[i]  = slice[:lengths][i]
      end

      # DENSE_STORAGE* ns = nm_dense_storage_create(s->dtype, shape, s->dim, NULL, 0);

      slice_copy(ns,
          reinterpret_cast<const DENSE_STORAGE*>(s->src),
          slice->lengths,
          0,
          nm_dense_storage_pos(s, slice->coords),
          0);

      return ns

    end
  end

  def dense_storage_ref(const STORAGE* storage, SLICE* slice) {
    DENSE_STORAGE* s = (DENSE_STORAGE*)storage;

    if (slice->single)
      return dense_storage_pos(slice[:coords])
    else
      ns = NMatrix.new :copy
      ns.dim        = dim;
      ns.dtype      = dtype
      ns.shape      = Array.new dim

      # replace by get_slice function
      # (0...ns.dim).each do |i|
      #   ns->offset[i] = slice->coords[i] + s->offset[i];
      #   ns->shape[i]  = slice->lengths[i];
      # end
      return ns
    end
  end
end