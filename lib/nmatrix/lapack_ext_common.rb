class NMatrix
  def NMatrix.register_lapack_extension(name)
    if (defined? @@lapack_extension)
      raise "Attempting to load #{name} when #{@@lapack_extension} is already loaded. You can only load one LAPACK extension."
    end

    @@lapack_extension = name
  end
end
