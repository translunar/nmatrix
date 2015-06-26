#We want this to be a C++ file since our complex types require C++.

File.open("lapacke.cpp","w") do |file|
  Dir["lapacke/**/*.c"].each do |file2|
    file.puts "#include \"#{file2}\""
  end
end
