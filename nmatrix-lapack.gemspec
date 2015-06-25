lib = File.expand_path('../lib/', __FILE__)
$:.unshift lib unless $:.include?(lib)

require 'nmatrix/version'

Gem::Specification.new do |gem|
  gem.name = "nmatrix-lapack"
  gem.version = NMatrix::VERSION::STRING
  gem.summary = "general LAPACK backend for nmatrix"
  gem.description = "For using linear algebra fuctions provided by LAPACK and BLAS"
  gem.homepage = 'http://sciruby.com'
  gem.authors = ['John Woods', 'Chris Wailes', 'Aleksey Timin']
  gem.email =  ['john.o.woods@gmail.com']
  gem.license = 'BSD 3-clause'
  gem.post_install_message = <<-EOF
***********************************************************
Welcome to SciRuby: Tools for Scientific Computing in Ruby!

NMatrix requires a C compiler, and has been tested only
with GCC 4.6+. We are happy to accept contributions
which improve the portability of this project.

Also required is ATLAS. Most Linux distributions and Mac
versions include ATLAS, but you may wish to compile it
yourself. The Ubuntu/Debian apt package for ATLAS WILL
NOT WORK with NMatrix if LAPACK is also installed.

More explicit instructions for NMatrix and SciRuby should
be available on the SciRuby website, sciruby.com, or
through our mailing list (which can be found on our web-
site).

Thanks for trying out NMatrix! Happy coding!

***********************************************************
EOF

  gem.files         = ["lib/nmatrix/lapack.rb"]
  gem.files         += `git ls-files -- ext/nmatrix_lapack`.split("\n")
  gem.files         += `git ls-files -- ext/nmatrix | grep ".h$"`.split("\n") #need nmatrix header files to compile
  gem.test_files    = `git ls-files -- spec`.split("\n")
  gem.test_files    -= `git ls-files -- spec/plugins`.split("\n")
  gem.test_files    += `git ls-files -- spec/plugins/lapack`.split("\n")
  gem.extensions = ['ext/nmatrix_lapack/extconf.rb']
  gem.require_paths = ["lib"]

  gem.required_ruby_version = '>= 1.9'

  gem.add_dependency 'nmatrix', NMatrix::VERSION::STRING
end

