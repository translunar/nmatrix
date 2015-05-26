lib = File.expand_path('../lib/', __FILE__)
$:.unshift lib unless $:.include?(lib)

require 'nmatrix/version'

Gem::Specification.new do |gem|
  gem.name = "nmatrix-atlas"
  gem.version = NMatrix::VERSION::STRING
  gem.summary = "ATLAS backend for nmatrix"
  gem.description = "For using linear algebra fuctions provided by ATLAS"
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

  gem.files         = `git ls-files -- lib | grep atlas`.split("\n")
  gem.files         += `git ls-files -- ext | grep atlas`.split("\n")
  #gem.test_files    = `git ls-files -- {test,spec,features}/*`.split("\n")
  #gem.executables   = `git ls-files -- bin/*`.split("\n").map{ |f| File.basename(f) }
  gem.extensions = ['ext/nmatrix_atlas/extconf.rb']
  gem.require_paths = ["lib"]

  gem.required_ruby_version = '>= 1.9'

  #gem.add_dependency 'packable', '~> 1.3', '>= 1.3.5'
  #gem.add_development_dependency 'bundler', '~>1.6'
  #gem.add_development_dependency 'pry', '~>0.10'
  #gem.add_development_dependency 'rake', '~>10.3'
  #gem.add_development_dependency 'rake-compiler', '~>0.8'
  #gem.add_development_dependency 'rdoc', '~>4.0', '>=4.0.1'
  #gem.add_development_dependency 'rspec', '~>2.14'
  #gem.add_development_dependency 'rspec-longrun', '~>1.0'
  gem.add_dependency 'nmatrix', NMatrix::VERSION::STRING
end

