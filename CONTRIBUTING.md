NMatrix is part of SciRuby, a collaborative effort to bring scientific computation to Ruby. If you want to help, please
do so!

This guide covers ways in which you can contribute to the development of SciRuby and, more specifically, NMatrix.

## How to help

There are various ways to help NMatrix: bug reports, coding and documentation. All of them are important.

First, you can help implement new features or bug fixes. To do that, visit our
[roadmap](https://github.com/SciRuby/nmatrix/wiki/Roadmap) or our [issue tracker][2]. If you find something that you
want to work on, post it in the issue or on our [mailing list][1].

You need to send tests together with your code. No exceptions. You can ask for our opinion, but we won't accept patches
without good spec coverage.

We use RSpec for testing. If you aren't familiar with it, there's a good
[guide to better specs with RSpec](http://betterspecs.org/) that shows a bit of the syntax and how to use it properly.
However, the best resource is probably the specs that already exist -- so just read them.

And don't forget to write documentation (we use RDoc). It's necessary to allow others to know what's available in the
library. There's a section on it later in this guide.

We only accept bug reports and pull requests in GitHub. You'll need to create a new (free) account if you don't have one
already. To learn how to create a pull request, please see
[this guide on collaborating](https://help.github.com/categories/63/articles).

If you have a question about how to use NMatrix or SciRuby in general or a feature/change in mind, please ask the
[sciruby-dev mailing list][1].

Thanks!

## Coding

To start helping with the code, you need to have all the dependencies in place:

- ATLAS and LAPACK
- GCC 4.3+
- git
- Ruby 1.9+
- `bundler` gem

Now, you need to clone the git repository:

```bash
$ git clone git://github.com/SciRuby/nmatrix.git
$ cd nmatrix
$ bundle install
$ rake compile
$ rake spec
```

This will install all dependencies, compile the extension and run the specs.

If everything's fine until now, you can create a new branch to work on your feature:

```bash
$ git branch new-feature
$ git checkout new-feature
```

Before commiting any code, please read our
[Contributor Agreement](http://github.com/SciRuby/sciruby/wiki/Contributor-Agreement).

### Guidelines for interfacing with C/C++

NMatrix uses a lot of C/C++ to speed up execution of processes and give more control over data types, storage types, etc. Since we are interfacing between two very different languages, things can get out of hand pretty fast.

Please go thorough this before you create any C accessors:

* Perform all pre-computation error checking in Ruby. 
    - Since this is (mostly) a trivial computation, doing this in Ruby before calling the actual C code will save immensely on time needed to read and write code. Plus, one won't need to write obscure code for accessing NMatrix attributes in C code (`NM_SHAPE0(obj)` for example), saving both the readers' and writers' time.
* Curate any extra data (cloned objects, trivial computations, etc.) in Ruby.
* Every private/protected function name implemented in C _MUST_ start and end with double underscores, and this name must be _the same_ as the name of the Ruby function that will call the computation.
    - For example, for `#solve`, you'll see a function `def solve .. end`; inside of which is a protected function implemented in C, which is called `__solve__`.
* The corresponding extern "C" exposed function should have a name format `nm_#{function_name}`, this  function should call a C++ function which should follow a convention `nm_#{sub_namespace}_#{function_name}` and the C++ templated function that is called by this function should be inside a namespace `nm::#{sub_namespace}`.
    - For example, the C accessor for `__solve__` is `nm_solve`, and the C++ templated function called by `nm_solve` is `nm::math::solve`.
* Do _NOT_ resolve VALUE into constituent elements unless they reach the function where the elements are needed or it is absolutely necessary. Passing around a VALUE in the C/C++ core is much more convienient than passing around `void*` pointers which point to an array of matrix elements. 
    - If you'll look at [this line](https://github.com/v0dro/nmatrix/commit/b957c5fdbcef0bf1ebe78922f84f9ea37938b247#diff-55f2ba27400bce950282e78db97bfbcfR1791), you'll notice that matrix elements aren't resolved until the final step i.e. passing the elements into the `nm::math::solve` function for the actual computation.

Basically, follow a practice of 'once you enter C, never look back!'.

If you have something more in mind, discuss it in the issue tracker or on [this](https://groups.google.com/forum/#!topic/sciruby-dev/OJxhrGG309o) thread.

## Documentation

There are two ways in which NMatrix is being documented: guides and comments, which are converted with RDoc into the
documentation seen in [sciruby.com](http://sciruby.com).

If you want to write a guide on how to use NMatrix to solve some problem or simply showing how to use one of its
features, write it as a wiki page and send an e-mail on the [mailing list][1]. We're working to improve this process.

If you aren't familiar with RDoc syntax,
[this is the official documentation](http://docs.seattlerb.org/rdoc/RDoc/Markup.html).

## Conclusion

This guide was heavily based on the
[Contributing to Ruby on Rails guide](http://edgeguides.rubyonrails.org/contributing_to_ruby_on_rails.html).

[1]: https://groups.google.com/forum/?fromgroups#!forum/sciruby-dev
[2]: https://github.com/sciruby/nmatrix/issues?sort=created&state=open
