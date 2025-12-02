# Development Guidelines

This document is for developers of PyKX, and strives to clarify some otherwise unspoken assumptions regarding PyKX, and to provide some guidelines for its development.


## Goals

PyKX aims to provide a Python-first interface to q. This means that the API should be as Pythonic as possible while ideally still being able to efficiently perform any task that one could be done with q itself.

The primary target audience of this package is naive Python users. These would be data scientists, and the majority of programmers who have at least a passing familiarity with Python. These users will rely heavily on the documentation, and examples. They will not be familiar with any advanced Python concepts, or even have a working knowledge of the [Python object model](https://docs.python.org/3/reference/datamodel.html). Additionally, and more importantly for any developers of PyKX who are primarily q programmers, these users will not be familiar with any q-isms, or vector languages in general. The closest thing most of the will have experienced to a vector language is the Python package [Numpy](https://numpy.org/), and as such we can expect that many of them will use PyKX primarily as a way to convert query results from their q database to Numpy arrays.

Other kinds of users who should also be considered include skilled q users, and skilled Python users. For them we try to include extra opt-in features which are entirely unnecessary to get the core experience of PyKX. For example the q console is, in part, provided for skilled q users, and it does not follow any Python conventions.


## What does Pythonic means?

The term "Pythonic" is vague, and is not standardized in Python or any of its documentation (unlike [The Zen of Python](https://www.python.org/dev/peps/pep-0020/); see `import this`), but it is widely used by the python community to describe code that follows Python-specific best practices, community recommendations, standardized style suggestions, and idioms. Pythonic code makes use of the features of the language in an effort to produce code which is clear, concise, and maintainable.

For example, instead of writing code to take the sum of a range of numbers like this:

```Python
a = 10
b = 1000
total = 0

while b >= a:
    total += a
    a += 1
```

Take advantage of the `sum` and `range` builtin functions to write it like this:

```Python
total = sum(range(10, 1001))
```

Instead of writing code to double every even value in a list like this:

```Python
arr = [1, 2, 3, 4, 5, 6]
length = len(arr)

for i in range(0, length):
    if arr[i] % 2 == 0:
        arr[i] *= 2
```

Take advantage of list comprehensions to write it like this:

```Python
arr = [x if x % 2 else x * 2 for x in range(1, 7)]
```

In addition to taking advantage of language features that can serve to make the code clearer, being Pythonic also involves following the widely accepted standardized conventions such as [PEP 8 (Style Guide for Python Code)](https://www.python.org/dev/peps/pep-0008/), [PEP 257 (Docstring Conventions)](https://www.python.org/dev/peps/pep-0257/), and [PEP 484 (Type Hints)](https://www.python.org/dev/peps/pep-0484/).

Additional Pythonic practices exist concerning the Python type system. Some of these are:
  - Use `isinstance(x, y)` instead of `type(x) is y`. This allows for instances of subclasses of `y` to return `True`.
  - Use `issubclass(x, y)` instead of `x is y`. This allows for subclasses of `y` to return `True`.
  - Use `super()` to delegate to a class higher up the inheritance tree, rather than referring to the higher class directly. This allows for the later insertion of classes in between the class and its super-class without any changes to the lower class.
  - Avoid type checking where possible; rely on [duck-typing](https://docs.python.org/3/glossary.html#term-duck-typing) instead.
  - When type checking is required, check against the standard [abstract base classes](https://docs.python.org/3/library/collections.abc.html) where applicable. These implement what's known as "virtual subclasses" by hooking into the mechanism behind `issubclass` to make classes which do not inherit them still match against them. For example, for any class `x` which implements `__contains__`, `__iter__`, and `__len__`, `issubclass(x, collections.abc.Collection) is True`.
  - When delegating the methods (particularly the `__init__` method) to the super-class, have the method accept and pass on arbitrary arguments to ensure future changes to the signature of the method do not break the delegation chain. This can be accomplished by having `*args` and `**kwargs` in the parameter list, and then calling the super-class method with `*args` and `**kwargs`.

Other miscellaneous Pythonic tips include:
  - Prefer `try` over `if` blocks where reasonable. See [EAFP](https://docs.python.org/3/glossary.html#term-eafp). For example, instead of checking if a key in a dictionary is present with an `if` statement prior to accessing it, one should access the dictionary within a `try` block, and handle the `KeyError` should it arise.
  - Avoid extracting a sequence from an iterator without good reason. Iterators can produce infinite sequences (which will freeze the process). Additionally it can be a waste of memory to extract a sequence, as it's common for less than the entire output of an iterator to be consumed.
  - Do not rely on CPython-specific behavior, such as the builtin function `id` returning the address of the object in memory.
  - Use context managers (i.e. `with` blocks) whenever possible. These blocks have block-entry and block-exit behaviors which are guaranteed to run, even if an exception is raised
  - Use the `is` keyword rather than the `==` operator to compare against classes and singletons (e.g. `True`, `False`, and `None`).
  - Assume other developers and end-users will be responsible: do not attempt to make variables private. To signal that a variable is not *intended* for public use, its name should be preceded by an underscore. No further obfuscation should be attempted.
  - Use an underscore (or double underscore) as a variable name when the variable name is required syntactically, but will be ignored. For instance, when unpacking a 3 element tuple where only the first and last arguments are needed, it can be written like so: `x, _, y = triple`. Another common case where this comes up is when using a for-loop over a range when the loop index isn't used, in which case the loop can be written as `for _ in range(x): ...`.
  - Prefer list/set/dict/generator comprehensions over for-loops so long as the resulting comprehension isn't too complex or too long.


## API Development Considerations

It's easy to provide new functions/methods, decide on argument order, calling conventions, etc., but it's terribly difficult to change those decisions after PyKX has been released. For this reason we should err on the side of not releasing such changes until we're confident that we have made the best decision, and that we probably won't want to change it in the future.


## Cython Tips

Cython is extremely valuable as it saves us from having to write (much) with the Python C-API. Since Cython is a super-set language of Python we can stick to Python wherever convenient, and only use Cython when we need to interface with C, or when greater performance is needed. This keeps the implementation small and simple, which goes a long way to reducing bugs.

All that said, Cython is not without its problems. It takes a relatively long time to compile the C files generated by Cython. The error messages produced by the Cython transpiler are often less than clear, and even worse is when the C compiler throws an error when compiling a file that was generated by Cython. In some cases Cython behaves subtly different from how Python does.

For these reasons we make a point to only use Cython when the benefits are significant. If a class need not be an extension type (i.e. a "cdef class"), then it shouldn't be. If a file only contains pure Python code, then it should not be a Cython file.

To deal with compilation errors in the Cython-generated C files, go to the line in the C file that the compiler threw the error at. Not far above that line there should be some comments that contain the section of the Cython source code that was used to generate the erroneous C code. From that you should be able to edit the Cython code to amend the error. Do not edit the generated C file, and do not check the generated C files into version control.
