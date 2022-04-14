cuda-wrapper
============

Lightweight C++11 CUDA API wrapper
----------------------------------

This library is a header-only, stateless and lightweight wrapper for the CUDA driver and runtime libraries.
It wraps the C and C++ CUDA functions calls in easy to use C++ STL like objects.

The library was originally created for the `HALMD <http://halmd.org/>`_ project but can be used as a standalone library.

Unfortunately, there is no documentation or examples (contributions welcome), *but* there are pretty detailed
unit tests so you may have a look at those.

Features
--------

* error handling using exceptions
* device managment
* allocator
* device and host vectors (with automatic allocation and iterators)
* copy functions (using the vector iterators)
* kernel configuration and launch
* streams
* events
* symbols
* textures using the texture object API

Usage example
-------------

A minimal working example with CUDA code and C++ host code in different
compilation units can be found at `examples/minimal <examples/minimal>`_.

Requirements
------------

* CUDA ≥ 7.0 support
* C++11 compiler
* CMake ≥ 2.8.12

Installation
------------

cuda-wrapper uses CMake, the headers are just installed into the directory specified to CMake::

  $ cmake path/to/cuda-wrapper
  $ make
  $ make install

Unit Tests
----------

cuda-wrapper uses the Boost testing framework, if it is not found unit tests are disabled.
To start testing, just run (after building)::

  $ ctest

Why use both the driver and the runtime library?
------------------------------------------------

We initially intended to completely switch to the driver library because it
offers much finer control over the devices and contexts than the runtime. The
problem with the driver API are functions and symbols: You need to load an
already compiled ptx or cubin file at runtime and than reference a function or
symbol with a string. This would be very unclean for a library so some
functions still use the runtime.
