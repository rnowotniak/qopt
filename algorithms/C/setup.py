#!/usr/bin/python

from distutils.core import setup, Extension

algorithms_module = Extension('_algorithms',
                           sources=['algorithms_wrap.cxx', 'rQIEA.cpp', 'framework.cpp'],
                           )

setup (name = 'algorithms',
       version = '0.1',
       author      = "SWIG Docs",
       description = """Simple swig algorithms from docs""",
       ext_modules = [algorithms_module],
       py_modules = ["algorithms"],
       )

