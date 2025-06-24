from setuptools import Extension, setup

module = Extension("mykmeanspp", sources=['kmeansmodule.c'])
setup(name='mykmeanspp',
     version='1.0',
     description='Python wrapper for custom C extension',
     ext_modules=[module])