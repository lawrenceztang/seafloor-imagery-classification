from distutils.core import setup
from Cython.Build import cythonize
import numpy
import pycocotools

setup(name='mask-rcnn',
      ext_modules=cythonize("pycocotools/_mask.pyx"), include_dirs=[numpy.get_include()])