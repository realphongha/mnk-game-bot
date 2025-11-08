import numpy
from setuptools import setup
from Cython.Build import cythonize

setup(
    packages=["board_state"],
    ext_modules=cythonize([
        "board_state/mnk_board.pyx",
        "board_state/mnk_state.pyx",
    ], annotate=True),
    include_dirs=[numpy.get_include()],
)
