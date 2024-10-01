import os
import sys
from setuptools import setup

FABM_BASE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "extern/fabm")
sys.path.insert(0, os.path.join(FABM_BASE, "src/drivers/python"))

from build_cmake import CMakeExtension, bdist_wheel, CMakeBuild

setup(
    packages=["fabmos", "fabmos.transport", "fabmos.input"],
    package_dir={"": "src"},
    ext_modules=[
        CMakeExtension(
            "fabmos.fabm_tmm",
            "-DPYFABM_DEFINITIONS=_FABM_DIMENSION_COUNT_=2;_FABM_DEPTH_DIMENSION_INDEX_=2;_FABM_MASK_TYPE_=integer;_FABM_MASKED_VALUE_=0;_FABM_BOTTOM_INDEX_=-1",
        ),
        CMakeExtension(
            "fabmos.fabm_gotm",
            "-DPYFABM_DEFINITIONS=_FABM_DIMENSION_COUNT_=2;_FABM_DEPTH_DIMENSION_INDEX_=2;_FABM_VERTICAL_BOTTOM_TO_SURFACE_",
        ),
        CMakeExtension(
            "fabmos.fabm_hz_only", "-DPYFABM_DEFINITIONS=_FABM_DIMENSION_COUNT_=1"
        ),
    ],
    cmdclass={"bdist_wheel": bdist_wheel, "build_ext": CMakeBuild},
    zip_safe=False,
)
