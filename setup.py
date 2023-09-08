import os
import sys
from setuptools import setup

FABM_BASE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "extern/fabm")
sys.path.insert(0, os.path.join(FABM_BASE, "src/drivers/python"))

from build_cmake import CMakeExtension, bdist_wheel, CMakeBuild

setup(
    packages=["fabmos", "fabmos.transport"],
    package_dir={"": "src"},
    ext_modules=[
        CMakeExtension("fabmos.fabm_tmm", "-DPYFABM_DEFINITIONS=_FABM_DIMENSION_COUNT_=3;_FABM_DEPTH_DIMENSION_INDEX_=3;_FABM_MASK_TYPE_=integer;_FABM_MASKED_VALUE_=0;_FABM_BOTTOM_INDEX_=-1"),
    ],
    cmdclass={"bdist_wheel": bdist_wheel, "build_ext": CMakeBuild},
    zip_safe=False,
    #use_scm_version={
        #'write_to': '_version.py',
        #'write_to_template': '__version__ = "{version}"',
        #'tag_regex': r'^(?P<prefix>v)?(?P<version>[^\+]+)(?P<suffix>.*)?$',
        #'local_scheme': "dirty-tag"
    #}
)
