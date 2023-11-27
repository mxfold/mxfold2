from glob import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

def build(setup_kwargs):
    ext_modules = [
        Pybind11Extension(
            "mxfold2.interface",
            sorted(glob("mxfold2/src/**/*.cpp", recursive=True)),
        ),
    ]
    setup_kwargs.update(
        {
            "ext_modules": ext_modules,
            "cmdclass": dict(build_ext=build_ext),
            "zip_safe": False,
        }
    )
