from typing import Any, Dict

from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension('mxfold2.interface',
        [
            "mxfold2/src/interface.cpp",
            "mxfold2/src/param/turner.cpp",  #param/turner.h 
            "mxfold2/src/param/positional.cpp", #param/positional.h 
            "mxfold2/src/param/bpscore.cpp", #param/bpscore.h     
            "mxfold2/src/param/mix.cpp", #param/mix.h
            #"mxfold2/src/param/util.h"
            "mxfold2/src/fold/fold.cpp", #fold/fold.h 
            "mxfold2/src/fold/zuker.cpp", #fold/zuker.h 
            "mxfold2/src/fold/nussinov.cpp", #fold/nussinov.h 
            #"mxfold2/src/fold/trimatrix.h
        ]
    )
]

def build(setup_kwargs: Dict[str, Any]) -> None:
    setup_kwargs.update(
        {
            "ext_modules": ext_modules,
            "cmdclass": {"build_ext": build_ext},
            "zip_safe": False,
        }
    )
