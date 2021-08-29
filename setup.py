import os
import sys
import pathlib

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        cwd = pathlib.Path().absolute()

        build_temp = pathlib.Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        extdir = pathlib.Path(self.get_ext_fullpath(ext.name))

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}".format(str(extdir.parent.absolute())),
            "-DPYTHON_EXECUTABLE={}".format(sys.executable),
            "-DCMAKE_BUILD_TYPE={}".format(cfg),
            "-DBUILD_PCS_TESTS=OFF",
            "-DBUILD_PCS_COVERAGE=OFF",
            "-DBUILD_PCS_PYTHON_BINDINGS=ON",
        ]

        build_args = ["--"]
        if self.parallel:
            build_args += ["-j{}".format(self.parallel)]

        os.chdir(str(build_temp))
        self.spawn(["cmake", str(cwd)] + cmake_args)
        if not self.dry_run:
            self.spawn(["cmake", "--build", "."] + build_args)
        os.chdir(str(cwd))


setup(
    name="pypcs",
    version="1.0",
    author="Alexandr Garkusha",
    description="Point cloud semantic segmentation library",
    ext_modules=[CMakeExtension("pypcs")],
    cmdclass={
        "build_ext": CMakeBuild,
    },
)
