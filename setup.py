#!/usr/bin/env python

# Copyright 2020 Jiang Shenghu
# SPDX-License-Identifier: Apache-2.0

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

import os
import shutil
import pathlib


class ExtensionISL(Extension):
    def __init__(self, sourcedir=pathlib.Path(__file__).parent / '3rdparty/isl'):
        super(ExtensionISL, self).__init__('isl', sources=[])
        self.sourcedir = pathlib.Path(sourcedir)


def _force_link(src, dst):
    src, dst = map(pathlib.Path, (src, dst))
    if dst.exists():
        os.remove(dst)
    os.link(src.resolve(), dst)


class PyvlovaExtBuild(build_ext):
    def __init__(self, *args, **kwargs):
        super(PyvlovaExtBuild, self).__init__(*args, **kwargs)

        self.pyvlova_root = pathlib.Path(__file__).parent
        self.ext_dir = self.pyvlova_root / 'pyvlova/_ext'

    def _prepare_ext(self):
        self.ext_dir.mkdir(parents=True, exist_ok=True)
        (self.ext_dir / '__init__.py').touch()

    def run(self):
        self._prepare_ext()

        for ext in self.extensions:
            if isinstance(ext, ExtensionISL):
                self.build_isl(ext)

    def build_isl(self, extension):
        self.announce('Build ISL', level=3)

        isl_root = extension.sourcedir

        files = {
            'isl.py': 'interface/isl.py',
            'libisl.so': '.libs/libisl.so',
        }

        self.spawn([str(isl_root / 'build.sh')])

        for dst, src in files.items():
            _force_link(isl_root / src, self.ext_dir / dst)


setup(
    name='pyvlova',
    version='0.1.dev1',
    description='A Simple Polyhedral Compiler for NN',
    author='Jiang Shenghu',
    author_email='ChieloNewctle@Yandex.com',
    url='https://github.com/TheGreatCold/pyvlova',
    packages=['pyvlova'],
    ext_modules=[ExtensionISL()],
    cmdclass={
        'build_ext': PyvlovaExtBuild,
    },
    include_package_data=True,
)
