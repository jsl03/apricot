# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
# ------------------------------------------------------------------------------

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools


__version__ = '0.93'


class get_pybind_include(object):
    """ Helper class to determine the pybind11 include path.

    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked.
    """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)


ext_modules = [
    Extension(
        'apricot.core.gp_internal',
        ['apricot/src/gp_internal.cpp'],
        include_dirs=[
            get_pybind_include(),
            get_pybind_include(user=True),
            '/usr/include/eigen3',
            'apricot/src'
        ],
        language='c++'
    ),
]


def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14] compiler flag.
    The c++14 is prefered over c++11 (when it is available).
    """
    if has_flag(compiler, '-std=c++14'):
        return '-std=c++14'
    elif has_flag(compiler, '-std=c++11'):
        return '-std=c++11'
    else:
        raise RuntimeError(
            'Unsupported compiler -- at least C++11 support is needed!'
        )


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])

        # extra compiler options
        opts += [
            '-O3',
            '-fPIC',
            '-march=native'
        ]

        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')

        elif ct == 'msvc':
            opts.append(
                '/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version()
            )

        for ext in self.extensions:
            ext.extra_compile_args = opts

        build_ext.build_extensions(self) 


setup(
    name='apricot',
    version=__version__,
    author='Joe Loxham',
    author_email='Jloxham@googlemail.com',
    url='',
    description='',
    long_description='',
    packages=setuptools.find_packages(),
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.2','pystan','numpy'],
    include_package_data=True,
    cmdclass={'build_ext': BuildExt},
    headers=[
        'apricot/src/misc.h',
        'apricot/src/gp_eq_kernel.h',
        'apricot/src/gp_m52_kernel.h',
        'apricot/src/gp_m32_kernel.h',
        'apricot/src/gp_rq_kernel.h',
        'apricot/src/gp_eq_mle_objective.h'
    ],
    zip_safe=False,
)
