import os

from setuptools import setup, Extension
import platform

if platform.system() == 'Linux':
    os.environ['CC'] = 'clang' # Clang is known to produce faster binaries.
    compiler_kwargs = dict(extra_compile_args=['-O3', '-mavx', '-ffast-math', '-DNDEBUG'], extra_link_args=['-lm'])
    extra_objects = []
    include_dirs = []
elif platform.system() == 'Darwin':
    os.environ['CC'] = 'clang' # Clang is known to produce faster binaries.
    compiler_kwargs = dict(extra_compile_args=['-O3', '-mavx', '-ffast-math', '-DNDEBUG'], extra_link_args=['-lm', '-L/opt/homebrew/lib'])
    extra_objects = []
    include_dirs = ['/opt/homebrew/include/']
elif platform.system() == 'Windows':
    compiler_kwargs = dict(extra_compile_args=['/fp:fast', '/Ox', '/Ob3', '/Oi', '/GL', '/arch:AVX', '-I .\\voltrace\\backend\\'])
    extra_objects = []
    include_dirs = []


backend_extension = Extension(
    name='voltrace.backend.voltrace_backend',
    sources=['voltrace/backend/voltrace-backend.c'],
    extra_objects=extra_objects,
    py_limited_api=True,
    **compiler_kwargs)

setup(
    name='voltrace',
    version='0.10.0',
    description='Solver and tracer for electrostatic problems',
    url='https://github.com/leon-vv/Voltrace',
    author='Léon van Velzen',
    author_email='leonvanvelzen@protonmail.com',
    keywords=['boundary element method', 'BEM', 'electrostatic', 'electromagnetic', 'electron microscope', 'electron', 'tracing', 'particle', 'tracer', 'electron optics'],
    license='MPL 2.0',
    ext_modules=[backend_extension],
    packages=['voltrace', 'voltrace.backend'],
    package_data={
        'voltrace.backend': ['*.c']
    },
    long_description = open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=['matplotlib', 'vedo', 'numpy', 'scipy', 'meshio'],
    project_urls = {
        'Documentation': "https://leon.science/voltrace",
        'Code': "https://github.com/leon-vv/voltrace",
        'Issues': "https://github.com/leon-vv/voltrace/issues"
    },
    include_dirs=include_dirs,
    python_requires='>=3.7',
    options={'bdist_wheel': {'py_limited_api': 'cp37'}},
)




