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
    compiler_kwargs = dict(extra_compile_args=['/fp:fast', '/Ox', '/Ob3', '/Oi', '/GL', '/arch:AVX', '-I .\\traceon\\backend\\'])
    extra_objects = []
    include_dirs = []


backend_extension = Extension(
    name='traceon.backend.traceon_backend',
    sources=['traceon/backend/traceon-backend.c'],
    extra_objects=extra_objects,
    py_limited_api=True,
    **compiler_kwargs)

setup(
    name='traceon',
    version='0.7.0rc2',
    description='Solver and tracer for electrostatic problems',
    url='https://github.com/leon-vv/Traceon',
    author='Léon van Velzen',
    author_email='leonvanvelzen@protonmail.com',
    keywords=['boundary element method', 'BEM', 'electrostatic', 'electromagnetic', 'electron microscope', 'electron', 'tracing', 'particle', 'tracer', 'electron optics'],
    license='AGPLv3',
    ext_modules=[backend_extension],
    packages=['traceon', 'traceon.backend'],
    package_data={
        'traceon.backend': ['*.c']
    },
    long_description = open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=['matplotlib', 'vedo', 'numpy', 'scipy', 'meshio'],
    project_urls = {
        'Documentation': "https://leon.science/traceon",
        'Code': "https://github.com/leon-vv/traceon",
        'Issues': "https://github.com/leon-vv/traceon/issues"
    },
    include_dirs=include_dirs,
    python_requires='>=3.7',
    options={'bdist_wheel': {'py_limited_api': 'cp37'}},
)




