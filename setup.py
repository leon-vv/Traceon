from setuptools import setup, Extension
import platform

VCPKG_DIR = 'C:\\Users\\leonv\\vcpkg'

if platform.system() in ['Linux', 'Darwin']:
    compiler_kwargs = dict(extra_compile_args=['-O3', '-mavx', '-ffast-math', '-DNDEBUG'], extra_link_args=['-lm', '-lgsl'])
    extra_objects = []
    include_dirs = []
elif platform.system() == 'Windows':
    compiler_kwargs = dict(extra_compile_args=['/fp:fast', '/Ox', '/Ob3', '/Oi', '/GL', '/arch:AVX', '-I .\\traceon\\backend\\'])
    extra_objects = [VCPKG_DIR + '\\packages\\gsl_x64-windows-static-release\\lib\\gsl.lib']
    include_dirs = VCPKG_DIR + '\\packages\\gsl_x64-windows-static-release\\include'


backend_extension = Extension(
    name='traceon.backend.traceon_backend',
    sources=['traceon/backend/traceon-backend.c'],
    extra_objects=extra_objects,
    **compiler_kwargs)

setup(
    name='traceon',
    version='0.4.0',
    description='Solver and tracer for electrostatic problems',
    url='https://github.com/leon-vv/Traceon',
    author='Léon van Velzen',
    author_email='leonvanvelzen@protonmail.com',
    keywords=['boundary element method', 'BEM', 'electrostatic', 'electromagnetic', 'electron microscope', 'electron', 'tracing', 'particle', 'tracer', 'electron optics'],
    license='AGPLv3',
    ext_modules=[backend_extension],
    packages=['traceon', 'traceon.backend'],
    long_description = open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=['matplotlib', 'vedo', 'numpy', 'gmsh>=4.9', 'pygmsh>=7.1.13', 'scipy'],
    project_urls = {
        'Documentation': "https://leon.science/traceon",
        'Code': "https://github.com/leon-vv/traceon",
        'Issues': "https://github.com/leon-vv/traceon/issues"
    },
    include_dirs=include_dirs,
)




