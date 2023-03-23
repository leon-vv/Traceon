from setuptools import setup, Extension
import platform


if platform.system() in ['Linux', 'Darwin']:
    compiler_kwargs = dict(extra_compile_args=['-O3', '-mavx', '-ffast-math'], extra_link_args=['-lm'])
elif platform.system() == 'Windows':
    compiler_kwargs = dict(extra_compile_args=['/fp:fast', '/Ox', '/Ob3', '/Oi', '/GL', '/arch:AVX'])


backend_extension = Extension(
    name='traceon.backend.traceon_backend',
    sources=['traceon/backend/traceon-backend.c'],
    **compiler_kwargs)

setup(
    name='traceon',
    version='0.1.2',
    description='Solver and tracer for electrostatic problems',
    url='https://github.com/leon-vv/Traceon',
    author='Léon van Velzen',
    author_email='leonvanvelzen@protonmail.com',
    keywords=['boundary element method', 'BEM', 'electrostatic', 'electromagnetic', 'electron microscope', 'electron', 'tracing', 'particle', 'tracer', 'electron optics'],
    license='AGPLv3',
    ext_modules=[backend_extension],
    packages=['traceon', 'traceon.backend'],
    include_package_data=True,
    long_description = open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=['matplotlib', 'numpy', 'gmsh>=4.9', 'pygmsh>=7.1.13', 'scipy'],
    project_urls = {
        'Documentation': "https://leon.science/traceon",
        'Code': "https://github.com/leon-vv/traceon",
        'Issues': "https://github.com/leon-vv/traceon/issues"
    }
)




