# Traceon

Traceon is a general software package used for numerical electron optics. Its main feature is the implementation of the Boundary Element Method (BEM) to quickly calculate the surface charge distribution. The program supports both radial symmetry and general three-dimensional geometries. Electron tracing can be done very quickly using accurate radial series interpolation in both geometries. The electron trajectories obtained can help determine the aberrations of the optical components under study.

The core of Traceon is completely free to use and open source. There is a commerical upgrade with more capabilities. The source code in this repository is distributed under the `MPL 2.0` license.

## Documentation

[Website](https://traceon.org/)

[Examples](https://github.com/leon-vv/Traceon/tree/main/examples)

[API documentation v0.8.0](https://traceon.org/docs/v0.8.0/index.html)

[API documentation v0.7.3](https://traceon.org/docs/v0.7.3/index.html)

[API documentation v0.6.0](https://traceon.org/docs/v0.6.0/index.html)

## Citation

Please cite the software as follows:

```
L.B. van Velzen. Traceon software (version 0.7.0). 2024. https://doi.org/10.5281/zenodo.14176070
```

## Installation

Install using the Python package manager:
```
pip install traceon
```

The installation is known to work on Linux, Windows and Mac OS. Please reach out to me if you have any installation problems.

## Validations

To ensure the accuracy of the package, different problems from the literature have been analyzed using this software. See [/validation](https://github.com/leon-vv/Traceon/tree/main/validation) directory for more information. The validations can easily be executed from the command line, for example:
```bash
python3 ./validation/edwards2007.py --help
python3 ./validation/capacitance-sphere.py --help
etc...
```

## License

[MPL 2.0](https://mozilla.org/MPL/2.0/)


## Help! I have a problem!

Don't worry. You can reach me.

[Open an issue](https://github.com/leon-vv/Traceon/issues)

[Send me an email](mailto:leonvanvelzen@protonmail.com)

## Gallery

![Image of 3D deflector](https://raw.githubusercontent.com/leon-vv/traceon/main/images/deflector-image.png)
![Image of Dohi mirror](https://raw.githubusercontent.com/leon-vv/traceon/main/images/dohi-mirror.png)
![Image of Einzel lens traces](https://raw.githubusercontent.com/leon-vv/traceon/main/images/einzel-lens-traces.png)

## Release notes

### v0.8.0
- New plotting module (charge density, equipotential lines)
- Automatic orientation of normal vectors
- Geometry functions for extruding/revolving edges of surfaces
- Tracing of particles other than electrons (charge, mass as input)
- Various bug fixes and improvements

**Breaking changes**:
- Call `P.show()` after `P.plot_mesh()` to show figures
- Normal vectors should be oriented automatically (please check if this works correctly for your geometry)
<br />

### v0.7.0
- Generate structured, high quality meshes using the new parametric mesher (drop GMSH)
- Consistenly use 3D points and geometries throughout codebase
- Add support for Fast Multipole Method (Traceon Pro)
- Add support for Mac OS on x64
- Big improvements to code quality, testing, infrastructure
- Drop dependency on GSL

### v0.6.0:
- New methods to integrate triangle potential and field contribution over a triangle
- Fix 3D convergence issues by more accurately calculating neighbouring triangle interactions
- Fix error calculation in particle tracing
- Introduce logging module to control verbosity of printing
- Clean up unit tests
- Remove higher order (curved) triangle support, in preparation of parametric meshers and improved FFM implementation

### v0.5.0:
- Add preliminary support for magnetostatics
- Improve and generalize mesh class (allow import/export)
- Make consistent use of SI units

### v0.4.0:
- Introduce Fast Multipole Method (FMM) for large 3D problems
- Compute 3D radial expansion coefficients using analytical formulas
- Further speed up computation of 3D radial expansion coefficients 
- Big code quality improvement of validation/ files

### v0.3.0:
- Use adaptive integration using GNU Scientific Library (GSL)
- Add support for boundary constraint
- Use [Vedo](https://vedo.embl.es/) for better plotting capabilities
- Use higher order triangle elements for 3D (curved triangles)
- Precompute jacobians/positions for better performance
- First implementation of element splitting based on charges (work in progress)

### v0.2.0:
- Use higher order charge distribution on line elements in radial symmetry
- Use higher order line elements (polynomials) in radial symmetry
- Better integration techniques, especially with regards to the logarithmic singularities

### v0.1.0:
- Uses the powerful [GMSH library](https://gmsh.info/) for meshing
- Solve for surface charge distribution using BEM
- General 3D geometries and radially symmetric geometries
- Dielectrics
- Floating conductors
- Accurate electron tracing using adaptive time steps
- Field/potential calculation by integration over surface charges
- Fast field/potential calculation by radial series expansion
- Superposition of electrostatic fields


