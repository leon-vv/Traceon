# Traceon
Traceon is a Python library designed for the modeling of electron microscopes.

Traceon features a Boundary Element Method (BEM) solver that computes electrostatic and magnetostatic fields for accurate particle tracing. By relying on the BEM rather than FEM, Traceon offers notable improvements in speed and accuracy compared to most commercial alternatives. It supports both radially symmetric and fully 3D geometries, while leveraging advanced axial field interpolations to greatly speed up the particle tracing.

### Wait, is this commercial?

The core of Traceon is completely free to use and open source, and distributed under the `MPL 2.0` license. The software downloaded when using `pip install traceon` does not include any closed source software. There is a commercial package called `traceon_pro` which builds on the `traceon` package to provide a more capable library.

| Feature | Traceon | Traceon Pro |
| --- | :---: | :---: |
| Parametric mesher  | ✅ | ✅|    
| Plot module | ✅ | ✅ |
| Radial symmetric solver (electrostatic, magnetostatic) | ✅ | ✅ |
| Radial symmetric particle tracer | ✅ | ✅ |
| Radial symmetric axial interpolation (fast tracing) | ✅ | ✅|
| 3D direct solver (electrostatic, magnetostatic) | | ✅|
| 3D solver using fast multipole method  | | ✅|
| 3D particle tracing | | ✅|
| 3D axial interpolation (fast tracing) | | ✅|
| Coulomb interaction | | ✅|



## Documentation

[Website](https://traceon.org/)

[Examples](https://github.com/leon-vv/Traceon/tree/main/examples)

[API documentation v0.10.0](https://traceon.org/docs/v0.10.0/traceon/index.html)

[API documentation v0.9.0](https://traceon.org/docs/v0.9.0/traceon/index.html)

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

### v0.10.0
- Make Field a GeometricObject (which can be moved, rotated, etc)
- Introduce FieldSuperposition class to represent summation of arbitrary fields
- Add type hints throughout codebase
- Add initial support for couloumb interactions while tracing (pro only)

**Breaking changes**
- `Tracer.__call__` now takes velocities with unit m/s instead of eV

### v0.9.0
- Add permanent magnets
- Small improvements to geometry module (add annulus methods)
- More general rotations for GeometricObject
- Various bug fixes and improvements

**Breaking changes**
- `FieldRadialAxial` was moved from `solver.py` to `field.py`

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


