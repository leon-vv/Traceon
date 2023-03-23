# Traceon

Traceon is a general software package used for numerical electron optics. Its main feature is the implementation of the Boundary Element Method (BEM) to quickly calculate the surface charge distribution. The program supports both radial symmetry and general three-dimensional geometries. Electron tracing can be done very quickly using accurate radial series interpolation in both geometries. The electron trajectories obtained can help determine the aberrations of the optical components under study.

Traceon is completely free to use and open source. The source code is distributed under the `AGPLv3` license.

## Documentation

[Examples](https://github.com/leon-vv/Traceon/tree/main/examples)

[API documentation](https://leon.science/traceon/index.html)

## Validations

To ensure the accuracy of the package, different problems from the literature have been analyzed using this software. See [/validation](https://github.com/leon-vv/Traceon/tree/main/validation) directory for more information. The validations can easily be executed from the command line, for example:
```bash
python3 ./validation/edwards2007.py --help
python3 ./validation/capacitance-sphere.py --help
etc...
```

## License

[AGPLv3](https://www.gnu.org/licenses/agpl-3.0.en.html)

## Installation

Install using the Python package manager:
```
pip install traceon
```

The installation is known to work on Linux and Windows. Please reach out to me if you have any installation problems.

## Help! I have a problem!

Don't worry. You can reach me.

[Open an issue](https://github.com/leon-vv/Traceon/issues)

[Send me an email](mailto:leonvanvelzen@protonmail.com)

## Gallery

![Image of 3D deflector](https://raw.githubusercontent.com/leon-vv/traceon/main/images/deflector-image.png)

## Features

- Uses the powerful [GMSH library](https://gmsh.info/) for meshing
- Solve for surface charge distribution using BEM
- General 3D geometries and radially symmetric geometries
- Dielectrics
- Floating conductors
- Accurate electron tracing using adaptive time steps
- Field/potential calculation by integration over surface charges
- Fast field/potential calculation by radial series expansion
- Superposition of electrostatic fields


