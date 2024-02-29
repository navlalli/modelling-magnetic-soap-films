[![CC BY 4.0][cc-by-shield]][cc-by]

# Thin film modelling of magnetic soap films

The packages required and their version numbers can be found in [`conda-env.yml`](conda-env.yml).
The python files and bash scripts in this project can be run with an active conda environment created from [`conda-env.yml`](conda-env.yml):
```
conda env create -f conda-env.yml
```

## Mesh generation

Mesh generation with [gmsh](https://gmsh.info/). The mesh files and scripts used are in the [`mesh`](mesh) directory.

## Magnetic field intensity

The magnetic field produced by the neodymium magnet used in the experiments was computed using [Magpylib](https://magpylib.readthedocs.io/en/latest/), with code provided in the [`Hfield`](Hfield) directory.

## Simulation files

The system of partial differential equations governing the film thickness field was solved using the Galerkin finite element method with [FEniCSx](https://fenicsproject.org/). The following files were used to solve the film thickness field over time:

* [`soap_unit.py`](soap_unit.py) - a circular soap film with dimensionless dimensions of unit size

* [`soap_exp.py`](soap_exp.py) - a circular soap film with dimensions aligning with the experiments

* [`mag_unit.py`](mag_unit.py) - a circular magnetic soap film with dimensionless dimensions of unit size under the forcing of an inhomogeneous magnetic field

* [`mag_exp.py`](mag_exp.py) - a circular magnetic soap film with dimensions aligning with the experiments under the forcing of an inhomogeneous magnetic field

* [`soap_vertical.py`](soap_vertical.py) - a soap film bounded by a vertical, rectangular frame 

All simulation files can be run using the shell script [`run_sim.sh`](run_sim.sh).

## License

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
