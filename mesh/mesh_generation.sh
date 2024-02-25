#!/bin/bash

# Create mesh from .geo file and convert to .xdmf and .h5 for use in fenicsx

mesh_name="radial_exp"
ID="195443"

conda activate fox06

# Create .msh file from .geo file
cd ${mesh_name}${ID}
gmsh ${mesh_name}.geo
cd ../

# Perform mesh conversion
python3 ./mesh_converter.py ${mesh_name} ${ID}
