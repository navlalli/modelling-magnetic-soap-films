#!/bin/bash

# Create mesh from .geo file and convert to .xdmf and .h5 for use in fenicsx

mesh_name="radial_unit"
ID="15046"

# Create .msh file from .geo file - requires the correct conda environment to be
# active
cd ${mesh_name}${ID}
gmsh ${mesh_name}.geo
cd ../

# Mesh conversion
python3 ./mesh_converter.py ${mesh_name} ${ID}
