#!/bin/bash

case_name="soap_unit"
case_ID="001"

sol_dir="sol_${case_name}/"
case_dir="${sol_dir}${case_ID}/"

mkdir -p "${case_dir}"

# Create snapshot of simulation file
cp -v "${case_name}.py" "${case_dir}${case_name}${case_ID}.py"

# Run simulation - requires the correct conda environment to be active
ncores=16
mpirun -np ${ncores} python3 -u ${case_name}.py ${case_ID} | tee ${case_dir}log.txt
