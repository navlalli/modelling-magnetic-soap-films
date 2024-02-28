// Mesh of unit dimensions with radial refinement
SetFactory("OpenCASCADE");

nond_radius = 0.5;

// Create circle
Circle(1) = {nond_radius, nond_radius, 0, nond_radius, 0, 2*Pi};

// Create curve loop from line
Curve Loop(1) = {1};

// Create surface from curve loop
Plane Surface(1) = {1};

// Create physical groups
Physical Curve("0", 2) = {1};
Physical Surface("0", 3) = {1};

// Set the following to 0 when element size is fully specified by mesh size field
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.MeshSizeExtendFromBoundary = 0;

// Set mesh size using a size field
Background Field = 1;
Field[1] = MathEval;
Field[1].F = Sprintf("-0.015*((x - %g)^2 + (y - %g)^2)^0.5 + 0.0165", nond_radius, nond_radius);

// Mesh and save
Mesh 2;
Save "radial_unit.msh";
