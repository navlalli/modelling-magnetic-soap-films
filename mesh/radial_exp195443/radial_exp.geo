// Mesh of exp dimensions with radial refinement
SetFactory("OpenCASCADE");

// Key dimensions
l_cap = 2;  // Capillary length (mm)
film_diam = 40.5;  // Film diameter (mm)
thin_film_diam = film_diam - 2 * l_cap;  // Diameter of thin film (mm)
radius = thin_film_diam / 2;  // Radius of thin film region (mm)
nond_radius = radius / l_cap;  // Non-dimensional radius

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
Field[1].F = Sprintf("-0.010*((x - %g)^2 + (y - %g)^2)^0.5 + 0.1265", nond_radius, nond_radius);

// Mesh and save
Mesh 2;
Save "radial_exp.msh";
