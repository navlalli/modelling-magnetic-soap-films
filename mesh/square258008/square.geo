// Square mesh
SetFactory("OpenCASCADE");

// Key dimensions
l_cap = 2;  // Capillary length (mm)
l_side = 40.0;  // Side length (mm)
nond_side = l_side / l_cap;
mesh_size = 0.06;

// Create points of rectangle
Point(1) = {0, 0, 0, mesh_size};
Point(2) = {0, nond_side, 0, mesh_size};
Point(3) = {nond_side, 0, 0, mesh_size};
Point(4) = {nond_side, nond_side, 0, mesh_size};

// Create lines of rectangle 
Line(1) = {1, 3};
Line(2) = {3, 4};
Line(3) = {4, 2};
Line(4) = {2, 1};

// Create curve loop from lines
Curve Loop(1) = {1, 2, 3, 4};

// Create surface from curve loops
Plane Surface(1) = {1};

// Create physical groups
Physical Curve("0", 2) = {1};
Physical Surface("0", 3) = {1};

// Mesh and save
Mesh 2;
Save "square.msh";
