""" Solve thickness field over time for a vertical soap film bounded by a
rectangular frame
"""

import numpy as np
import pyvista as pv
import ufl
import time 
import sys

from dolfinx import fem, io, plot, nls, log
from ufl import ds, dx, grad, div, inner, dot, FacetNormal, Identity, exp, ln, as_vector
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from datetime import datetime

comm = MPI.COMM_WORLD
rank = comm.rank 
if rank == 0:
    print(f"N. of cores = {comm.Get_size()}")

case = "soap_vertical"
ID = sys.argv[1] if len(sys.argv) > 1 else "TMP"
case_dir = f"./sol_{case}/{ID}/"

# =============================================================================
# Pyvista plotting 
# =============================================================================
def plot_mesh(domain):
    """ Plot the mesh with pyvista """ 
    tdim = domain.topology.dim
    topology, cell_types, geometry = plot.create_vtk_mesh(domain, tdim)
    grid = pv.UnstructuredGrid(topology, cell_types, geometry)
    print(f"{grid.n_cells = }")
    print(f"{grid.n_points = }")
    p = pv.Plotter()
    p.set_background([222, 222, 222])
    p.add_mesh(grid, show_edges=True)
    p.add_axes()
    p.view_yx()
    p.camera.roll += 180
    p.camera.azimuth = 180
    p.show_grid()
    p.show()

def plot_func_mixed(func_space, arr, func_str): 
    """ Plot function from a mixed space """
    grid = pv.UnstructuredGrid(*plot.create_vtk_mesh(func_space))
    grid.point_data[func_str] = arr
    p = pv.Plotter()
    p.add_mesh(grid, show_edges=True)
    # p.add_axes()
    p.show_bounds(xlabel="", ylabel="")
    p.add_title(func_str, font_size=16)
    p.add_axes()
    p.view_yx()
    p.camera.roll += 180
    p.camera.azimuth = 180
    p.show()

# =============================================================================
# Tools 
# =============================================================================
def assembler(ufl_form):
    """ Assembles the ufl form """
    local_assembly = fem.assemble_scalar(ufl_form)
    global_assembly = comm.allreduce(local_assembly, op=MPI.SUM)
    return global_assembly     

# =============================================================================
# Mesh and function spaces 
# =============================================================================
mesh_dir = "./mesh/square258008/"
mesh_file = "square.xdmf"
Ldomain = 20.0

# Load in mesh from XDMF file
with io.XDMFFile(comm, mesh_dir + mesh_file, "r") as xdmf:
    domain = xdmf.read_mesh(name="Grid")   
tdim = domain.topology.dim
fdim = tdim - 1
x = ufl.SpatialCoordinate(domain)

# Plot mesh    
# if rank == 0:
#     plot_mesh(domain) 
# sys.exit(0)

DG0 = fem.FunctionSpace(domain, ("DG", 0))
V = fem.FunctionSpace(domain, ("CG", 1))
V2 = fem.FunctionSpace(domain, ("CG", 2))
W = fem.VectorFunctionSpace(domain, ("CG", 1))
W0 = fem.VectorFunctionSpace(domain, ("DG", 0))

hFE = ufl.FiniteElement("CG", domain.ufl_cell(), 1)
pFE = ufl.FiniteElement("CG", domain.ufl_cell(), 1)
GammaFE = ufl.FiniteElement("CG", domain.ufl_cell(), 1)
VsFE = ufl.VectorElement("CG", domain.ufl_cell(), 1)

ME = fem.FunctionSpace(domain, ufl.MixedElement([hFE, pFE, GammaFE, VsFE]))

# =============================================================================
# Define the variational problem
# =============================================================================
du = ufl.TrialFunction(ME)
v0, v1, v2, v3 = ufl.TestFunctions(ME)
u = fem.Function(ME)  # Current solution
u_n = fem.Function(ME)  # Solution from previous time step
h, p, Gamma, Vs = ufl.split(u)
h_n, p_n, Gamma_n, Vs_n = ufl.split(u_n)

epsilon = fem.Constant(domain, ScalarType(5e-3))  # Thin film parameter
Gr = fem.Constant(domain, ScalarType(1))  # Gravity number
Ma = fem.Constant(domain, ScalarType(500))  # Marangoni number
Ca = fem.Constant(domain, ScalarType(1e-7))  # Capillary number 
Bq_d = fem.Constant(domain, ScalarType(5e-4))  # Dilatational Boussinesq number
Bq_sh = fem.Constant(domain, ScalarType(5e-4))  # Shear Boussinesq number
Je = fem.Constant(domain, ScalarType(0.0))  # Evaporation

# Non-dimensional numbers relevant to surfactant transport
Lambda = fem.Constant(domain, ScalarType(0.15))
Pe_s = fem.Constant(domain, ScalarType(10))

# Time discretisation
dt = fem.Constant(domain, ScalarType(0.001))
theta = 1.0
h_mid = (1.0 - theta) * h_n + theta * h
p_mid = (1.0 - theta) * p_n + theta * p
Gamma_mid = (1.0 - theta) * Gamma_n + theta * Gamma
Vs_mid = (1.0 - theta) * Vs_n + theta * Vs

# Non-dimensional terms relevant to boundary conditions
n = FacetNormal(domain)
P = fem.Constant(domain, ScalarType(5.0))  # Capillary suction strength
hn = fem.Constant(domain, ScalarType(0.0))  # partial h / partial nd
hn.value = np.tan(0.0 * np.pi / 180)

A = 8.3e-6
B = 4.2e6
D = 3.0e3
def Pi(h_t):
    """ Disjoining pressure """
    return - A / h_t**3 + B * exp(- D * h_t)

def gamma(Gamma_t):
    """ Surface tension as a function of interfacial surfactant concentration """
    return 1 + Lambda * ln(1 - Gamma_t)

def pre_curv(Gamma_t):
    """ Strength of the curvature related term in pressure equation """
    return (epsilon**2 * Ma * gamma(Gamma_t) + epsilon**3 / Ca)

def surf_rheo(Vs_t):
    """ Tensor associated with surface rheology """
    return Bq_d * div(Vs_t) * Identity(2) + Bq_sh * grad(Vs_t)

# Weak form: note that p is thermodynamic pressure + conjoining pressure
L0 = h * v0 * dx - h_n * v0 * dx \
     + dt * div(Vs_mid * h_mid) * v0 * dx \
     + dt * 1/3 * h_mid**3 * dot(grad(p_mid), grad(v0)) * dx \
     + dt * 1/3 * Gr * div(as_vector([h_mid**3, 0])) * v0 * dx \
     + dt * Je * v0 * dx \
     + dt * 1/3 * h_mid**3 * dot(grad(Pi(h_mid)), n) * v0 * ds \
     + dt * P * h_mid**3 * v0 * ds

L1 = p * v1 * dx - pre_curv(Gamma) * dot(grad(h), grad(v1)) * dx + Pi(h) * v1 * dx \
     + pre_curv(Gamma) * hn * v1 * ds

L2 = Gamma * v2 * dx - Gamma_n * v2 * dx \
     - dt * Gamma_mid * dot(Vs_mid, grad(v2)) * dx \
     + dt / Pe_s / (1 - Gamma_mid) * dot(grad(Gamma_mid), grad(v2)) * dx
     
L3 = - inner(surf_rheo(Vs), grad(v3)) * dx + Ma * dot(grad(gamma(Gamma)), v3) * dx \
     - h * dot(grad(p), v3) * dx + h * Gr * dot(as_vector([1, 0]), v3) * dx

L = L0 + L1 + L2 + L3

# =============================================================================
# Initial condition
# =============================================================================
u.x.array[:] = 0.0
u_n.x.array[:] = 0.0

# Initial h
h_initial = 0.5
# Uniform initial h 
u.sub(0).interpolate(lambda x: np.full(x.shape[1], h_initial))
u_n.sub(0).interpolate(lambda x: np.full(x.shape[1], h_initial))
# Uniform initial p
p_initial = 0.0
u.sub(1).interpolate(lambda x: np.full(x.shape[1], p_initial))
u_n.sub(1).interpolate(lambda x: np.full(x.shape[1], p_initial))
# Uniform initial Gamma
Gamma_initial = 0.1
u.sub(2).interpolate(lambda x: np.full(x.shape[1], Gamma_initial))
u_n.sub(2).interpolate(lambda x: np.full(x.shape[1], Gamma_initial))
# Uniform initial Vs
Vs_initial = 0.0
u.sub(3).interpolate(lambda x: (np.full(x.shape[1], Vs_initial),
                                np.full(x.shape[1], Vs_initial)))
u_n.sub(3).interpolate(lambda x: (np.full(x.shape[1], Vs_initial),
                                  np.full(x.shape[1], Vs_initial)))
u.x.scatter_forward()

# # Plot initial condition for h
# if rank == 0:
#     V_h, dofs_h = ME.sub(0).collapse()
#     plot_func_mixed(V_h, u.x.array[dofs_h].real, "h") 

# =============================================================================
# Solver
# =============================================================================
a = ufl.derivative(L, u, du)
bcs = []
problem = fem.petsc.NonlinearProblem(L, u, bcs, a) 
solver = nls.petsc.NewtonSolver(comm, problem)
solver.convergence_criterion = "residual"
solver.max_it = 50
solver.rtol = 1e-10
solver.atol = 1e-10
solver.report = True

log.set_log_level(log.LogLevel.WARNING)  # INFO, WARNING, ERROR, OFF
# Save all logging to file
log.set_output_file(case_dir + f"log_{case}.txt")

# =============================================================================
# Saving
# =============================================================================
save = {}
def save_file(save, *args):
    """Create XDMFfiles with the mesh added and variable named  
    args = tuple of strings corresponding to the files you want to save 
    """
    for func_save in args:
        save[func_save] = io.XDMFFile(comm, case_dir + func_save + ".xdmf", "w")
        save[func_save].write_mesh(domain)
    return save 
   
save = save_file(save, "h", "p", "Gamma", "Vs")
hs = u.sub(0)
ps = u.sub(1)
Gammas = u.sub(2)
Vs_s = u.sub(3)
hs.name = "h"
ps.name = "p"
Gammas.name = "Gamma"
Vs_s.name = "Vs"
save["h"].write_function(hs, 0.0)  
save["p"].write_function(ps, 0.0)
save["Gamma"].write_function(Gammas, 0.0)
save["Vs"].write_function(Vs_s, 0.0)

# =============================================================================
# Time stepping
# =============================================================================
t = 0.0  # Initial time 
nsteps = 2000  # N. of timesteps to run
step = 0  # Start time step number
start_time = time.time()
while (step < nsteps):
    t += dt.value
    step += 1
    
    if rank == 0:
        print(f"Timestep {step} / {nsteps}: {t = :.6e}")    

    num_its, converged = solver.solve(u)
    assert(converged)
    u.x.scatter_forward()
    if rank == 0:
        print(f"Number of iterations: {num_its:d}")

    if step % 10 == 0:
        save["h"].write_function(hs, t)
        save["p"].write_function(ps, t)
        save["Gamma"].write_function(Gammas, t)
        save["Vs"].write_function(Vs_s, t)

    # Update solution at previous time step
    u_n.x.array[:] = u.x.array

end_time = time.time()

# Close XDMF files
for key in save:
    save[key].close()

if rank == 0:
    print("End of simulation")
    time_elapsed = end_time - start_time
    average_time_step = time_elapsed / nsteps
    print(f"\nSolving took {time_elapsed:.2f} s") 
    print(f"Average time per time step {average_time_step:.2f} s")
    now = datetime.now()
    print(f"Finished running {__file__} at {now.time().strftime('%H:%M:%S')} on "
          f"{now.date().strftime('%d/%m/%Y')}")
