""" Solving thickness field over time for a horizontal soap film of unit size """

import numpy as np
import pyvista as pv
import ufl
import time 
import sys

from dolfinx import fem, io, mesh, plot, nls, log
from ufl import ds, dx, grad, div, inner, dot, FacetNormal, Identity, exp, ln
from scipy.interpolate import RegularGridInterpolator
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from datetime import datetime

comm = MPI.COMM_WORLD
rank = comm.rank 
if rank == 0:
    print(f"N. of cores = {comm.Get_size()}")

case = "soap_unit"
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
    p.add_mesh(grid, show_edges=True)
    p.add_axes()
    p.view_xy()
    p.show_grid()
    p.show()

def plot_func(fem_func, func_str, show_warp=0):
    """ Plot function defined on mesh with pyvista 
    fem_func = fem.Function() that you want to plot
    func_str = name of function for use in pyvista
    """
    fs = fem_func.function_space
    cells, types, x = plot.create_vtk_mesh(fs)
    grid = pv.UnstructuredGrid(cells, types, x)
    grid.point_data[func_str] = fem_func.x.array.real
    warp = grid.warp_by_scalar()
    p = pv.Plotter()
    if show_warp:
        p.add_mesh(warp, show_edges=False)
    else:
        p.add_mesh(grid, show_edges=False)
    p.add_title(func_str, font_size=16)
    p.view_xy()
    p.add_axes()
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
    p.view_xy()
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
mesh_dir = "./mesh/radial_unit10095/"
mesh_file = "radial_unit.xdmf"
Ldomain = 1.0

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
cFE = ufl.FiniteElement("CG", domain.ufl_cell(), 1)
GammaFE = ufl.FiniteElement("CG", domain.ufl_cell(), 1)
VsFE = ufl.VectorElement("CG", domain.ufl_cell(), 1)

ME = fem.FunctionSpace(domain, ufl.MixedElement([hFE, pFE, cFE, GammaFE, VsFE]))

# =============================================================================
# Defining the variational problem
# =============================================================================
du = ufl.TrialFunction(ME)
v0, v1, v2, v3, v4 = ufl.TestFunctions(ME)
u = fem.Function(ME)  # Current solution
u_n = fem.Function(ME)  # Solution from previous time step
h, p, c, Gamma, Vs = ufl.split(u)
h_n, p_n, c_n, Gamma_n, Vs_n = ufl.split(u_n)

epsilon = fem.Constant(domain, ScalarType(5e-3))  # Thin film parameter
Gr = fem.Constant(domain, ScalarType(0.0))  # Gravity number
Ma = fem.Constant(domain, ScalarType(100))  # Marangoni number
Ca = fem.Constant(domain, ScalarType(1e-7))  # Capillary number 
Bq_d = fem.Constant(domain, ScalarType(1.0e-3))  # Dilational Boussinesq number
Bq_sh = fem.Constant(domain, ScalarType(1.0e-3))  # Shear Boussinesq number
Psi = fem.Constant(domain, ScalarType(0.0))  # Magnetic number
Je = fem.Constant(domain, ScalarType(0.0))  # Evaporation

# Non-dimensional numbers relevant to magnetite NP transport - not relevant here
alpha = fem.Constant(domain, ScalarType(15.9))
Pe_c = fem.Constant(domain, ScalarType(2000.0))
phi_c = fem.Constant(domain, ScalarType(7.9e-3))

# Non-dimensional numbers relevant to surfactant transport
Lambda = fem.Constant(domain, ScalarType(0.15))
Pe_s = fem.Constant(domain, ScalarType(1.0))

# Time discretisation
theta = 1.0
h_mid = (1.0 - theta) * h_n + theta * h
p_mid = (1.0 - theta) * p_n + theta * p
c_mid = (1.0 - theta) * c_n + theta * c
Gamma_mid = (1.0 - theta) * Gamma_n + theta * Gamma
Vs_mid = (1.0 - theta) * Vs_n + theta * Vs

H = fem.Function(V)  # Magnitude of magnetic field intensity
H.x.array[:] = 0.1  # Assign an arbitray non-zero value
C = fem.Function(V)  # Shape of centreline
C.x.array[:] = 0.0

# Boundary fluxes
n = FacetNormal(domain)
P = fem.Constant(domain, ScalarType(5.0))  # Capillary suction strength
hn = fem.Constant(domain, ScalarType(0.0))  # partial h / partial nd
hn.value = np.tan(30.0 * np.pi / 180)

A = 8.3e-6
B = 4.2e6
D = 3.0e3
def Pi(h_t):
    """ Disjoining pressure """
    return - A / h_t**3 + B * exp(- D * h_t)

def langevin(arg):
    """ Langevin function for M / Ms """
    return 1 / ufl.tanh(arg) - 1 / arg 

def M(c_t):
    """ Magnetisation as a function of magnetite NP concentration and H """
    return c_t * langevin(alpha * H) / langevin(alpha)

def gamma(Gamma_t):
    """ Surface tension as a function of interfacial surfactant concentration """
    return 1 + Lambda * ln(1 - Gamma_t)

def pre_curv(Gamma_t):
    """ Strength of the curvature related term in pressure equation """
    return (epsilon**2 * Ma * gamma(Gamma_t) + epsilon**3 / Ca)

def drainage():
    """ Return drainage flux """
    return Vs_mid * h_mid - 1/3 * h_mid**3 * (grad(p_mid) - Psi * M(c_mid) * grad(H))

def surf_rheo(Vs_t):
    """ Tensor associated with surface rheology """
    return Bq_d * div(Vs_t) * Identity(2) + Bq_sh * grad(Vs_t)

dt = fem.Constant(domain, ScalarType(0.001))

# Weak form
L0 = h * v0 * dx - h_n * v0 * dx \
     + dt * div(Vs_mid * h_mid) * v0 * dx \
     + dt * 1/3 * div(h_mid**3 * Psi * M(c_mid) * grad(H)) * v0 * dx \
     + dt * 1/3 * h_mid**3 * dot(grad(p_mid), grad(v0)) * dx \
     + dt * Je * v0 * dx \
     + dt * 1/3 * h_mid**3 * dot(grad(Pi(h_mid)), n) * v0 * ds \
     - dt * 1/3 * h_mid**3 * epsilon * Gr * dot(grad(h_mid + C), n) * v0 * ds \
     + dt * P * h_mid**3 * v0 * ds

L1 = p * v1 * dx - epsilon * Gr * (h + C) * v1 * dx \
     - pre_curv(Gamma) * dot(grad(h), grad(v1)) * dx \
     + Pi(h) * v1 * dx \
     + pre_curv(Gamma) * hn * v1 * ds

L2 = c * v2 * dx - c_n * v2 * dx \
     + dt * dot(drainage() / h_mid, grad(c_mid)) * v2 * dx \
     - dt * alpha / Pe_c * c_mid * (1 - 6.55 * phi_c * c_mid) * langevin(alpha * H) * dot(grad(H), grad(v2)) * dx \
     + dt / Pe_c * (1 + 1.45 * phi_c * c_mid) * dot(grad(c_mid), grad(v2)) * dx

L3 = Gamma * v3 * dx - Gamma_n * v3 * dx \
     - dt * Gamma_mid * dot(Vs_mid, grad(v3)) * dx \
     + dt * 1 / Pe_s / (1 - Gamma_mid) * dot(grad(Gamma_mid), grad(v3)) * dx

L4 = - inner(surf_rheo(Vs), grad(v4)) * dx \
     + Ma * dot(grad(gamma(Gamma)), v4) * dx \
     - h * (dot(grad(p), v4) - Psi * M(c) * dot(grad(H), v4)) * dx

L = L0 + L1 + L2 + L3 + L4

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
# Uniform initial c
c_initial = 0.0
u.sub(2).interpolate(lambda x: np.full(x.shape[1], c_initial))
u_n.sub(2).interpolate(lambda x: np.full(x.shape[1], c_initial))
# Uniform initial Gamma
Gamma_initial = 0.5
u.sub(3).interpolate(lambda x: np.full(x.shape[1], Gamma_initial))
u_n.sub(3).interpolate(lambda x: np.full(x.shape[1], Gamma_initial))
# Uniform initial Vs
Vs_initial = 0.0
u.sub(4).interpolate(lambda x: (np.full(x.shape[1], Vs_initial),
                                np.full(x.shape[1], Vs_initial)))
u_n.sub(4).interpolate(lambda x: (np.full(x.shape[1], Vs_initial),
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
   
save = save_file(save, "h", "p", "c", "Gamma", "Vs", "gradp")
hs = u.sub(0)
ps = u.sub(1)
cs = u.sub(2)
Gammas = u.sub(3)
Vs_s = u.sub(4)
hs.name = "h"
ps.name = "p"
cs.name = "c"
Gammas.name = "Gamma"
Vs_s.name = "Vs"
save["h"].write_function(hs, 0.0)  
save["p"].write_function(ps, 0.0)
save["c"].write_function(cs, 0.0)
save["Gamma"].write_function(Gammas, 0.0)
save["Vs"].write_function(Vs_s, 0.0)

# Area
boundary_area_ufl = fem.form(fem.Constant(domain, ScalarType(1.0)) * ds)
boundary_area = assembler(boundary_area_ufl)
domain_area_ufl = fem.form(fem.Constant(domain, ScalarType(1.0)) * dx)
domain_area = assembler(domain_area_ufl)
if rank == 0:
    print(f"{boundary_area = :.6e}")
    print(f"{domain_area = :.6e}")

# h domain
h_ufl = fem.form(h * dx)
h_domain = assembler(h_ufl) / domain_area  # Initial h_domain

if rank == 0:
    h_domain_l = [h_domain]
    t_save = [0.0]

def post_process(t):
    """ Post-processing of solution """
    h_domain = assembler(h_ufl) / domain_area
    if rank == 0:
        h_domain_l.append(h_domain)
        t_save.append(t)

def save_h(t_arr, h_arr):
    """ Save average h over the domain """
    header = f"Nond time, nond average thickness for Ca = {Ca.value :.6e}, " \
             + f"Ma = {Ma.value :.6e}, Bq_s = {Bq_sh.value :.6e}, Bq_d = {Bq_d.value :.6e}"
    save_arr = np.column_stack((t_arr, h_arr))
    np.savetxt(case_dir + f"h_{ID}.txt", save_arr, header=header)

# =============================================================================
# Time stepping
# =============================================================================
t = 0.0  # Initial time 
nsteps = 1000  # N. of timesteps to run
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

    if step % 1 == 0:
        post_process(t)

    if step % 2 == 0:
        save["h"].write_function(hs, t)
        save["p"].write_function(ps, t)
        save["c"].write_function(cs, t)
        save["Gamma"].write_function(Gammas, t)
        save["Vs"].write_function(Vs_s, t)

    # Update solution at previous time step
    u_n.x.array[:] = u.x.array

end_time = time.time()

# Close XDMF files
for key in save:
    save[key].close()

# Save time values and average thickness
if rank == 0:
    save_h(np.array(t_save), np.array(h_domain_l))

if rank == 0:
    print("End of simulation")
    time_elapsed = end_time-start_time
    average_time_step = time_elapsed / nsteps
    average_step_cpu = time_elapsed / (nsteps * comm.Get_size())
    print(f"\nSolving took {time_elapsed:.2f} s") 
    print(f"Average time per time step {average_time_step:.2f} s")
    print(f"Average time per time step normalised by ncpus {average_step_cpu:.2f} s")
    now = datetime.now()
    print(f"Finished running {__file__} at {now.time().strftime('%H:%M:%S')} on {now.date().strftime('%d/%m/%Y')} \n")
