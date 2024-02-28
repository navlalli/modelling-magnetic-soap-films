""" Using derived lubrication model for a vertical soap film to investigate
whether marginal regeneration can be seen at the bottom boundary 
"""

import numpy as np
import pyvista as pv
import ufl
import time 
import matplotlib.pyplot as plt
import sys
import os

from dolfinx import fem, io, mesh, plot, nls, log
# from dolfinx.fem.petsc import NonlinearProblem
# from dolfinx.nls.petsc import NewtonSolver
from ufl import ds, dx, grad, div, inner, dot, sqrt, FacetNormal, Identity, exp, ln, as_vector
# from scipy.interpolate import RegularGridInterpolator
# from scipy.constants import mu_0, pi, k, N_A, e, epsilon_0, gas_constant
from mpi4py import MPI

from petsc4py import PETSc
from petsc4py.PETSc import ScalarType
from datetime import datetime

comm = MPI.COMM_WORLD
rank = comm.rank 
if rank == 0:
    print(f"N. of cores = {comm.Get_size()}")

main_dir = "/home/nav/navScripts/fenicsx/vertical-soap/"
case = "vertical_soap"
sol_dir = main_dir + f"sol_{case}/"
ID = sys.argv[1] if len(sys.argv) > 1 else "TMP"
case_dir = sol_dir + ID + "/"

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
# Uniform meshes on a square domain
# mesh_dir = "/home/nav/navScripts/fenicsx/vertical-soap/mesh/uniform_square10478/"
# mesh_dir = "/home/nav/navScripts/fenicsx/vertical-soap/mesh/uniform_square23252/"
# mesh_dir = "/home/nav/navScripts/fenicsx/vertical-soap/mesh/uniform_square47288/"
# mesh_dir = "/home/nav/navScripts/fenicsx/vertical-soap/mesh/uniform_square103086/"
mesh_dir = "/home/nav/navScripts/fenicsx/vertical-soap/mesh/uniform_square258008/"
# mesh_dir = "/home/nav/navScripts/fenicsx/vertical-soap/mesh/uniform_square512678/"
mesh_file = "uniform_square.xdmf"
Ldomain = 20.0  # Not sure how to automate this as each processor owns parts of the domain

# Load in mesh from XDMF file
with io.XDMFFile(comm, mesh_dir + mesh_file, "r") as xdmf:
    domain = xdmf.read_mesh(name="Grid")   
tdim = domain.topology.dim
fdim = tdim - 1

# Plot mesh    
# if rank == 0:
#     plot_mesh(domain) 
# sys.exit(0)

DG0 = fem.FunctionSpace(domain, ("DG", 0))
V = fem.FunctionSpace(domain, ("CG", 1))
V2 = fem.FunctionSpace(domain, ("CG", 2))
W0 = fem.VectorFunctionSpace(domain, ("DG", 0))
W = fem.VectorFunctionSpace(domain, ("CG", 1))

hFE = ufl.FiniteElement("CG", domain.ufl_cell(), 1)
pFE = ufl.FiniteElement("CG", domain.ufl_cell(), 1)
GammaFE = ufl.FiniteElement("CG", domain.ufl_cell(), 1)
VsFE = ufl.VectorElement("CG", domain.ufl_cell(), 1)

ME = fem.FunctionSpace(domain, ufl.MixedElement([hFE, pFE, GammaFE, VsFE]))

# =============================================================================
# Create tags
# =============================================================================
x = ufl.SpatialCoordinate(domain)

# =============================================================================
# Defining the variational problem
# =============================================================================
du = ufl.TrialFunction(ME)
v0, v1, v2, v3 = ufl.TestFunctions(ME)
u = fem.Function(ME)  # Current solution
u_n = fem.Function(ME)  # Solution from previous time step
h, p, Gamma, Vs = ufl.split(u)
h_n, p_n, Gamma_n, Vs_n = ufl.split(u_n)

epsilon = fem.Constant(domain, ScalarType(5e-3))  # Lubrication parameter
Gr = fem.Constant(domain, ScalarType(1))  # Gravity number
Ma = fem.Constant(domain, ScalarType(500))  # Marangoni number
Ca = fem.Constant(domain, ScalarType(1e-7))  # Capillary number 
Bq_d = fem.Constant(domain, ScalarType(5e-4))  # Dilational Boussinesq number
Bq_sh = fem.Constant(domain, ScalarType(5e-4))  # Shear Boussinesq number
Je = fem.Constant(domain, ScalarType(0.0))  # Je / (rho * U * epsilon)

# Non-dimensional terms relevant to surfactant transport
Lambda = fem.Constant(domain, ScalarType(0.15))
Pe_s = fem.Constant(domain, ScalarType(10))

# Time discretisation
theta = 1.0
h_mid = (1.0 - theta) * h_n + theta * h
p_mid = (1.0 - theta) * p_n + theta * p
Gamma_mid = (1.0 - theta) * Gamma_n + theta * Gamma
Vs_mid = (1.0 - theta) * Vs_n + theta * Vs

# Boundary fluxes
n = FacetNormal(domain)
suction_power = fem.Constant(domain, ScalarType(0.0))
suction_power.value = 5.0
hn = fem.Constant(domain, ScalarType(0.0))
hn.value = np.tan(0.0 * np.pi / 180)  # partialh / partialn

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

dt = fem.Constant(domain, ScalarType(0.001))

# Weak form
L0 = h * v0 * dx - h_n * v0 * dx \
     + dt * div(Vs_mid * h_mid) * v0 * dx \
     + dt * 1/3 * h_mid**3 * dot(grad(p_mid), grad(v0)) * dx \
     + dt * 1/3 * Gr * div(as_vector([h_mid**3, 0])) * v0 * dx \
     + dt * Je * v0 * dx \
     + dt * 1/3 * h_mid**3 * dot(grad(Pi(h_mid)), n) * v0 * ds \
     + dt * suction_power * h_mid**3 * v0 * ds

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

# Non-uniform initial h
# def initial_h(x):
#     """ Define spatially varying initial condition for h """
#     return h_initial * (x[0] + x[1]) / 40 + 0.1
# u.sub(0).interpolate(initial_h)
# u_n.sub(0).interpolate(initial_h)
# Plot initial h
# if rank == 0:
#     V_h, dofs_h = ME.sub(0).collapse()
#     plot_func_mixed(V_h, u.x.array[dofs_h].real, "h") 
# sys.exit(0)

# =============================================================================
# Solver
# =============================================================================
a = ufl.derivative(L, u, du)
bcs = []
# problem = NonlinearProblem(L, u, bcs, a) 
# solver = NewtonSolver(comm, problem)
problem = fem.petsc.NonlinearProblem(L, u, bcs, a) 
solver = nls.petsc.NewtonSolver(comm, problem)
solver.convergence_criterion = "residual"  # "residual" or "incremental"
solver.max_it = 20
solver.rtol = 1e-10
solver.atol = 1e-10
solver.report = True
# ksp = solver.krylov_solver
# opts = PETSc.Options()
# option_prefix = ksp.getOptionsPrefix()
# print(opts[f"{option_prefix}ksp_type"])
# print(opts[f"{option_prefix}pc_type"])
# print(opts[f"{option_prefix}pc_factor_mat_solver_type"])
# sys.exit(0)

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

t_save = [0.0]  # The values of t at which saving is done
def save_t(arr):
    """ Save arr to file """
    np.savetxt(case_dir + "t.txt", arr, header="Time values at which saving was done")

# Area
boundary_area_ufl = fem.form(fem.Constant(domain, ScalarType(1.0)) * ds)
boundary_area = assembler(boundary_area_ufl)
Qn_cap_ufl = fem.form(-1/3 * h_mid**3 * dot(grad(p_mid + Pi(h_mid)), n) * ds)
Qn_bc_ufl = fem.form(suction_power * h_mid**3 * ds)
h_neu_ufl = fem.form(dot(grad(h_mid), n) * ds)
if rank == 0:
    Qn_cap_l = []
    Qn_bc_l = []
    h_neu_l = []

def test_sol():
    """ Analyse solution runtime """
    Qn_cap = assembler(Qn_cap_ufl) / boundary_area
    Qn_bc = assembler(Qn_bc_ufl) / boundary_area
    h_neu = assembler(h_neu_ufl) / boundary_area
    if rank == 0:
        Qn_cap_l.append(Qn_cap)
        Qn_bc_l.append(Qn_bc)
        h_neu_l.append(h_neu)

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

    # Save solution
    if step % 10 == 0:
        save["h"].write_function(hs, t)
        save["p"].write_function(ps, t)
        save["Gamma"].write_function(Gammas, t)
        save["Vs"].write_function(Vs_s, t)
        t_save.append(t)
        test_sol()

    # Update solution at previous time step
    u_n.x.array[:] = u.x.array

end_time = time.time()

# Close XDMF files
for key in save:
    save[key].close()

# Summary stats
if rank == 0:
    save_t(np.array(t_save))
    print("\nSimulation complete")
    now = datetime.now()
    print(f"Finished running {__file__} at {now.time().strftime('%H:%M:%S')} on "
          f"{now.date().strftime('%d/%m/%Y')}")
    time_elapsed = end_time - start_time
    average_time_step = time_elapsed / nsteps
    print(f"\nSolving time: {time_elapsed:.2f} s") 
    print(f"Average time per time step: {average_time_step:.2f} s\n")
    mosaic = "AB"
    width = 10
    fig, axs = plt.subplot_mosaic(mosaic, constrained_layout=True, figsize=(width, 0.5*width))
    axs["A"].plot(t_save[1:], Qn_cap_l, 'b-')
    axs["A"].plot(t_save[1:], Qn_bc_l, 'k--')
    axs["A"].set_xlabel("t")
    axs["A"].set_ylabel("Qn")
    axs["B"].plot(t_save[1:], h_neu_l, 'b--')
    axs["B"].set_xlabel("t")
    axs["B"].set_ylabel("h_neu")
    save_fig = 1
    if save_fig:
        fig_name = f"overview{ID}.png"
        path_fig = sol_dir + "png/" + fig_name 
        if os.path.exists(path_fig):
            print("File already exists - file not saved")
        else:
            fig.savefig(path_fig, dpi=200, bbox_inches='tight')
            print(f"Saved {fig_name}")
    plt.show()
