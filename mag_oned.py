""" Solve thickness over time in one dimension for a vertical magnetic soap 
film with same assumptions, simulation parameters, and magnetic field as was 
used in Moulton, D. E., & Pelesko, J. A. (2010). Reverse draining of a magnetic
soap film. Physical Review E, 81(4), 046320.

Solving was fastest with just one core in operation
"""

import numpy as np
import pyvista as pv
import ufl
import time 
import sys

from dolfinx import fem, io, mesh, plot, nls, log
from ufl import ds, dx, grad, dot, FacetNormal
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from datetime import datetime

comm = MPI.COMM_WORLD
rank = comm.rank 
if rank == 0:
    print(f"N. of cores = {comm.Get_size()}")

case = "mag_oned"
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
    grid['coords'] = grid.points[:, 0]
    print(f"{grid.n_cells = }")
    print(f"{grid.n_points = }")
    p = pv.Plotter()
    p.add_mesh(grid, show_edges=True, scalar_bar_args={"title": "x-position"})
    p.view_zx()
    p.add_axes()
    p.show()
    print(f"Min x-coord = {np.min(grid.points[:, 0])}")
    print(f"Max x-coord = {np.max(grid.points[:, 0])}")

def plot_func(fem_func, func_str):
    """ Plot function defined on mesh with pyvista 
    fem_func = fem.Function() that you want to plot
    func_str = name of function for use in pyvista
    """
    fs = fem_func.function_space
    cells, types, x = plot.create_vtk_mesh(fs)
    grid = pv.UnstructuredGrid(cells, types, x)
    grid.point_data[func_str] = fem_func.x.array.real
    p = pv.Plotter()
    p.add_mesh(grid, show_edges=False)
    p.add_title(func_str, font_size=16)
    p.view_xy()
    p.add_axes()
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
ncells = 400
xstart = 0.0  # x-coord of start of mesh
xend = 22.5  # x-coord of end of mesh
domain = mesh.create_interval(comm, ncells, [xstart, xend])
tdim = domain.topology.dim
fdim = tdim - 1

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

ME = fem.FunctionSpace(domain, ufl.MixedElement([hFE, pFE]))

# =============================================================================
# Define the variational problem
# =============================================================================
du = ufl.TrialFunction(ME)
v1, v0 = ufl.TestFunctions(ME)
u = fem.Function(ME)  # Current solution
u_n = fem.Function(ME)  # Solution from previous time step
h, p = ufl.split(u)
h_n, p_n = ufl.split(u_n)

epsilon = fem.Constant(domain, ScalarType(5e-3))  # Thin film parameter
Gr = fem.Constant(domain, ScalarType(1.0))  # Gravity number
Ca = fem.Constant(domain, ScalarType(2.5667e-5))  # Capillary number 
Psi = fem.Constant(domain, ScalarType(47.470))  # Magnetic number

# Time discretisation
dt = fem.Constant(domain, ScalarType(0.0002))
theta = 1.0
h_mid = (1.0 - theta) * h_n + theta * h
p_mid = (1.0 - theta) * p_n + theta * p

# Applied magnetic field
H = fem.Function(V2)  # Magnitude of magnetic field intensity
d = 4.25  # Gap between film and current loop
def expr_H(x):
    """ Variation of H with position x """
    eta = 4.5455e-2
    return (1 + eta**2 * (x[0] + d)**2)**(-3/2)
H.interpolate(expr_H)
# plot_func(H, "H")

# Non-dimensional terms relevant to boundary conditions
n = FacetNormal(domain)
Qn = fem.Constant(domain, ScalarType(0.0))  # No-flux boundary condition

def pre_curv():
    """ Strength of the curvature related term in pressure equation """
    return epsilon**3 / Ca

def M():
    """ Magnetisation """
    return H

# Weak form
L0 = h * v0 * dx - h_n * v0 * dx \
     + dt * 1/3 * h_mid**3 * dot(grad(p_mid), grad(v0)) * dx \
     - dt * 1/3 * h_mid**3 * Psi * M() * dot(grad(H), grad(v0)) * dx \
     - dt * 1/3 * h_mid**3 * Gr * v0.dx(0) * dx \
     + dt * Qn * v0 * ds

L1 = p * v1 * dx - pre_curv() * dot(grad(h), grad(v1)) * dx

L = L0 + L1


# =============================================================================
# Initial condition
# =============================================================================
u.x.array[:] = 0.0
u_n.x.array[:] = 0.0

# Initial h
h_initial = 50.0
# Uniform initial h 
u.sub(0).interpolate(lambda x: np.full(x.shape[1], h_initial))
u_n.sub(0).interpolate(lambda x: np.full(x.shape[1], h_initial))
# Uniform initial p
p_initial = 0.0
u.sub(1).interpolate(lambda x: np.full(x.shape[1], p_initial))
u_n.sub(1).interpolate(lambda x: np.full(x.shape[1], p_initial))

# =============================================================================
# Dirichlet boundary conditions 
# =============================================================================
h_end = fem.Function(V)
h_end.x.array[:] = h_initial

def on_boundary(x):
    return np.logical_or(np.isclose(x[0], xstart), np.isclose(x[0], xend))

V_h, _ = ME.sub(0).collapse()  # Sub-space for h and dofs in mixed spaced
boundary_dofs_h = fem.locate_dofs_geometrical((ME.sub(0), V_h), on_boundary)
bc_h = fem.dirichletbc(h_end, boundary_dofs_h, ME.sub(0))
print(f"{rank = }: {boundary_dofs_h = }")

# =============================================================================
# Solver
# =============================================================================
a = ufl.derivative(L, u, du)
bcs = [bc_h]
problem = fem.petsc.NonlinearProblem(L, u, bcs, a) 
solver = nls.petsc.NewtonSolver(comm, problem)
solver.convergence_criterion = "residual"
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
   
save = save_file(save, "h", "p")
hs = u.sub(0)
ps = u.sub(1)
hs.name = "h"
ps.name = "p"
save["h"].write_function(hs, 0.0)  
save["p"].write_function(ps, 0.0)

# Area
domain_area_ufl = fem.form(fem.Constant(domain, ScalarType(1.0)) * dx)
domain_area = assembler(domain_area_ufl)
if rank == 0:
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
    header = "Nond time, nond average thickness"
    save_arr = np.column_stack((t_arr, h_arr))
    np.savetxt(case_dir + f"h_{ID}.txt", save_arr, header=header)

# =============================================================================
# Time stepping
# =============================================================================
t = 0.0  # Initial time 
nsteps = 81000  # N. of timesteps to run
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

    if step % 250 == 0:
        save["h"].write_function(hs, t)
        save["p"].write_function(ps, t)
        post_process(t)

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
