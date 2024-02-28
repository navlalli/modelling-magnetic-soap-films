""" Solve thickness field over time for a magnetic soap film under the forcing
of an inhomogeneous magnetic field with same size as in the experiments
"""

import numpy as np
import pyvista as pv
import ufl
import time 
import sys

from dolfinx import fem, io, plot, nls, log
from ufl import ds, dx, grad, div, inner, dot, FacetNormal, Identity, exp, ln
from scipy.interpolate import RegularGridInterpolator
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from datetime import datetime

comm = MPI.COMM_WORLD
rank = comm.rank 
if rank == 0:
    print(f"N. of cores = {comm.Get_size()}")

case = "mag_exp"
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
mesh_dir = "./mesh/radial_exp195443/"
mesh_file = "radial_exp.xdmf"
Ldomain = 18.25

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
Gr = fem.Constant(domain, ScalarType(245.0))  # Gravity number
Ma = fem.Constant(domain, ScalarType(40e3))  # Marangoni number
Ca = fem.Constant(domain, ScalarType(1e-7))  # Capillary number 
Bq_d = fem.Constant(domain, ScalarType(2.5e-4))  # Dilational Boussinesq number
Bq_sh = fem.Constant(domain, ScalarType(2.5e-4))  # Shear Boussinesq number
Psi = fem.Constant(domain, ScalarType(50.0))  # Magnetic number
Je = fem.Constant(domain, ScalarType(0.0))  # Evaporation

# Non-dimensional numbers relevant to magnetite NP transport
alpha = fem.Constant(domain, ScalarType(15.9))
Pe_c = fem.Constant(domain, ScalarType(2000.0))  # Not relevant since uniform c
phi_c = fem.Constant(domain, ScalarType(7.9e-3))  # Not relevant since uniform c

# Non-dimensional numbers relevant to surfactant transport
Lambda = fem.Constant(domain, ScalarType(0.15))
Pe_s = fem.Constant(domain, ScalarType(10.0))

# Time discretisation
theta = 1.0
h_mid = (1.0 - theta) * h_n + theta * h
p_mid = (1.0 - theta) * p_n + theta * p
c_mid = (1.0 - theta) * c_n + theta * c
Gamma_mid = (1.0 - theta) * Gamma_n + theta * Gamma
Vs_mid = (1.0 - theta) * Vs_n + theta * Vs

H = fem.Function(V2)  # Magnitude of magnetic field intensity
C = fem.Function(V2)  # Centre surface

# Boundary fluxes
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

def langevin(arg):
    """ Langevin function for M / Ms """
    return 1 / ufl.tanh(arg) - 1 / arg 

c_uniform = fem.Constant(domain, ScalarType(0.5))
def M(c_t):
    """ Magnetisation as a function of only H """
    return c_uniform * langevin(alpha * H) / langevin(alpha)

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
Gamma_initial = 0.02
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

def free_shape(x):
    """ Define the centre surface of the film """
    epsilon_loc = 5e-3
    R = 100e3  # R / H
    xc = Ldomain / 2
    yc = Ldomain / 2
    Ccentre = R - np.sqrt(R**2 - xc**2 / (epsilon_loc**2))  # C at the centre
    zc = Ccentre - R 
    C = zc + np.sqrt(R**2 - 1 / epsilon_loc**2 * ((x[0] - xc)**2 + (x[1] - yc)**2))
    # When plotting, multiply by epsilon so both the domain and what you are plotting
    # are normalised by L
    # C = epsilon_loc * (zc + np.sqrt(R**2 - 1 / epsilon_loc**2 * ((x[0] - xc)**2 + (x[1] - yc)**2)))
    return C

C.interpolate(free_shape)
# if rank == 0:
#     plot_func(C, "C", show_warp=1)

# =============================================================================
# External magnetic field 
# =============================================================================
def import_Hmag():
    """ Import the magnetic field created by magpy """
    Hdir = "Hfield/exp_meniscus_left/"
    lcap = 2  # Capillary length (mm)
    xpos, ypos = np.loadtxt(Hdir + "pos.txt", unpack=True)
    xpos_nond = xpos / lcap  # 0 <= xpos_nond <= 18.25
    ypos_nond = ypos / lcap  # 0 <= ypos_nond <= 18.25
    Hmag = np.loadtxt(Hdir + 'Hmag.txt') * 1000  # Convert from kA/m to A/m
    # Linear interpolation of the external magnetic field - rows of Hmag correspond
    # with y position and cols with x position 
    interp_Hmag = RegularGridInterpolator((ypos_nond, xpos_nond), Hmag,
                                          method='linear', bounds_error=False,
                                          fill_value=None)
    return interp_Hmag

interp_Hmag = import_Hmag()

def expr_Hmag(x):
    """ Interpolate Hmag to a function space defined on the mesh """
    N = x.shape[1]
    Hmag = np.zeros(N)
    for i in range(N):
        # Interpolation function takes (y, x)
        Hmag[i] = float(interp_Hmag((x[1, i], x[0, i])))
    return (Hmag)

H.interpolate(expr_Hmag)
# Non-dimensionalise Hmag
Harr = H.x.array
# Length on each rank gathered to rank 0
Hcounts = np.array(comm.gather(len(Harr), 0))  
if rank == 0:
    Hgather = np.zeros(np.sum(Hcounts), dtype=float)
else:
    Hgather = None
# Gather on rank 0
comm.Gatherv(Harr, (Hgather, Hcounts), 0)
if rank == 0:
    Hc = np.max(Hgather) 
else:
    Hc = None
# Broadcast Hc from rank 0 to all other ranks
Hc = comm.bcast(Hc, root=0)
print(f"{rank = }: {Hc = }")
H.x.array[:] = H.x.array[:] / Hc
# if rank == 0:
#     plot_func(H, "Hmag", show_warp=0)

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
   
save = save_file(save, "h", "p", "c", "Gamma", "Vs")
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

def eval_circ(x):
    """ Circle in the centre of the domain """
    values = np.zeros(x.shape[1], dtype=ScalarType)
    centre = (Ldomain/2, Ldomain/2)
    radius = Ldomain/4
    mask = (x[0]-centre[0])**2 + (x[1]-centre[1])**2 < radius**2
    values[mask] = 1
    return values

circ = fem.Function(DG0)
circ.interpolate(eval_circ)

def eval_left(x):
    values = np.zeros(x.shape[1], dtype=ScalarType)
    values[x[0] < Ldomain/2] = 1
    return values

left = fem.Function(DG0)
left.interpolate(eval_left)

def eval_right(x):
    values = np.zeros(x.shape[1], dtype=ScalarType)
    values[x[0] > Ldomain/2] = 1
    return values

right = fem.Function(DG0)
right.interpolate(eval_right)

# Area
boundary_area_ufl = fem.form(fem.Constant(domain, ScalarType(1.0)) * ds)
boundary_area = assembler(boundary_area_ufl)
domain_area_ufl = fem.form(fem.Constant(domain, ScalarType(1.0)) * dx)
domain_area = assembler(domain_area_ufl)
domain_area_circ_ufl = fem.form(circ * dx)
domain_area_circ = assembler(domain_area_circ_ufl)
domain_area_left_ufl = fem.form(left * dx)
domain_area_left = assembler(domain_area_left_ufl)
domain_area_right_ufl = fem.form(right * dx)
domain_area_right = assembler(domain_area_right_ufl)
if rank == 0:
    print(f"{boundary_area = :.6e}")
    print(f"{domain_area = :.6e}")
    print(f"{domain_area_circ = :.6e}")
    print(f"{domain_area_left = :.6e}")
    print(f"{domain_area_right = :.6e}")

# h domain
h_ufl = fem.form(h * dx)
h_domain = assembler(h_ufl) / domain_area  # Initial h_domain
h_circ_ufl = fem.form(h * circ * dx)
h_circ = assembler(h_circ_ufl) / domain_area_circ  # Initial h_circ
h_left_ufl = fem.form(h * left * dx)
h_left = assembler(h_left_ufl) / domain_area_left  # Initial h_left
h_right_ufl = fem.form(h * right * dx)
h_right = assembler(h_right_ufl) / domain_area_right  # Initial h_right

if rank == 0:
    h_domain_l = [h_domain]
    h_circ_l = [h_circ]
    h_left_l = [h_left]
    h_right_l = [h_right]
    t_save = [0.0]

def post_process(t):
    """ Post-processing of solution """
    h_domain = assembler(h_ufl) / domain_area
    h_circ = assembler(h_circ_ufl) / domain_area_circ
    h_left = assembler(h_left_ufl) / domain_area_left
    h_right = assembler(h_right_ufl) / domain_area_right
    if rank == 0:
        h_domain_l.append(h_domain)
        h_circ_l.append(h_circ)
        h_left_l.append(h_left)
        h_right_l.append(h_right)
        t_save.append(t)

def save_h(t_arr, h_circ, h_left, h_right, h_domain):
    """ Save average h over the domain """
    header = f"Nond time, nond average thickness in centre, left, right, domain "\
             + f"Ca = {Ca.value :.6e}, Gamma_i = {Gamma_initial}, " \
             + f"Psi = {Psi.value :.6e}, c_uniform = {c_uniform.value}"
    save_arr = np.column_stack((t_arr, h_circ, h_left, h_right, h_domain))
    np.savetxt(case_dir + f"h_{ID}.txt", save_arr, header=header)

# =============================================================================
# Time stepping
# =============================================================================
t = 0.0  # Initial time 
nsteps = 500  # N. of timesteps to run
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

# Save time values and average thicknesses
if rank == 0:
    save_h(np.array(t_save), np.array(h_circ_l), np.array(h_left_l),
           np.array(h_right_l), np.array(h_domain_l))

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
