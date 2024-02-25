""" Solving nond PDE for thickness with magnetic forcing term. Surfactant transport
incorporated with an imposed surface velocity field.

Should I rename p to pm in the variational form since I'm actually solving for the
modified pressure rather than the pressure (pm = p - Pi, where I am calling pm 'p')

"""

import numpy as np
import pyvista as pv
import ufl
import time 
import matplotlib.pyplot as plt
import sys
import os
# import adios4dolfinx

from dolfinx import fem, io, mesh, plot, nls, log
from ufl import ds, dx, grad, div, inner, dot, sqrt, FacetNormal, Identity, exp, ln
from scipy.interpolate import RegularGridInterpolator
from scipy.constants import mu_0, pi, k, N_A, e, epsilon_0, gas_constant
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from datetime import datetime

comm = MPI.COMM_WORLD
rank = comm.rank 
if rank == 0:
    print(f"N. of cores = {comm.Get_size()}")

main_dir = "/home/nav/navScripts/fenicsx/ferro-thickness/"
case = "surf_ferro_thickness"
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
    p.add_mesh(grid, show_edges=True)
    p.add_axes()
    p.view_xy()
    p.show_grid()
    p.show()

def plot_func(fem_func, func_str, show_warp=0, save=0):
    """ Plot function defined on mesh with pyvista 
    fem_func = fem.Function() that you want to plot
    func_str = name of function for use in pyvista
    """
    fs = fem_func.function_space
    cells, types, x = plot.create_vtk_mesh(fs)
    grid = pv.UnstructuredGrid(cells, types, x)
    grid.point_data[func_str] = fem_func.x.array.real
    warp = grid.warp_by_scalar()
    if save == 0:
        p = pv.Plotter()
        if show_warp:
            p.add_mesh(warp, show_edges=False)
        else:
            p.add_mesh(grid, show_edges=False)
        p.add_title(func_str, font_size=16)
        p.view_xy()
        p.add_axes()
        # print(f"{p.camera_position = }")
        p.show()
    elif save == 1:
        p = pv.Plotter(off_screen=True)
        p.add_mesh(warp, show_edges=False, show_scalar_bar=False)
        p.add_title(func_str, font_size=16)
        p.set_background('k')
        p.camera_position = [(0.75805559, -1.45149515702, 1.43771559498),
                             (0.5213000018, 0.43619180695, 0.089387636564),
                             (-0.0592598927835, 0.57527888070, 0.81580786617)]
        # p.show(screenshot="./sol_surf_ferro_thickness/png/curvB.png")
        p.show()
        print("Saved")

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
# All meshes are in non-d coordinates
# mesh_dir = "/home/nav/navScripts/fenicsx/ferro-thickness/mesh/uniform_meniscus15249/"
# mesh_file = "uniform_unit.xdmf"
# mesh_dir = "/home/nav/navScripts/fenicsx/ferro-thickness/mesh/radial_refinement33078/"
# mesh_dir = "/home/nav/navScripts/fenicsx/ferro-thickness/mesh/radial_refinement13179/"
# mesh_file = "radial_refinement.xdmf"
# Ldomain = 18.25  # Not sure how to automate this as each processor owns parts of the domain

# mesh_dir = "/home/nav/navScripts/fenicsx/ferro-thickness/mesh/uniform_98513/"
# mesh_dir = "/home/nav/navScripts/fenicsx/ferro-thickness/mesh/uniform_28561/"
# Ldomain = 1.0  # Not sure how to automate this as each processor owns parts of the domain
# mesh_file = "uniform_unit.xdmf"

# Meshes with radial refinement for Ldomain = 18.25 
# mesh_dir = "/home/nav/navScripts/fenicsx/ferro-thickness/mesh/radial_refinement7760/"
# mesh_dir = "/home/nav/navScripts/fenicsx/ferro-thickness/mesh/radial_refinement13179/"
# mesh_dir = "/home/nav/navScripts/fenicsx/ferro-thickness/mesh/radial_refinement20370/"
# mesh_dir = "/home/nav/navScripts/fenicsx/ferro-thickness/mesh/radial_refinement33078/"
# mesh_dir = "/home/nav/navScripts/fenicsx/ferro-thickness/mesh/radial_refinement39116/"
# mesh_dir = "/home/nav/navScripts/fenicsx/ferro-thickness/mesh/radial_refinement92030/"
# mesh_dir = "/home/nav/navScripts/fenicsx/ferro-thickness/mesh/radial_refinement133976/"
mesh_dir = "/home/nav/navScripts/fenicsx/ferro-thickness/mesh/radial_refinement195443/"
mesh_file = "radial_refinement.xdmf"
Ldomain = 18.25  # Not sure how to automate this as each processor owns parts of the domain

# Unit meshes with radial refinement
# mesh_dir = "/home/nav/navScripts/fenicsx/ferro-thickness/mesh/radial_unit_5030/"
# mesh_dir = "/home/nav/navScripts/fenicsx/ferro-thickness/mesh/radial_unit_7803/"
# mesh_dir = "/home/nav/navScripts/fenicsx/ferro-thickness/mesh/radial_unit_10095/"
# mesh_dir = "/home/nav/navScripts/fenicsx/ferro-thickness/mesh/radial_unit_15046/"
# mesh_dir = "/home/nav/navScripts/fenicsx/ferro-thickness/mesh/radial_unit_29900/"
# mesh_dir = "/home/nav/navScripts/fenicsx/ferro-thickness/mesh/radial_unit_80006/"
# Ldomain = 1.0  # Not sure how to automate this as each processor owns parts of the domain
# mesh_file = "radial_unit.xdmf"

# Load in mesh from XDMF file
with io.XDMFFile(comm, mesh_dir + mesh_file, "r") as xdmf:
    domain = xdmf.read_mesh(name="Grid")   
# ncells = 100
# domain = mesh.create_unit_square(MPI.COMM_WORLD, ncells, ncells)
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
cFE = ufl.FiniteElement("CG", domain.ufl_cell(), 1)
GammaFE = ufl.FiniteElement("CG", domain.ufl_cell(), 1)
VsFE = ufl.VectorElement("CG", domain.ufl_cell(), 1)

ME = fem.FunctionSpace(domain, ufl.MixedElement([hFE, pFE, cFE, GammaFE, VsFE]))

# =============================================================================
# Create tags
# =============================================================================
x = ufl.SpatialCoordinate(domain)
# boundaries = [(1, lambda x: np.logical_and(x[0] <= Ldomain/2, x[1] <= Ldomain/2)), (2, lambda x: np.logical_and(x[0] >= Ldomain/2, x[1] >= Ldomain/2)),
#               (3, lambda x: np.logical_and(x[0] <= Ldomain/2, x[1] >= Ldomain/2)), (4, lambda x: np.logical_and(x[0] >= Ldomain/2, x[1] <= Ldomain/2))]

# For splitting domain into ds(1) on left and ds(2) on right
boundaries = [(1, lambda x: x[0] <= Ldomain/2), (2, lambda x: x[0] >= Ldomain/2)]


facet_indices, facet_markers = [], []
for (marker, locator) in boundaries:
    facets = mesh.locate_entities(domain, fdim, locator)
    facet_indices.append(facets)
    facet_markers.append(np.full_like(facets, marker))
facet_indices = np.hstack(facet_indices).astype(np.int32)
facet_markers = np.hstack(facet_markers).astype(np.int32)
sorted_facets = np.argsort(facet_indices)
facet_tag = mesh.meshtags(domain, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)

# =============================================================================
# Defining the variational problem
# =============================================================================
du = ufl.TrialFunction(ME)
v0, v1, v2, v3, v4 = ufl.TestFunctions(ME)
u = fem.Function(ME)  # Current solution
u_n = fem.Function(ME)  # Solution from previous time step
h, p, c, Gamma, Vs = ufl.split(u)
h_n, p_n, c_n, Gamma_n, Vs_n = ufl.split(u_n)

epsilon = fem.Constant(domain, ScalarType(5e-3))  # Lubrication parameter
Gr = fem.Constant(domain, ScalarType(245.0))  # Gravity number
Ma = fem.Constant(domain, ScalarType(40e3))  # Marangoni number
Ca = fem.Constant(domain, ScalarType(1e-7))  # Capillary number 
Bq_d = fem.Constant(domain, ScalarType(2.5e-4))  # Dilational Boussinesq number
Bq_s = fem.Constant(domain, ScalarType(2.5e-4))  # Shear Boussinesq number
Psi = fem.Constant(domain, ScalarType(50.0))  # H * epsilon * mu_0 * Hc * Mc / (mu * U)
Je = fem.Constant(domain, ScalarType(0.0))  # Je / (rho * U * epsilon)

# Non-dimensional terms relevant to magnetophoresis - not used as uniform
# beta = fem.Constant(domain, ScalarType(3e-1))  # b0 * mu0 * m * Hc / (U * L)
alpha = fem.Constant(domain, ScalarType(15.9))  # mu0 * m * H / (kb * T)
Pe_c = fem.Constant(domain, ScalarType(2000.0))  # U * L / D0
Z = fem.Constant(domain, ScalarType(7.9e-3))
# Pe_c = fem.Constant(domain, ScalarType(0.5))  # b * mu0 * m * Hc / D
# xi = fem.Constant(domain, ScalarType(5.0))  # tc * U / L

# Non-dimensional terms relevant to surfactant transport
Lambda = fem.Constant(domain, ScalarType(0.15))
Pe_s = fem.Constant(domain, ScalarType(10.0))

# Time discretisation
theta = 1.0
h_mid = (1.0 - theta) * h_n + theta * h
p_mid = (1.0 - theta) * p_n + theta * p
c_mid = (1.0 - theta) * c_n + theta * c
Gamma_mid = (1.0 - theta) * Gamma_n + theta * Gamma
Vs_mid = (1.0 - theta) * Vs_n + theta * Vs

H = fem.Function(V2)  # Magnitude of magnetic field intensity vector field - nond
C = fem.Function(V2)  # Shape of centreline

# Boundary fluxes
n = FacetNormal(domain)
Qn_cap_power = fem.Constant(domain, ScalarType(0.0))  # dot(Q, n)
Qn_cap_power.value = 5.0
# Qn_mag_set = fem.Constant(domain , ScalarType(0.0))
hn = fem.Constant(domain, ScalarType(0.0))
hn.value = np.tan(0.0 * np.pi / 180)  # partialh / partialn
# hn_left = fem.Constant(domain, ScalarType(np.tan(0.0 * np.pi / 180)))
# hn_right = fem.Constant(domain, ScalarType(np.tan(0.0 * np.pi / 180)))
# Jn_c = fem.Constant(domain, ScalarType(0.0))  # Jc dot n, flux at boundary for NPs
cn = fem.Constant(domain, ScalarType(0.0))
# Jn_surf = fem.Function(V2)  # Jsurf dot n, flux at boundary for surfactant
Jn_surf = fem.Constant(domain, ScalarType(0.0))  # fem.Function(V2)  # Jsurf dot n, flux at boundary for surfactant
source_surf = fem.Function(V2)  # Surfactant source term

def Jn_surf_neu(x):
    """ Define spatially varying dot(JGamma, n) """
    xcoords = x[0]
    ycoords = x[1]
    alpha = np.arctan2(ycoords - Ldomain/2, xcoords - Ldomain/2)
    omega = 10
    strength = 60
    return strength * np.sin(omega * alpha)

def trigger_field(x):
    """ Define spatially destabilisation field """
    xcoords = x[0]
    ycoords = x[1]
    rad = Ldomain/2
    alpha = np.arctan2(ycoords - rad, xcoords - rad)
    omega = 8.0
    strength = 3000.0
    field = strength * np.sin(omega * alpha) + strength
    field[(xcoords-rad)**2 + (ycoords-rad)**2 < 0.98 * rad**2] = 0.0 
    field[alpha < 0.799] = 0.0
    field[alpha > 0.80] = 0.0
    return field

# Jn_surf.interpolate(Jn_surf_neu)
# plot_func(Jn_surf, "Jn_surf")
# sys.exit(0)
source_surf.interpolate(trigger_field)
# plot_func(source_surf, "source_surf")
# sys.exit(0)

# heq = 0.05  # Dimensionless equilibrium h
# A = 1e-6  # Dimensionless Hamaker constant - sim fails with A = 0 
# dis_n = 9
# dis_m = 3
    # return A * ((heq / h_t)**dis_n - (heq / h_t)**dis_m)
A = 8.3e-6
B = 4.2e6
D = 3.0e3
def Pi(h_t):
    """ Disjoining pressure """
    return - A / h_t**3 + B * exp(- D * h_t)

def langevin(arg):
    """ Langevin function for M / Ms """
    return 1 / ufl.tanh(arg) - 1 / arg 

# def M(c_t):
#     """ Magnetisation as a function of c and H """
#     return c_t * langevin(alpha * H) / langevin(alpha)
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
    return Bq_d * div(Vs_t) * Identity(2) + Bq_s * grad(Vs_t)

dt = fem.Constant(domain, ScalarType(0.001))
sc = fem.Constant(domain, ScalarType(1.0))
trigger = fem.Constant(domain, ScalarType(0.0))

# Weak form
L0 = h * v0 * dx - h_n * v0 * dx \
     + dt * div(Vs_mid * h_mid) * v0 * dx \
     + dt * 1/3 * div(h_mid**3 * Psi * M(c_mid) * grad(H)) * v0 * dx \
     + dt * 1/3 * h_mid**3 * dot(grad(p_mid), grad(v0)) * dx \
     + dt * Je * v0 * dx \
     + dt * 1/3 * h_mid**3 * dot(grad(Pi(h_mid)), n) * v0 * ds \
     - dt * 1/3 * h_mid**3 * epsilon * Gr * dot(grad(h_mid + C), n) * v0 * ds \
     + dt * Qn_cap_power * h_mid**3 * v0 * ds
     # - dt * 1/3 * h_mid**3 * Psi * M(c_mid) * dot(grad(H), grad(v0)) * dx \

     # + dt * Qn_p * v0 * ds
     # - dt * h_mid * dot(Vs_mid, grad(v0)) * dx \
     # - dt * 1/3 * h_mid**3 * epsilon * Gr * dot(grad(h_mid + B), n) * v0 * ds \
     # - dt * 1/3 * h_mid**3 * dot(grad(p_mid), n) * v0 * ds
     # + dt * (Qn_p + 1/3 * h_mid**3 * mag_nond * M(c_mid) * dot(grad(H), n)) * v0 * ds
     # + dt * (-1/3 * h_mid**3 * dot(grad(p_mid), n) + Qn_mag_set) * v0 * ds
     # + sc * dt * div(Vs * h_mid) * v0 * dx \

L1 = p * v1 * dx - epsilon * Gr * (h + C) * v1 * dx \
     - pre_curv(Gamma) * dot(grad(h), grad(v1)) * dx \
     + Pi(h) * v1 * dx \
     + pre_curv(Gamma) * hn * v1 * ds
     # + div(grad(B)) * pre_curv(Gamma) * v1 * dx \
     # + (epsilon**2 * Ma * gamma + epsilon**3 / Ca) * hn_left * v1 * ds(1) \
     # + (epsilon**2 * Ma * gamma + epsilon**3 / Ca) * hn_right * v1 * ds(2)

L2 = c * v2 * dx - c_n * v2 * dx \
     + dt * dot(drainage() / h_mid, grad(c_mid)) * v2 * dx \
     - dt * alpha / Pe_c * c_mid * (1 - 6.55 * Z * c_mid) * langevin(alpha * H) * dot(grad(H), grad(v2)) * dx \
     + dt / Pe_c * (1 + 1.45 * Z * c_mid) * dot(grad(c_mid), grad(v2)) * dx \

     # - dt * c_mid * dot(drainage() / h_mid, n) * v2 * ds
     # - dt * fem.Constant(domain, ScalarType(0.02)) * v2  * ds

     # - dt / Pe_c * (1 + 1.45 * Z * c_mid) * cn * v2 * ds
     # + dt * beta * div(c_mid * (1 - 6.55 * Z * c_mid) * langevin(alpha * H) * grad(H)) * v2 * dx \
     # + dt * dot(Vs_mid, grad(c_mid)) * v2 * dx \
     # - dt * 1/3 * h_mid**2 * dot((grad(p_mid) - Psi * M(c_mid) * grad(H)), grad(c_mid)) * v2 * dx \
     # - dt * c_mid * div(drainage() * v2 / h_mid) * dx \
     # - dt * c_mid / h_mid * dot(drainage(), grad(v2)) * dx \
     # + dt / h_mid * dot(drainage(), grad(c_mid * v2)) * dx \

L3 = Gamma * v3 * dx - Gamma_n * v3 * dx \
     - dt * Gamma_mid * dot(Vs_mid, grad(v3)) * dx \
     + dt * 1 / Pe_s / (1 - Gamma_mid) * dot(grad(Gamma_mid), grad(v3)) * dx \
     - dt * trigger * source_surf * v3 * dx \
     + dt * sc * Jn_surf * v3 * ds

# # Should this be multplied by negative 1
# L4 = - Bq * inner(grad(Vs), grad(v4)) * dx \
#      + Ma * dot(grad(gamma(Gamma)), v4) * dx \
#      - h * (dot(grad(p), v4) - Psi * M(c) * dot(grad(H), v4)) * dx
#      # + Bq * dot(ufl.as_vector([1.0, 1.0]), v4) * ds
     
# With surface dilational effects
L4 = - inner(surf_rheo(Vs), grad(v4)) * dx \
     + Ma * dot(grad(gamma(Gamma)), v4) * dx \
     - h * (dot(grad(p), v4) - Psi * M(c) * dot(grad(H), v4)) * dx

# DANGEROUS to add here because it's easy to forget that you are changing the 
# variational form here
# add_dp = 1  # Just set A to 0 to switch off dp
# if add_dp: 
    # L0 += dt * 1/3 * h_mid**3 * dot(grad(Pi(h_mid)), n) * v0 * ds
    # L1 += Pi(h) * v1 * dx

# SUPG
SUPG = 0
if SUPG:
    vel = Vs_mid - 1/3 * h_mid**2 * grad(p_mid)
    tau = fem.Constant(domain, ScalarType(0.001))  # Perhaps should vary with local velocity
    L0 += tau * dot(vel, grad(v0)) * ((h - h_n) / dt + dot(vel, grad(h_mid))) * dx

L = L0 + L1 + L2 + L3 + L4

# if rank == 0:
#     def print_params():
#         """ Print parameters of interest """
#         print("\nNon-dimensional numbers:")
#         # Thickness equation
#         print("\nThickness")
#         print(f"{epsilon.value = :.4e}")
#         print(f"{Gr.value = :.4e}")
#         print(f"{Ma.value = :.4e}")
#         print(f"{Ca.value = :.4e}")
#         print(f"{Bq_s.value = :.4e}")
#         print(f"{Bq_d.value = :.4e}")
#         print(f"{epsilon.value**2 * Ma.value + epsilon.value**3 / Ca.value = :.4e}")
#         print(f"{Psi.value = :.4e}")
#         # Magnetophoresis
#         print("\nMagnetophoresis")
#         print(f"{alpha.value = :.4e}")
#         print(f"{beta.value = :.4e}")
#         print(f"{Pe_c.value = :.4e}")
#         # Surfactant transport
#         print("\nSurfactant")
#         print(f"{Lambda = :.4e}")
#         print(f"{Pe_s.value = :.4e}\n")
#
#     print_params()

# =============================================================================
# Initial condition
# =============================================================================
u.x.array[:] = 0.0
u_n.x.array[:] = 0.0

# Start sim when 0 and restart sim when 1
restart = 0  
if restart == 0:
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
# elif restart == 1:
    # domain = adios4dolfinx.read_mesh(comm, sol_dir + f"restart/domain.bp", "BP4", mesh.GhostMode.shared_facet)
    # Initial h
    # adios4dolfinx.read_function(u.sub(0), sol_dir + f"restart/h.bp") 
    # adios4dolfinx.read_function(u_n.sub(0), sol_dir + f"restart/h.bp") 
    # # Initial p
    # adios4dolfinx.read_function(u.sub(1), sol_dir + f"restart/p.bp") 
    # adios4dolfinx.read_function(u_n.sub(1), sol_dir + f"restart/p.bp") 
    # # Initial c
    # adios4dolfinx.read_function(u.sub(2), sol_dir + f"restart/c.bp") 
    # adios4dolfinx.read_function(u_n.sub(2), sol_dir + f"restart/c.bp") 
    # # Initial Gamma
    # adios4dolfinx.read_function(u.sub(3), sol_dir + f"restart/Gamma.bp") 
    # adios4dolfinx.read_function(u_n.sub(3), sol_dir + f"restart/Gamma.bp") 
    # # Initial Vs
    # adios4dolfinx.read_function(u.sub(4), sol_dir + f"restart/Vs.bp") 
    # adios4dolfinx.read_function(u_n.sub(4), sol_dir + f"restart/Vs.bp") 

else:
    raise Exception(f"{restart = } is not valid")

# sys.exit(0)

def free_shape(x):
    """ Define the curvature of the initially uniform thickness film """
    epsilon_loc = 5e-3
    R = 100e3  # R / H  # 10e3
    xc = Ldomain / 2
    yc = Ldomain / 2
    Cc = R - np.sqrt(R**2 - xc**2 / (epsilon_loc**2))  # C at the centre
    # print(f"{Cc = }")
    zc = Cc - R 
    C = zc + np.sqrt(R**2 - 1 / epsilon_loc**2 * ((x[0] - xc)**2 + (x[1] - yc)**2))
    # When plotting, multiply by epsilon so both the domain and what you are plotting
    # are normalised by L
    # C = epsilon_loc * (zc + np.sqrt(R**2 - 1 / epsilon_loc**2 * ((x[0] - xc)**2 + (x[1] - yc)**2)))
    return C

C.interpolate(free_shape)
# C.x.array[:] = 0
# plot_func(C, "C", show_warp=1, save=0)
# sys.exit(0)

save_curvC = 0
if save_curvC:
    curvC = fem.Function(DG0)
    curvC.name = "centreline_curvature"
    curvC_expr = fem.Expression(div(grad(C)), DG0.element.interpolation_points())
    curvC.interpolate(curvC_expr)
    save_curvC = io.XDMFFile(comm, sol_dir + "curvB.xdmf", "w")
    save_curvC.write_mesh(domain)
    save_curvC.write_function(curvC)
    save_curvC.close()
    print(f"Saved {case_dir}curvC.xdmf")
    sys.exit(0)

Vsmag = 0.1
def radial_Vs(x):
    """ For creating a velocity field which is locally radial outwards - used
    in updating the Vs bc
    """
    xcoords = x[0]
    ycoords = x[1]
    # Vsmag = 0.05 * (0.5 - np.sqrt((xcoords-0.5)**2 + (ycoords-0.5)**2))
    alpha = np.arctan2(ycoords - Ldomain/2, xcoords - Ldomain/2)
    us = Vsmag * np.cos(alpha)  # x cpt of surface velocity
    vs = Vsmag * np.sin(alpha)  # y cpt of surface velocity
    return (us, vs)

radial_initial_Vs = 0 
if radial_initial_Vs:
    u.sub(4).interpolate(radial_Vs)
    u_n.sub(4).interpolate(radial_Vs)

np.random.seed(8)  # Set seed for random number generation
def rand_initial_h(x):
    """ Add noise to initial thickness field """
    N = np.shape(x)[1]
    noise_amp = 0.01
    h = h_initial + (noise_amp * np.random.rand(N) - noise_amp / 2)
    return h

initial_from_rand = 0
if initial_from_rand:
    u.sub(0).interpolate(rand_initial_h)
    u_n.sub(0).interpolate(rand_initial_h)

def initial_h(x):
    """ Define spatially varying initial condition for h """
    centre = (0.5, 0.5)
    h_centre = 0.8
    # Initial
    # B = (h_initial - h_centre) / 0.5
    # h = h_centre + B * np.sqrt((x[0]-centre[0])**2 + (x[1]-centre[1])**2)
    # Quadratic
    h = h_centre + 4 * (h_initial - h_centre) * ((x[0] - centre[0])**2 + (x[1] - centre[1])**2)
    return h

initial_from_func = 0
if initial_from_func:
    # Initial h set by initial_h()
    u.sub(0).interpolate(initial_h)
    u_n.sub(0).interpolate(initial_h)

    # Pressure found using h created by initial_h
    CG2 = fem.FunctionSpace(domain, ("CG", 2))
    h_init = fem.Function(CG2)
    h_init.interpolate(initial_h)
    V_p, dofs_p = ME.sub(1).collapse()
    expr = fem.Expression(-(epsilon**2 * Ma * gamma(Gamma_initial) + epsilon**3 / Ca) * div(grad(h_init)),
                          V_p.element.interpolation_points())
    u.sub(1).interpolate(expr)
    u_n.sub(1).interpolate(expr)

u.x.scatter_forward()

# For analysing h as a single array on rank 0
V_h, dofs_h = ME.sub(0).collapse()
h_arr = u.x.array[dofs_h].real
# Length of thickness array on each rank gathered to rank 0
sendcounts = np.array(comm.gather(len(h_arr), 0))
if rank == 0:
    h_gather = np.zeros(np.sum(sendcounts), dtype=float)
else:
    h_gather = None

# Vs.interpolate(radial_Vs)

# xdmf = io.XDMFFile(comm, sol_dir + "Vs" + ".xdmf", "w")
# xdmf.write_mesh(domain)
# xdmf.write_function(Vs, 0)

# ncpts = 2  # N. of cpts of the vector
# topology, cell_types, geometry = plot.create_vtk_mesh(VsFS)
# values = np.zeros((geometry.shape[0], 3), dtype=np.float64)
# values[:, :ncpts] = Vs.x.array.real.reshape((geometry.shape[0], ncpts))
#     
# # Create a point cloud of glyphs
# function_grid = pv.UnstructuredGrid(topology, cell_types, geometry)
# function_grid["Vs"] = values
# glyphs = function_grid.glyph(orient="Vs", factor=0.15)
#     
# # Create a pyvista-grid for the mesh
# grid = pv.UnstructuredGrid(*plot.create_vtk_mesh(domain, domain.topology.dim))
#     
# # Create plotter
# plotter = pv.Plotter()
# plotter.add_mesh(grid, style="wireframe", color="k")
# plotter.add_mesh(glyphs)
# plotter.show_bounds()
# plotter.view_xy()
# plotter.show()
# sys.exit(0)

# Plot initial condition for h 
# if rank == 0:
#     V_h, dofs_h = ME.sub(0).collapse()
#     plot_func_mixed(V_h, u.x.array[dofs_h].real, "h") 
# sys.exit(0)
# Plot initial condition for p 
# if rank == 0:
#     plot_func_mixed(V_p, u.x.array[dofs_p].real, "p") 
# sys.exit(0)

# =============================================================================
# Dirichlet boundary conditions 
# =============================================================================
# h_end = ScalarType(h_initial)
# h_end = fem.Constant(domain, ScalarType(h_initial))
h_end = fem.Function(V)  # fem.Constant(domain, ScalarType(h_initial))
h_end.x.array[:] = h_initial

# Determine boundary facets (line segments) 
# domain.topology.create_connectivity(fdim, tdim)
# boundary_facets = np.flatnonzero(mesh.exterior_facet_indices(domain.topology))
# boundary_dofs_h = fem.locate_dofs_topological(ME.sub(0), fdim, boundary_facets)
# bc_h = fem.dirichletbc(h_end, boundary_dofs_h, ME.sub(0))

V_h, _ = ME.sub(0).collapse()  # Sub-space for h and dofs in mixed spaced
boundary_facets = mesh.locate_entities_boundary(domain, fdim, 
                                                lambda x: np.full(x.shape[1], True, dtype=bool))
# boundary_dofs_h[0] gives the dofs in the entire mixed function space
# boundary_dofs_h[1] gives the dofs in the function space just for h
boundary_dofs_h = fem.locate_dofs_topological((ME.sub(0), V_h), fdim, boundary_facets)
bc_h = fem.dirichletbc(h_end, boundary_dofs_h, ME.sub(0))
# np.savetxt(case_dir + "boundary_dofs_old.txt", boundary_dofs_h_old, fmt="%s")
# np.savetxt(case_dir + "boundary_dofs_r1.txt", boundary_dofs_h[0], fmt="%s")
# np.savetxt(case_dir + "boundary_dofs_r2.txt", boundary_dofs_h[1], fmt="%s")
# sys.exit(0)

V_c, _ = ME.sub(2).collapse()  # Sub-space for c and dofs in mixed spaced
boundary_dofs_c = fem.locate_dofs_topological((ME.sub(2), V_c), fdim, boundary_facets)
c_boundary = fem.Function(V)  # fem.Constant(domain, ScalarType(h_initial))
c_boundary.x.array[:] = c_initial
bc_c = fem.dirichletbc(c_boundary, boundary_dofs_c, ME.sub(2))

# For analysing h at the boundary as a single array on rank 0
h_boundary_arr = u.x.array[boundary_dofs_h[0]].real
# Length of thickness array on each rank gathered to rank 0
sendcounts_boundary = np.array(comm.gather(len(h_boundary_arr), 0))
if rank == 0:
    h_boundary_gather = np.zeros(np.sum(sendcounts_boundary), dtype=float)
else:
    h_boundary_gather = None

W_Vs, _ = ME.sub(4).collapse()  # Sub-space for Vs and dofs in mixed spaced
boundary_dofs_Vs = fem.locate_dofs_topological((ME.sub(4), W_Vs), fdim, boundary_facets)
Vs_boundary = fem.Function(W_Vs)
Vs_boundary.interpolate(radial_Vs)
bc_Vs = fem.dirichletbc(Vs_boundary, boundary_dofs_Vs, ME.sub(4))


# p_initial = -0.1
# p_end = fem.Constant(domain, ScalarType(p_initial))
# boundary_dofs_p = fem.locate_dofs_topological(ME.sub(1), fdim, boundary_facets)
# bc_p = fem.dirichletbc(p_end, boundary_dofs_p, ME.sub(1))

# print(f"{boundary_facets = }")
# print(f"{np.shape(boundary_facets) = }")
# print(f"{np.min(boundary_facets) = }")
# print(f"{np.max(boundary_facets) = }")

# Determine the dofs of the boundary facets 
# boundary_dofs_h = fem.locate_dofs_geometrical(ME.sub(0), lambda x: x[0] >= 0.5)

# print(f"{boundary_dofs_h = }")
# print(f"{np.shape(boundary_dofs_h) = }")
# print(f"{np.min(boundary_dofs_h) = }")
# print(f"{np.max(boundary_dofs_h) = }")

# =============================================================================
# External magnetic field 
# =============================================================================
# def import_Hmag():
#     """ Import the magnetic field created by magpy """
#     Hdir = main_dir + "Hfield/"
#     xpos = np.loadtxt(Hdir + 'xpos.txt')
#     xpos_unit = (xpos - xpos[0]) / np.max(xpos - xpos[0])
#     ypos = np.loadtxt(Hdir + 'ypos.txt')
#     ypos_unit = (ypos - ypos[0]) / np.max(ypos - ypos[0])
#     Hmag = np.loadtxt(Hdir + 'Hmag.txt') * 1000  # Convert from kA/m to A/m
#     # Linear interpolation of the external magnetic field - rows of Hmag correspond
#     # with y position and cols with x position 
#     interp_Hmag = RegularGridInterpolator((ypos_unit, xpos_unit), Hmag,
#                                           method='linear', bounds_error=False,
#                                           fill_value=None)
#     return interp_Hmag

def import_Hmag():
    """ Import the magnetic field created by magpy """
    Hdir = main_dir + "Hfield/exp_meniscus_left/"  # Include 2 mm meniscus length
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
# plot_func(H, "Hmag", show_warp=0)
# sys.exit(0)
# boundary_area_ufl = fem.form(fem.Constant(domain, ScalarType(1.0)) * ds)
# boundary_area = assembler(boundary_area_ufl)
# print(f"{assembler(fem.form(dot(grad(H), n) * ds(1))) / (boundary_area / 2) = :.6f}")
# print(f"{assembler(fem.form(dot(grad(H), n) * ds(2))) / (boundary_area / 2) = :.6f}")
# print(f"{assembler(fem.form(dot(grad(H), n) * ds(3))) / (boundary_area / 2) = :.6f}")
# print(f"{assembler(fem.form(dot(grad(H), n) * ds(4))) / (boundary_area / 2) = :.6f}")
# print(f"{assembler(fem.form(dot(grad(H), n) * ds)) / boundary_area = :.6f}")

# Plot magnitude of grad(Hmag)
# CG1 = fem.FunctionSpace(domain, ("CG", 1))
# expr = fem.Expression(ufl.sqrt(ufl.dot(grad(H), grad(H))), CG1.element.interpolation_points())
# g_Hmag = fem.Function(CG1)
# g_Hmag.interpolate(expr)
# plot_func(g_Hmag, "g")

# =============================================================================
# Solver
# =============================================================================
a = ufl.derivative(L, u, du)
# bcs = [bc_h]
# bcs = [bc_Vs]
# bcs = [bc_c]
bcs = []
problem = fem.petsc.NonlinearProblem(L, u, bcs, a) 
solver = nls.petsc.NewtonSolver(comm, problem)
solver.convergence_criterion = "residual"  # "residual" or "incremental"
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

# Save gradPi over the domain
save_gradPi = 0
if save_gradPi:
    gradPi = fem.Function(W0)
    gradPi_expr = fem.Expression(h * grad(Pi(h)), W0.element.interpolation_points())
    gradPi.interpolate(gradPi_expr)
    save["gradPi"].write_function(gradPi, 0.0)

# Save gradp over the domain
save_gradp = 1
if save_gradp:
    gradp = fem.Function(W0)
    gradp.name = "gradp"
    # if add_dp == 1:
        # gradp_expr = fem.Expression(1/3 * h**3 * grad(p + Pi(h)), W0.element.interpolation_points())
    gradp_expr = fem.Expression(grad(p - epsilon * Gr * (h + C) + Pi(h)), W0.element.interpolation_points())
    # else:
    #     # gradp_expr = fem.Expression(1/3 * h**3 * grad(p), W0.element.interpolation_points())
    #     gradp_expr = fem.Expression(grad(p), W0.element.interpolation_points())
    gradp.interpolate(gradp_expr)
    save["gradp"].write_function(gradp, 0.0)

# grad_pc_n = fem.Function(V)
# grad_pc_n_expr = fem.Expression(dot(grad(p - epsilon * Gr * (h + B) + Pi(h)), n), V.element.interpolation_points())
# grad_pc_n_expr = fem.Expression(dot(grad(h)), n), V.element.interpolation_points())
# sys.exit(0)

# W1 = fem.VectorFunctionSpace(domain, ("CG", 1))
# mforce_expr = fem.Expression(-1/3 * h**3 * (grad(p) - mag_nond * M(c_mid) * grad(H)), W1.element.interpolation_points())
# mforce = fem.Function(W1)
# mforce.interpolate(mforce_expr)
# save["mforce"].write_function(mforce, 0.0)
# sys.exit(0)

# mforce_expr = fem.Expression(M(c), W1.element.interpolation_points())
# mforce_expr = fem.Expression(M(c) * grad(H), W1.element.interpolation_points())

# save["mforce"] = io.VTXWriter(comm, sol_dir + "mforce.bp", mforce)
# save["mforce"].write(0.0)

# WDG0 = fem.VectorFunctionSpace(domain, ("DG", 0))
# gradp = fem.Function(WDG0)
# gradp_expr = fem.Expression(grad(p), WDG0.element.interpolation_points())
# gradp.interpolate(gradp_expr)
# save["gradp"].write_function(gradp, 0)

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

h_dir_ufl = fem.form(h * ds)
h_neu_ufl = fem.form(dot(grad(h_mid), n) * ds)

# pmod_dir_ufl = fem.form(p * ds)
pmod_neu_ufl = fem.form(dot(grad(p_mid), n) * ds)
p_neu_ufl = fem.form(dot(grad(p_mid + Pi(h_mid) - epsilon * Gr * (h_mid + C)), n) * ds)

def mag(vec):
    """ Returns ufl for magnitude of a vector """
    return sqrt(dot(vec, vec))

# h flux
# Qn_ufl = fem.form(-1/3 * h_mid**3 * (dot(grad(p_mid), n) 
#                   - mag_nond * M(c_mid) * dot(grad(H), n)) * ds)
Qn_cap_ufl = fem.form(-1/3 * h_mid**3 * dot(grad(p_mid - epsilon * Gr * (h_mid + C) + Pi(h_mid)), n) * ds)
Qn_cap_ufl_left = fem.form(-1/3 * h_mid**3 * dot(grad(p_mid - epsilon * Gr * (h_mid + C) + Pi(h_mid)), n) * ds(1))
Qn_cap_ufl_right = fem.form(-1/3 * h_mid**3 * dot(grad(p_mid - epsilon * Gr * (h_mid + C) + Pi(h_mid)), n) * ds(2))
Qn_mag_ufl = fem.form(1/3 * h_mid**3 * Psi * M(c_mid) * dot(grad(H), n) * ds)
Qn_mag_ufl_left = fem.form(1/3 * h_mid**3 * Psi * M(c_mid) * dot(grad(H), n) * ds(1))
Qn_mag_ufl_right = fem.form(1/3 * h_mid**3 * Psi * M(c_mid) * dot(grad(H), n) * ds(2))
Qn_Pi_ufl = fem.form(1/3 * h_mid**3 * dot(grad(Pi(h_mid)), n) * ds)
Qn_Vs_ufl = fem.form(h_mid * dot(Vs_mid, n) * ds)
Qn_G_ufl = fem.form(-1/3 * h_mid**3 * epsilon * Gr * dot(grad(h_mid + C), n) * ds)
Qn_bc_ufl = fem.form(Qn_cap_power * h_mid**3 * ds)

# Q_ufl = fem.form(-1/3 * h_mid**3 * (mag(grad(p_mid)) - mag_nond * M(c_mid) * mag(grad(H))) * dx)
Q_cap_ufl = fem.form(-1/3 * h_mid**3 * mag(grad(p_mid - epsilon * Gr * (h_mid + C) + Pi(h_mid))) * dx)
Q_mag_ufl = fem.form(1/3 * h_mid**3 * Psi * M(c_mid) * mag(grad(H)) * dx)
Q_Pi_ufl = fem.form(1/3 * h_mid**3 * mag(grad(Pi(h_mid))) * dx)
Q_Vs_ufl = fem.form(h_mid * mag(Vs) * dx)
Q_G_ufl = fem.form(-1/3 * h_mid**3 * epsilon * Gr * mag(grad(h_mid + C)) * dx)
# Je_ufl = fem.form(Je * dx)

# if add_dp == 0:
#     curv_ufl = fem.form(- (p_mid - epsilon * Gr * (h_mid + B)) / pre_curv(Gamma_mid) * ds)
# elif add_dp == 1:
curv_ufl = fem.form(- (p_mid - epsilon * Gr * (h_mid + C) + Pi(h_mid)) / pre_curv(Gamma_mid) * ds)

# c flux
# Jn_c_ufl = fem.form((- 1 / Pe_c * dot(grad(c_mid), n) + c_mid * langevin(alpha * H) * dot(grad(H), n)) * ds)
Jn_c_diff_ufl = fem.form(- 1 / Pe_c * (1 + 1.45 * Z * c_mid) * dot(grad(c_mid), n) * ds)
Jn_c_mag_ufl = fem.form(alpha / Pe_c * c_mid * (1 - 6.55 * Z * c_mid) * langevin(alpha * H) * dot(grad(H), n) * ds)
Jn_c_hydro_ufl = fem.form(c_mid * dot(drainage() / h_mid, n) * ds)
# J_c_ufl = fem.form((c_mid * langevin(alpha * H) * mag(grad(H)) - 1 / Pe_c * mag(grad(c_mid))) * dx)
c_dir_ufl = fem.form(c * ds)
c_neu_ufl = fem.form(dot(grad(c_mid), n) * ds)

# h domain
h_ufl = fem.form(h * dx)
h_domain = assembler(h_ufl) / domain_area  # Initial h_domain
h_circ_ufl = fem.form(h * circ * dx)
h_circ = assembler(h_circ_ufl) / domain_area_circ
h_left_ufl = fem.form(h * left * dx)
h_left = assembler(h_left_ufl) / domain_area_left  # Initial h_left
h_right_ufl = fem.form(h * right * dx)
h_right = assembler(h_right_ufl) / domain_area_right  # Initial h_right
# c domain
c_ufl = fem.form(c * dx)
c_left_ufl = fem.form(c * left * dx)
# M domain
M_ufl = fem.form(M(c) * dx)
M_left_ufl = fem.form(M(c) * left * dx)

# Surfactant 
# Jn_surf_ufl = fem.form((Gamma_mid * dot(Vs, n) - 1 / Pe_Gamma * dot(grad(Gamma_mid), n)) * ds)
Jn_surf_conv_ufl = fem.form(Gamma_mid * dot(Vs, n) * ds)
Jn_surf_diff_ufl = fem.form(- 1 / Pe_s * dot(grad(Gamma_mid), n) * ds)

J_surf_ufl = fem.form((mag(Vs) * Gamma_mid - 1 / Pe_s * mag(grad(Gamma_mid))) * dx)

Gamma_ufl = fem.form(Gamma * dx)

# Vs
us_n_ufl = fem.form(dot(grad(Vs[0]), n) * ds)
vs_n_ufl = fem.form(dot(grad(Vs[1]), n) * ds)

# print(f"{assembler(fem.form(dot(grad(H), n) * ds(1))) / (boundary_area / 2) = }")
# print(f"{assembler(fem.form(dot(grad(H), n) * ds(2))) / (boundary_area / 2) = }")
# sys.exit(0)

if rank == 0:
    h_dir_l = []
    h_neu_l = []
    # pmod_dir_l = []
    pmod_neu_l = []
    p_neu_l = []
    # Qn
    # Qn_l = []
    Qn_cap_l = []
    Qn_cap_l_left = []
    Qn_cap_l_right = []
    Qn_mag_l = []
    Qn_mag_l_left = []
    Qn_mag_l_right = []
    Qn_Pi_l = []
    Qn_bc_l = []
    Qn_Vs_l = []
    Qn_G_l = []
    curv_l = []
    # Q domain
    # Q_l = [] 
    Q_cap_l = []
    Q_mag_l = []
    Q_Pi_l = []
    Q_Vs_l = []
    Q_G_l = []
    # c
    c_dir_l = []
    c_neu_l = []
    # J_c_l = []
    # Jn
    # Jn_c_l = []
    Jn_c_diff_l = []
    Jn_c_mag_l = []
    Jn_c_hydro_l = []
    # Jn_c_conv_l = []
    # h domain
    h_domain_l = []
    h_circ_l = []
    h_left_l = []
    # c domain
    c_domain_l = []
    c_left_l = []
    # M domain
    M_domain_l = []
    M_left_l = []
    # Surfactant
    # Jn_surf_l = []
    Jn_surf_conv_l = []
    Jn_surf_diff_l = []
    J_surf_l = []
    Gamma_domain_l = []
    # Vs
    us_n_l = []
    vs_n_l = []

def test_sol():
    """ Test boundary conditions and other solution parameters """
    # h
    h_dir = assembler(h_dir_ufl) / boundary_area
    h_neu = assembler(h_neu_ufl) / boundary_area
    if rank == 0:
        h_dir_l.append(h_dir)
        h_neu_l.append(h_neu)
    # h domain
    h_domain = assembler(h_ufl) / domain_area
    h_circ = assembler(h_circ_ufl) / domain_area_circ
    h_left = assembler(h_left_ufl) / domain_area_left
    if rank == 0:
        h_domain_l.append(h_domain)
        h_circ_l.append(h_circ)
        h_left_l.append(h_left)
    # c domain 
    c_domain = assembler(c_ufl) / domain_area
    c_left = assembler(c_left_ufl) / domain_area_left
    if rank == 0:
        c_domain_l.append(c_domain) 
        c_left_l.append(c_left)
    # M domain
    M_domain = assembler(M_ufl) / domain_area
    M_left = assembler(M_left_ufl) / domain_area_left
    if rank == 0:
        M_domain_l.append(M_domain)
        M_left_l.append(M_left)
    # Magnetophoresis 
    # Jn_c = assembler(Jn_c_ufl) / boundary_area
    c_dir = assembler(c_dir_ufl) / boundary_area
    c_neu = assembler(c_neu_ufl) / boundary_area
    # Jn_c_conv = assembler(Jn_c_conv_ufl) / boundary_area
    Jn_c_diff = assembler(Jn_c_diff_ufl) / boundary_area
    Jn_c_mag = assembler(Jn_c_mag_ufl) / boundary_area
    Jn_c_hydro = assembler(Jn_c_hydro_ufl) / boundary_area
    # J_c = assembler(J_c_ufl) / domain_area
    if rank == 0:
        c_dir_l.append(c_dir)
        c_neu_l.append(c_neu)
        # Jn_c_l.append(Jn_c)
        # Jn_c_conv_l.append(Jn_c_conv)
        Jn_c_diff_l.append(Jn_c_diff)
        Jn_c_mag_l.append(Jn_c_mag)
        Jn_c_hydro_l.append(Jn_c_hydro)
        # J_c_l.append(J_c)
    # pmod
    # pmod_dir = assembler(pmod_dir_ufl) / boundary_area
    pmod_neu = assembler(pmod_neu_ufl) / boundary_area
    if rank == 0:
        # pmod_dir_l.append(pmod_dir)
        pmod_neu_l.append(pmod_neu)
    # p
    p_neu = assembler(p_neu_ufl) / boundary_area
    if rank == 0:
        p_neu_l.append(p_neu)
    # Qn
    # Qn = assembler(Qn_ufl) / boundary_area
    Qn_cap = assembler(Qn_cap_ufl) / boundary_area
    Qn_cap_left = assembler(Qn_cap_ufl_left) / (boundary_area / 2)
    Qn_cap_right = assembler(Qn_cap_ufl_right) / (boundary_area / 2)
    Qn_mag = assembler(Qn_mag_ufl) / boundary_area
    Qn_mag_left = assembler(Qn_mag_ufl_left) / (boundary_area / 2)
    Qn_mag_right = assembler(Qn_mag_ufl_right) / (boundary_area / 2)
    Qn_Pi = assembler(Qn_Pi_ufl) / boundary_area
    Qn_Vs = assembler(Qn_Vs_ufl) / boundary_area
    Qn_G = assembler(Qn_G_ufl) / boundary_area
    Qn_bc = assembler(Qn_bc_ufl) / boundary_area
    curv = assembler(curv_ufl) / boundary_area
    if rank == 0:
        # Qn_l.append(Qn)
        Qn_cap_l.append(Qn_cap)
        Qn_cap_l_left.append(Qn_cap_left)
        Qn_cap_l_right.append(Qn_cap_right)
        Qn_mag_l.append(Qn_mag)
        Qn_mag_l_left.append(Qn_mag_left)
        Qn_mag_l_right.append(Qn_mag_right)
        Qn_Pi_l.append(Qn_Pi)
        Qn_bc_l.append(Qn_bc)
        Qn_Vs_l.append(Qn_Vs)
        Qn_G_l.append(Qn_G)
        curv_l.append(curv)
    # Q domain
    # Q = assembler(Q_ufl) / domain_area
    Q_cap = assembler(Q_cap_ufl) / domain_area
    Q_mag = assembler(Q_mag_ufl) / domain_area
    Q_Pi = assembler(Q_Pi_ufl) / domain_area
    Q_Vs = assembler(Q_Vs_ufl) / domain_area
    Q_G = assembler(Q_G_ufl) / domain_area
    if rank == 0:
        # Q_l.append(Q)
        Q_cap_l.append(Q_cap)
        Q_mag_l.append(Q_mag)
        Q_Pi_l.append(Q_Pi)
        Q_Vs_l.append(Q_Vs)
        Q_G_l.append(Q_G)
    # Surfactant
    Jn_surf_conv = assembler(Jn_surf_conv_ufl) / boundary_area
    Jn_surf_diff = assembler(Jn_surf_diff_ufl) / boundary_area
    J_surf = assembler(J_surf_ufl) / domain_area
    Gamma_domain = assembler(Gamma_ufl) / domain_area
    if rank == 0:
        Jn_surf_conv_l.append(Jn_surf_conv)
        Jn_surf_diff_l.append(Jn_surf_diff)
        J_surf_l.append(J_surf)
        Gamma_domain_l.append(Gamma_domain)
    # Vs
    us_n = assembler(us_n_ufl) / boundary_area
    vs_n = assembler(vs_n_ufl)/ boundary_area
    if rank == 0:
        us_n_l.append(us_n)
        vs_n_l.append(vs_n)

def update_Q_bc():
    """ Update Q boundary condition """
    h_dir = assembler(h_dir_ufl) / boundary_area
    # h_domain = assembler(h_ufl) / domain_area
    Qn_p_proposed = Qn_cap_val * (h_dir / h_initial)**3
    if Qn_p_proposed < Qn_cap_val:
        Qn_cap_power.value = Qn_p_proposed
    # h_ave_current = assembler(h_neu_ufl) / boundary_area
    # Qn_p_proposed = Qn_p_initial * (h_ave_current / h_initial)**3
    # if Qn_p_proposed < Qn_p.value:
    #     Qn_p.value = Qn_p_proposed
    # else:
    #     Qn_p.value *= 0.99
    if rank == 0:
        print(f"{Qn_cap_power.value = }")

def update_Je():
    """ Update evaporation flux """
    h_ave_centre = assembler(h_circ_ufl) / domain_area_circ
    if h_ave_centre < heq + 0.03:
        Je.value = 0.0
    if rank == 0:
        print(f"{Je.value = }")

def update_Bq():
    """ Update Boussinesq numbers """
    h_ave_centre = assembler(h_circ_ufl) / domain_area_circ
    if h_ave_centre < heq + 0.03:
        Bq_s.value = 1e6
        Bq_d.value = 1e6
    if rank == 0:
        print(f"{Bq_s.value = }")
        print(f"{Bq_d.value = }")

def switch_on_magnetic():
    """ Switch on magnetic forcing - gradually for stability """
    Psi.value += 0.005
    hn.value *= 0.95
    # sc.value *= 0.99
    # hn_left.value *= 0.9  # np.tan(0.0 * np.pi / 180)
    # hn_right.value *= 0.9  # np.tan(0.0 * np.pi / 180)
    # Jn.value *= 0.99

def ramp_hn_down():
    """ Ramp hn down to approx 0, necessary for a stable black film? """
    hn.value *= 0.95

t_save = [0.0]  # The values of t at which saving is done
def save_t(arr):
    """ Save arr to file """
    np.savetxt(case_dir + "t.txt", arr, header="Time values at which saving was done")

# def save_params():
#     """ Save relevant parameters value """
#     param_path_file = sol_dir + "params/params" + ID + ".md"
#     if os.path.exists(param_path_file):
#         print("Param file already exists - file not saved")
#     else:
#         param_file = open(param_path_file, "w")  # Overwrites if already present 
#         param_file.write(f"# Params for simulation {ID}")
#         param_file.write("\n## Nond numbers for thickness")
#         param_file.write(f"\nepsilon = {epsilon.value  :.6e}")
#         param_file.write(f"\nGr = {Gr.value :.6e}")
#         param_file.write(f"\nMa = {Ma.value :.6e}")
#         param_file.write(f"\nCa = {Ca.value :.6e}")
#         param_file.write(f"\n{epsilon.value**2 * Ma.value + epsilon.value**3 / Ca.value = :.6e}")
#         param_file.write(f"\nBq_s = {Bq_s.value :.6e}")
#         param_file.write(f"\nBq_d = {Bq_d.value :.6e}")
#         param_file.write(f"\nPsi = {Psi.value :.6e}")
#
#         param_file.write("\n## Nond numbers for magnetophoresis")
#         param_file.write(f"\nalpha = {alpha.value :.6e}")
#         param_file.write(f"\nbeta = {beta.value :.6e}")
#         param_file.write(f"\nPe_c = {Pe_c.value :.6e}")
#
#         param_file.write("\n## Nond numbers for surfactant transport")
#         param_file.write(f"\nLambda = {Lambda :.6e}")
#         param_file.write(f"\nPe_Gamma = {Pe_s.value :.6e}")
#
#         param_file.write("\n## Disjoining pressure")
#         param_file.write(f"\nA = {A :.6e}")
#         param_file.write(f"\nB = {B :.6e}")
#         param_file.write(f"\nD = {D :.6e}")
#         # param_file.write(f"\nA = {A :.6e}")
#         # param_file.write(f"\nn = {dis_n :.6e}")
#         # param_file.write(f"\nm = {dis_m :.6e}")
#
#         param_file.write("\n## Boundary conditions")
#         param_file.write(f"\nQn_cap_val = {Qn_cap_power.value :.6e}")
#         param_file.write(f"\nhn_val = {hn.value :.6e}")
#
#         param_file.write("\n## Simulation")
#         param_file.write(f"\ndt = {dt.value :.6e}")
#         param_file.write(f"\nnsteps = {nsteps :.6e}")
#
#         param_file.close()
#         print(f"Saved {param_path_file}")
        
def save_h(t_arr, h_circ, h_left, h_right, h_domain):
    """ Save average h over the domain """
    h_time_dir = "/home/nav/navScripts/fenicsx/ferro-thickness/paper/EXPMAG/"
    header = f"Nond time, nond average thickness in centre, left, right, domain "\
             + f"Ca = {Ca.value :.6e}, Gamma_i = {Gamma_initial}, " \
             + f"Psi = {Psi.value :.6e}, c_uniform = {c_uniform.value}"
    save_arr = np.column_stack((t_arr, h_circ, h_left, h_right, h_domain))
    np.savetxt(h_time_dir + f"h_{ID}.txt", save_arr, header=header)

if rank == 0:
    h_boundary_mean = []
    h_boundary_std = []
    h_domain_t = []
    h_circ_t = []
    h_left_t = []
    h_right_t = []
    t_t = []
    # Initial
    h_domain_t.append(h_domain)
    h_circ_t.append(h_circ)
    h_left_t.append(h_left)
    h_right_t.append(h_right)
    t_t.append(0.0)

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

    # if step == 10:
        # sc.value = 1.0
        # Jn_surf.value = 1.5
        # Pe_Gamma.value = 1.0

    # if step == 4:
    #     dt.value
    #     trigger.value = 1.0
    #     if rank == 0:
    #         print(f"{trigger.value = }")
    # if step == 5:
    #     trigger.value = 0.0
    #     if rank == 0:
    #         print(f"{trigger.value = }")

    # if step == 1570:
    #     dt.value *= 0.1
    # if step > 2000:
    #     hn.value *= 0.95

    # if step == 5:
    #     V_c, dofs_c = ME.sub(2).collapse()
    #     u.x.array[dofs_c] = c_initial

    num_its, converged = solver.solve(u)
    assert(converged)
    u.x.scatter_forward()
    if rank == 0:
        print(f"Number of iterations: {num_its:d}")

        # if num_its > 2:
        #     dt.value *= 0.85
        # elif num_its == 1:
        #     dt.value *= 1/0.995
        # print(f"{dt.value = }")
    
    # Update
    # if step == 25:
    #     Jn_surf.value = 0.6
    # if step == 210:
    #     dt.value = 0.1 * dt.value
    # if step % 2 == 0:
        # update_Je()
        # update_Bq()
        # update_Q_bc()

    if step % 1 == 0:
        # h domain
        h_domain = assembler(h_ufl) / domain_area
        h_circ = assembler(h_circ_ufl) / domain_area_circ
        h_left = assembler(h_left_ufl) / domain_area_left
        h_right = assembler(h_right_ufl) / domain_area_right
        if rank == 0:
            h_domain_t.append(h_domain)
            h_circ_t.append(h_circ)
            h_left_t.append(h_left)
            h_right_t.append(h_right)
            t_t.append(t)

    if step % 2 == 0:
        save["h"].write_function(hs, t)
        save["p"].write_function(ps, t)
        save["c"].write_function(cs, t)
        save["Gamma"].write_function(Gammas, t)
        save["Vs"].write_function(Vs_s, t)
        # gradPi.interpolate(gradPi_expr)
        # save["gradPi"].write_function(gradPi, t)
        gradp.interpolate(gradp_expr)
        save["gradp"].write_function(gradp, t)
        # mforce.interpolate(mforce_expr)
        # save["mforce"].write_function(mforce, t)
        test_sol()
        t_save.append(t)
        # Gather on rank 0
        h_boundary_arr = u.x.array[boundary_dofs_h[0]].real
        comm.Gatherv(h_boundary_arr, (h_boundary_gather, sendcounts_boundary), 0)
        if rank == 0:
            h_boundary_mean.append(np.mean(h_boundary_gather))
            h_boundary_std.append(np.std(h_boundary_gather))

    # if 100 < step < 251:
    #     switch_on_magnetic()

    # if 350 < step < 501:
    #     ramp_hn_down()

    # Update solution at previous time step
    u_n.x.array[:] = u.x.array

end_time = time.time()
# Save time values and nond params
if rank == 0:
    save_t(np.array(t_save))
    # save_params()
    save_h(np.array(t_t), np.array(h_circ_t), np.array(h_left_t), np.array(h_right_t), np.array(h_domain_t))
    # print(f"Change in thickness due to evaporation {Je.value * nsteps * dt.value :.2f}")

# Close XDMF files
for key in save:
    save[key].close()

# def save_restart(domain, func_dict):
#     """ Save files for restarting """
#     adios4dolfinx.write_mesh(domain, sol_dir + "restart/domain.bp") 
#
#     for func_name, func in func_dict.items():
#         adios4dolfinx.write_function(func, sol_dir + f"restart/{func_name}.bp") 
#
# # Save files for restarting 
# save_restart(domain, {"h": hs, "p": ps, "c": cs, "Gamma": Gammas, "Vs": Vs_s})

# Gather on rank 0
h_arr = u.x.array[dofs_h].real
comm.Gatherv(h_arr, (h_gather, sendcounts), 0)

plot = 1
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
    print(f"{np.min(h_gather) = :.6e}")
    print(f"{np.max(h_gather) = :.6e}")

    ms = 0.5
    def plot_list(ax, lst, ylabel=""):
        """ Plot the provided list """
        ax.plot(t_save[1:], lst, 'k*-', ms=ms, label=ylabel)
        ax.set_ylabel(ylabel)

    if plot:
        mosaic = "ABC;DEF;GHI;JKL"
        height = 12
        fig, axs = plt.subplot_mosaic(mosaic, sharex=True, constrained_layout=True,
                                      figsize=(1.7*height, height))
        plot_list(axs["A"], h_dir_l, "h_dir")
        h_boundary_mean = np.array(h_boundary_mean)
        h_boundary_std = np.array(h_boundary_std)
        axs["A"].plot(t_save[1:], h_boundary_mean, 'b--')
        axs["A"].fill_between(t_save[1:], h_boundary_mean-h_boundary_std, 
                              h_boundary_mean+h_boundary_std, color='red', alpha=0.3)
        # ratio = np.array(Qn_cap_l) / np.array(h_dir_l)**3
        # axs["A"].plot(np.linspace(0, 1, len(h_boundary_mean)), ratio, 'g-')

        plot_list(axs["B"], h_neu_l, "h_neu")
        axtwin = axs["B"].twinx()
        axtwin.plot(t_save[1:], us_n_l, '*-', ms=ms, label="us_n", color="orange")
        axtwin.plot(t_save[1:], vs_n_l, '*-', ms=ms, label="vs_n", color="red")
        axtwin.set_ylabel(r"$\nabla Vs \cdot n$")
        axtwin.legend(loc="best")

        plot_list(axs["C"], p_neu_l, "p_neu")
        axs["C"].plot(t_save[1:], pmod_neu_l, 'b*-', ms=ms, label="pmod_neu")
        axs["C"].legend(loc="best")
        axtwin = axs["C"].twinx()
        axtwin.plot(t_save[1:], curv_l, '*-', ms=ms, color="red")
        axtwin.set_ylabel(r"$\nabla^{2} h$")

        # plot_list(axs["D"], Qn_l, "Qn")
        axs["D"].plot(t_save[1:], Qn_cap_l, 'b*-.', ms=ms, label="Qn_cap")
        axs["D"].plot(t_save[1:], Qn_mag_l, 'm*-', ms=ms, label="Qn_mag")
        axs["D"].plot(t_save[1:], Qn_Pi_l, 'r*-', ms=ms, label="Qn_Pi")
        axs["D"].plot(t_save[1:], Qn_Vs_l, '*-', ms=ms, color='orange', label="Qn_Vs")
        axs["D"].plot(t_save[1:], Qn_G_l, '*-', ms=ms, color='saddlebrown', label="Qn_G")
        Qn_sum = np.array(Qn_cap_l) + np.array(Qn_mag_l) + np.array(Qn_Pi_l) \
                 + np.array(Qn_Vs_l) + np.array(Qn_G_l)
        Qn_cap_mag_grav = np.array(Qn_cap_l) + np.array(Qn_mag_l) + np.array(Qn_G_l)
        axs["D"].plot(t_save[1:], Qn_sum, 'k*--', ms=ms, label="Qn_sum")
        axs["D"].plot(t_save[1:], Qn_cap_mag_grav, '--', ms=ms, lw=0.75, color="forestgreen", label="Qn_cap+mag+grav")
        axs["D"].plot(t_save[1:], Qn_bc_l, 'c:', ms=ms, label="Qn_bc")
        axs["D"].legend(loc="best")

        # plot_list(axs["E"], Q_l, "Qtotal")
        axs["E"].plot(t_save[1:], Q_cap_l, 'b*-', ms=ms, label="Q_cap")
        axs["E"].plot(t_save[1:], Q_mag_l, 'm*-', ms=ms, label="Q_mag")
        axs["E"].plot(t_save[1:], Q_Pi_l, 'r*-', ms=ms, label="Q_Pi")
        axs["E"].plot(t_save[1:], Q_Vs_l, '*-', ms=ms, color='orange', label="Q_Vs")
        axs["E"].plot(t_save[1:], Q_G_l, '*-', ms=ms, color='saddlebrown', label="Q_G")
        axs["E"].set_ylabel("|Q|")
        axs["E"].legend(loc="best")

        # plot_list(axs["F"], h_domain_l, "h_domain")
        axs["F"].plot(t_save[1:], h_domain_l, 'k*-', ms=ms, label="h_domain")
        axs["F"].plot(t_save[1:], h_circ_l, 'g*-', ms=ms, label="h_centre")
        axs["F"].plot(t_save[1:], h_left_l, 'b*--', ms=ms, label="h_left")
        axs["F"].set_ylabel("h_ave")
        axs["F"].legend(loc="best")

        # plot_list(axs["G"], Jn_c_l, "Jn_sum")
        # Jn_c_sum = np.array(Jn_c_conv_l) + np.array(Jn_c_diff_l)
        axs["G"].plot(t_save[1:], Jn_c_diff_l, 'b*-', ms=ms, label="Jn_c_diff")
        axs["G"].plot(t_save[1:], Jn_c_mag_l, 'm*-', ms=ms, label="Jn_c_mag")
        axs["G"].plot(t_save[1:], Jn_c_hydro_l, 'r*-', ms=ms, label="Jn_c_hydro")
        Jn_c_total = np.array(Jn_c_diff_l) + np.array(Jn_c_mag_l) + np.array(Jn_c_hydro_l)
        Jn_c_diff_mag = np.array(Jn_c_diff_l) + np.array(Jn_c_mag_l)
        axs["G"].plot(t_save[1:], Jn_c_total, 'k*-', ms=ms, label="Jn_c_diff + mag + hydro")
        axs["G"].plot(t_save[1:], Jn_c_diff_mag, 'c*-', ms=ms, label="Jn_c_diff + mag")
        # axs["G"].plot(np.linspace(0, 1, len(Jn_c_sum)), Jn_c_sum, 'k*-', ms=ms, label="Jn_c_sum")
        # axs["G"].plot(t_save[1:], Jn_c_conv_l, 'r*-', ms=ms, label="Jn_c_conv")
        # axs["G"].plot(t_save[1:], Jn_c_diff_l, 'b*-', ms=ms, label="Jn_c_diff")
        # axs["G"].plot(t_save[1:], c_dir_l, 'k*-')
        axs["G"].set_ylabel("Jn_c")
        # axs["G"].set_title("Update with drag")
        axs["G"].legend(loc="best")

        # plot_list(axs["H"], J_c_l, "|J|")
        # axs["H"].set_title("Update with drag")
        axs["H"].plot(t_save[1:], c_dir_l, 'k*-', ms=ms)
        axs["H"].set_ylabel("c_dir")
        axtwin = axs["H"].twinx()
        axtwin.plot(t_save[1:], c_neu_l, 'r*-', ms=ms)
        axtwin.set_ylabel("c_neu", color='r')

        plot_list(axs["I"], c_domain_l, "c_domain")
        axs["I"].plot(t_save[1:], c_left_l, 'b*--', ms=ms, label="c_left")
        axs["I"].plot(t_save[1:], M_domain_l, 'g*-', ms=ms, label="M_domain")
        axs["I"].plot(t_save[1:], M_left_l, 'r*--', ms=ms, label="M_left")
        axs["I"].legend(loc="best")

        Jn_surf_sum = np.array(Jn_surf_conv_l) + np.array(Jn_surf_diff_l)
        axs["J"].plot(t_save[1:], Jn_surf_sum, 'k*-', ms=ms, label="Jn_sum")
        axs["J"].plot(t_save[1:], Jn_surf_conv_l, 'r*-', ms=ms, label="Jn_conv")
        axs["J"].plot(t_save[1:], Jn_surf_diff_l, 'b*-', ms=ms, label="Jn_diff")
        axs["J"].set_ylabel("Jn_surf")
        axs["J"].legend(loc="best")

        # axs["K"].plot(t_save[1:], J_surf_l, 'k*-', ms=ms, label="|Jsurf|")
        # Qn_left_sum = np.array(Qn_cap_l_left) + np.array(Qn_mag_l_left)
        # Qn_right_sum = np.array(Qn_mag_l_left) + np.array(Qn_mag_l_right)
        axs["K"].plot(t_save[1:], Qn_cap_l_left, 'b*-', ms=ms, label="Qn_cap_left")
        axs["K"].plot(t_save[1:], Qn_cap_l_right, 'b*--', ms=ms, label="Qn_cap_right")
        axs["K"].plot(t_save[1:], Qn_mag_l_left, 'm*-', ms=ms, label="Qn_mag_left")
        axs["K"].plot(t_save[1:], Qn_mag_l_right, 'm*--', ms=ms, label="Qn_mag_right")
        print(f"{np.min(np.array(Qn_cap_l_left)) = }")
        # axs["K"].plot(t_save[1:], Qn_left_sum, 'k*-', ms=ms, lw=2, label="Qn_left_sum")
        # axs["K"].plot(t_save[1:], Qn_right_sum, 'k*--', ms=ms, lw=2, label="Qn_right_sum")
        axs["K"].set_ylabel("Qn_")
        axs["K"].legend(loc="best")

        axs["L"].plot(t_save[1:], Gamma_domain_l, 'k*-', ms=ms, label="Gamma_domain")
        axs["L"].set_ylabel("Gamma_domain")

        fig.suptitle(ID)
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
