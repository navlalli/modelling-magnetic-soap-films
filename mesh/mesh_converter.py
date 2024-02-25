""" Convert from .msh to .xdmf for use in fenicsx 
Must add physical markers for subdomains and external boundaries in gmsh.
"""

import meshio
import sys

def mesh_conversion(mesh_dir, mesh_file):
    """ Convert mesh from .msh (created with gmsh) to .xdmf
    mesh_file = name of .msh file (excluding .msh)
    """
    mesh_from_file = meshio.read(mesh_dir + mesh_file + ".msh")
    
    def create_mesh(mesh, cell_type, prune_z=False):
        cells = mesh.get_cells_type(cell_type)
        cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
        points = mesh.points[:,:2] if prune_z else mesh.points
        out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read":[cell_data]})
        return out_mesh

    # Mesh and cell markers - prune_z = True since 2d meshes
    triangle_mesh = create_mesh(mesh_from_file, "triangle", prune_z=True)
    xdmf_file = mesh_file 
    meshio.write(mesh_dir + xdmf_file + ".xdmf", triangle_mesh)
    # Facet markers 
    line_mesh = create_mesh(mesh_from_file, "line", prune_z=True)
    meshio.write(mesh_dir + xdmf_file + "Facet.xdmf", line_mesh)


if __name__ == "__main__":
    mesh_name = sys.argv[1]  # e.g. radial_unit
    ID = sys.argv[2]  # N. of cells used
    mesh_conversion(f"./{mesh_name}{ID}/", mesh_name)
    
    print("Mesh conversion complete")
