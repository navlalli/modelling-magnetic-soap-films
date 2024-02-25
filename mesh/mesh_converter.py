""" Converting from .msh to .xdmf for use in Fenics 
Must add physical markers for subdomains and external boundaries for this to work 
Note how prune_z=True is called in meshConversion(). This is for 2d meshes. Set 
to false for 3d meshes.
"""

import meshio

def mesh_conversion(mesh_file):
    """ Convert mesh from .msh (created in gmsh) to .xdmf
    mesh_file = name of .msh file (excluding .msh), string 
    """
    mesh_from_file = meshio.read(mesh_dir + mesh_file + ".msh")
    
    def create_mesh(mesh, cell_type, prune_z=False):
        cells = mesh.get_cells_type(cell_type)
        cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
        points = mesh.points[:,:2] if prune_z else mesh.points
        out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read":[cell_data]})
        return out_mesh

    # mesh and cell markers 
    triangle_mesh = create_mesh(mesh_from_file, "triangle", prune_z=True)
    xdmf_file = mesh_file 
    meshio.write(mesh_dir + xdmf_file + ".xdmf", triangle_mesh)
    # facet markers 
    line_mesh = create_mesh(mesh_from_file, "line", prune_z=True)
    meshio.write(mesh_dir + xdmf_file + "Facet.xdmf", line_mesh)


if __name__ == "__main__":
    # mesh_dir = "/home/nav/navScripts/fenicsx/ferro-thickness/mesh/uniform_meniscus15249/"
    # mesh_conversion("uniform_meniscus")
    mesh_dir = "/home/nav/navScripts/fenicsx/ferro-thickness/mesh/radial_refinement1039200/"
    mesh_conversion("radial_refinement")
    # mesh_dir = "/home/nav/navScripts/fenicsx/ferro-thickness/mesh/radial_unit_80006/"
    # mesh_conversion("radial_unit")
    
    print("Finished successfully")
