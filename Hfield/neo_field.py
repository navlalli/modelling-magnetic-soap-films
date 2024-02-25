""" Produce magnetic H field of neodymium magnet for magnetic soap film modelling """

import numpy as np
import magpylib as magpy
import matplotlib.pyplot as plt 
    
from scipy.spatial.transform import Rotation

def field_viewer(xs, ys, field_cpt, title="", cbar_label=""):
    """ View the cpt of the magnetic field """
    nx = len(xs)
    ny = len(ys)
    # Specify the boundaries of each interval
    x_inter = np.linspace(xs[0], xs[-1], nx+1) 
    y_inter = np.linspace(ys[0], ys[-1], ny+1)
    fig, ax = plt.subplots(constrained_layout=True)
    c = ax.pcolormesh(x_inter, y_inter, field_cpt, cmap='viridis', shading='flat')
    d = fig.colorbar(c, ax=ax)
    ax.set_xlabel("x position (mm)", fontsize=14)
    ax.set_ylabel("y position (mm)", fontsize=14)
    ax.set_title(title, fontsize=14)
    d.set_label(cbar_label, fontsize=14)
    plt.show()

def stream_viewer(grid, field, title=""):
    """ Streamlines of the magnetic field """
    mag_field = np.linalg.norm(field, axis=2)    
    fig, ax = plt.subplots(figsize=(4/3*5,5))
    ax.streamplot(grid[:,:,0], grid[:,:,1], field[:,:,0], field[:,:,1],
                  density=2.0, color=mag_field, linewidth=1, cmap='winter')   
    ax.set_xlabel("x position (mm)", fontsize=14)
    ax.set_ylabel("y position (mm)", fontsize=14)
    ax.set_title(title)
    plt.show()

def check_Hmag(Hmag):
    """ Check Hmag is symmetrical """
    nr = 55
    np.testing.assert_allclose(Hmag[nr, :], Hmag[-nr-1, :])
    line = Hmag[750, :]
    print("Test passed")
    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(np.linspace(0, 1, len(line)), line, 'r-')
    ax.set_xlabel("", fontsize=12)
    ax.set_ylabel("", fontsize=12)
    # ax.legend(loc="best")
    plt.show()

def left_exp():
    """ Create Hfield for domain aligning with experiment """
    # Magnet dimensions
    diam = 38.5  # (mm)
    height = 10.0  # (mm)
    # Glass and film dimensions
    glass_diam = 44.5  # (mm)
    cap_len = 2  # Capillary length (mm)
    thick = 2  # Thickness of glass (mm)
    film_diam = glass_diam - (2 * thick)
    thin_film_diam = film_diam - (2 * cap_len)
    # Create cylindrical magnet
    remanence = 1430  # (mT)
    rotation_object = Rotation.from_euler('y', 90, degrees=True)
    neo = magpy.magnet.Cylinder(
        magnetization = (0, 0, -remanence),  # In CS before rotation 
        dimension = (diam, height),  
        position = (0, 0, 0),
        orientation = rotation_object)  # Magnet extends -height<x<0 and -diam/2<y<diam/2

    print(f"{remanence = :.2f} mT")
    
    # Move magnet so (0, 0) is (0, 0) in the simulations - without neo.move, (0, 0)
    # is the centre of the magnet
    neo.move((-height/2-thick-cap_len, glass_diam/2-thick-cap_len, 0.0))
    print(f"{neo.position = }")
    def plot_magnet():
        """ Plot to check orientation """
        fig = magpy.show(neo, return_fig=True)
        ax = fig.get_axes()[0]
        ax.plot([0.0], [0.0], 'kx', ms=17)
        # ax.plot([10.0], [-2.5-gap], 'rx', ms=30)
        plt.show()

    # plot_magnet()
    # sys.exit(0)

    # Create a grid in the x-y plane at z = 0
    nPoints = 1501
    nx = nPoints
    ny = nPoints
    xs = np.linspace(0.0, thin_film_diam, nx)
    ys = np.linspace(0.0, thin_film_diam, ny)
    # x coords - grid[:,:,0], y coords - grid[:,:,1], z coords - grid[:,:,2] and
    # x varies along cols and y varies along rows
    grid = np.array([[(x,y,0) for x in xs] for y in ys]) 

    Hfield = neo.getH(grid)  # (mT)
    Hx_magpy = Hfield[:,:,0]
    Hy_magpy = Hfield[:,:,1]
    # Hz_magpy = Hfield[:,:,2]    
    Hmag = np.linalg.norm(Hfield, axis=2)    
    i, j = np.unravel_index(Hmag.argmax(), Hmag.shape)
    print(f"{i = } {j = }")
    print(f"{Hmag[i, j] = }")
    print(f"{np.max(Hmag) = }")
    print(f"{np.shape(Hmag) = }")
    check_Hmag(Hmag)
    # print(f"{np.shape(xs) = }")
    # print(f"{np.shape(ys) = }")
    # print(f"{np.shape(Hx_magpy) = }")
    # print(f"{np.shape(Hy_magpy) = }")
    
    field_viewer(xs, ys, Hmag, title="Hmag", cbar_label="Hmag (kA/m)")

    # stream_viewer(grid, Hfield, title="Hmag (kA/m)")
        
    def save_field():
        """ Save x and y positions and Hmag for phoresis work """
        save_dir = "/home/nav/navScripts/fenicsx/ferro-thickness/Hfield/exp_meniscus_left/"
        header = f"x position (mm), yposition (mm) with origin in bottom left of mesh for domain of {thin_film_diam} mm diameter. Saved from {f_name}"
        pos = np.column_stack((xs, ys))
        np.savetxt(save_dir + "pos.txt", pos, header=header)
        header = f"Hmag (kA/m) = sqrt(Hx**2 + Hy**2 + Hz**2), for domain of {thin_film_diam} mm diameter. Saved from {f_name}"
        np.savetxt(save_dir + "Hmag.txt", Hmag, header=header)
        print("Saved")

    # save_field()

def left_unit():
    """ Create Hfield for domain of unit diameter """
    # Magnet dimensions
    diam = 10.0  # (mm)
    height = 10.0  # (mm)
    # Glass and film dimensions
    cap_len = 2  # Capillary length (mm)
    thick = 2  # Thickness of glass (mm)
    thin_film_diam = 2  # Thin film diameter (mm)
    glass_diam = 2 * (cap_len + thick) + thin_film_diam  # Glass outer diameter (mm)
    # Create cylindrical magnet
    remanence = 1430  # (mT)
    rotation_object = Rotation.from_euler('y', 90, degrees=True)
    neo = magpy.magnet.Cylinder(
        magnetization = (0, 0, -remanence),  # In CS before rotation 
        dimension = (diam, height),  
        position = (0, 0, 0),
        orientation = rotation_object)  # Magnet extends -height<x<0 and -diam/2<y<diam/2

    print(f"{remanence = :.2f} mT")
    
    # Move magnet so (0, 0) is (0, 0) in the simulations - without neo.move, (0, 0)
    # is the centre of the magnet
    neo.move((-height/2-thick-cap_len, glass_diam/2-thick-cap_len, 0.0))
    print(f"{neo.position = }")
    def plot_magnet():
        """ Plot to check orientation """
        fig = magpy.show(neo, return_fig=True)
        ax = fig.get_axes()[0]
        ax.plot([0.0], [0.0], 'kx', ms=17)
        # ax.plot([10.0], [-2.5-gap], 'rx', ms=30)
        plt.show()

    # plot_magnet()
    # sys.exit(0)

    # Create a grid in the x-y plane at z = 0
    nPoints = 1501
    nx = nPoints
    ny = nPoints
    xs = np.linspace(0.0, thin_film_diam, nx)
    ys = np.linspace(0.0, thin_film_diam, ny)
    # x coords - grid[:,:,0], y coords - grid[:,:,1], z coords - grid[:,:,2] and
    # x varies along cols and y varies along rows
    grid = np.array([[(x,y,0) for x in xs] for y in ys]) 

    Hfield = neo.getH(grid)  # (mT)
    Hx_magpy = Hfield[:,:,0]
    Hy_magpy = Hfield[:,:,1]
    # Hz_magpy = Hfield[:,:,2]    
    Hmag = np.linalg.norm(Hfield, axis=2)    
    check_Hmag(Hmag)
    # print(f"{np.shape(xs) = }")
    # print(f"{np.shape(ys) = }")
    # print(f"{np.shape(Hx_magpy) = }")
    # print(f"{np.shape(Hy_magpy) = }")
    
    # field_viewer(xs, ys, Hmag, title="Hmag", cbar_label="Hmag (kA/m)")

    # stream_viewer(grid, Hfield, title="Hmag (kA/m)")
        
    def save_field():
        """ Save x and y positions and Hmag for phoresis work """
        save_dir = "/home/nav/navScripts/fenicsx/ferro-thickness/Hfield/unit_meniscus_left/"
        header = f"x position (mm), yposition (mm) with origin in bottom left of mesh for domain of {thin_film_diam} mm diameter. Saved from {f_name}"
        pos = np.column_stack((xs, ys))
        np.savetxt(save_dir + "pos.txt", pos, header=header)
        header = f"Hmag (kA/m) = sqrt(Hx**2 + Hy**2 + Hz**2), for domain of {thin_film_diam} mm diameter. Saved from {f_name}"
        np.savetxt(save_dir + "Hmag.txt", Hmag, header=header)
        print("Saved")

    # save_field()

if __name__ == "__main__":
    f_name = "neo_field_horizontal.py"
    left_exp()
    # left_unit()
