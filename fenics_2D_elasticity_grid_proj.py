import fenics as fe
import matplotlib.pyplot as plt
import numpy as np
import cv2
from util.image import draw_keypoints


import pyvista as pv

#https://fenics-solid-tutorial.readthedocs.io/en/latest/2DPlaneStrain/2D_Elasticity.html#

# --------------------
# Functions and classes
# --------------------
def bottom(x, on_boundary):
    return (on_boundary and fe.near(x[1], 0.0))

# Strain function
def epsilon(u):
    return fe.sym(fe.grad(u))

# Stress function
def sigma(u, lambda_, mu):
    return lambda_*fe.div(u)*fe.Identity(2) + 2*mu*epsilon(u)

def create_deformed_medical_image_pair(image_dir, deformed_image_dir):

    # --------------------
    # Parameters
    # --------------------

    # Density
    rho = fe.Constant(200.0)

    # Young's modulus and Poisson's ratio
    E = 0.01e9
    nu = 0.4

    # Lame's constants
    lambda_ = E*nu/(1+nu)/(1-2*nu)
    mu = E/2/(1+nu)

    l_x, l_y = 10.0, 10.0  # Domain dimensions
    n_x, n_y = 50, 50  # Number of elements

    # Load
    g_zy = 12e6
    g_zx = 1e6
    b_z = 000.0
    g = fe.Constant((g_zx, g_zy))
    b = fe.Constant((0.0, b_z))

    # Model type
    model = "plane_strain"
    if model == "plane_stress":
        lambda_ = 2*mu*lambda_/(lambda_+2*mu)

    # --------------------
    # Geometry
    # --------------------
    mesh = fe.RectangleMesh(fe.Point(0.0, 0.0), fe.Point(l_x, l_y), n_x, n_y)


    # Definition of Neumann condition domain
    boundaries = fe.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)

    top = fe.AutoSubDomain(lambda x: fe.near(x[1], l_y))

    top.mark(boundaries, 1)
    ds = fe.ds(subdomain_data=boundaries)

    # --------------------
    # Function spaces
    # --------------------
    V = fe.VectorFunctionSpace(mesh, "CG", 1)
    u_tr = fe.TrialFunction(V)
    u_test = fe.TestFunction(V)

    # --------------------
    # Boundary conditions
    # --------------------
    bc = fe.DirichletBC(V, fe.Constant((0.0, 0.0)), bottom)

    # --------------------
    # Weak form
    # --------------------
    a = fe.inner(sigma(u_tr, lambda_, mu), epsilon(u_test))*fe.dx
    l = rho*fe.dot(b, u_test)*fe.dx + fe.inner(g, u_test)*ds(1)

    # --------------------
    # Solver
    # --------------------
    u = fe.Function(V)
    A_ass, L_ass = fe.assemble_system(a, l, bc)

    fe.solve(A_ass, u.vector(), L_ass)

    print(np.amax(u.vector()[:]))

    # --------------------
    # Post-process
    # --------------------
    fe.plot(u, mode="displacement")
    fe.plot(mesh)
    # plt.show()

    displacements_at_vertices = np.array([u(x) for x in mesh.coordinates()])

    #updated_coords = mesh.coordinates() + displacements_at_vertices
    #mesh.coordinates()[:] = updated_coords
    #fe.plot(mesh, title="Deformed Mesh")
    #plt.show()


    # --------------------
    # Create PyVista Plane
    # --------------------
    plane = pv.Plane(
        center=(l_x / 2, l_y / 2, 0),
        i_size=l_x,
        j_size=l_y,
        i_resolution=n_x,
        j_resolution=n_y,
    )


    # Apply displacements to the PyVista Plane
    plane_points = plane.points
    #print(plane_points)

    for i, point in enumerate(plane_points):
        point[:2] += displacements_at_vertices[i]  # Apply x and y displacements

    plane.points = plane_points
    #print(plane.points)

    # Load a texture image
    texture = pv.read_texture(image_dir)

    # Extract the image size
    img = cv2.imread(image_dir)
    img_height, img_width = img.shape[:2]



    # Apply the texture and visualize
    # plotter = pv.Plotter(window_size=[400, 800])
    plotter = pv.Plotter(off_screen=True)


    # Calculate mesh bounds (for your PyVista plane, which represents the mesh)
    bounds = plane.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
    x_center = (bounds[0] + bounds[1]) / 2.0
    y_center = (bounds[2] + bounds[3]) / 2.0

    # print(bounds)
    new_lx = bounds[1] - bounds[0]
    new_ly = bounds[3] - bounds[2]

    # Set up the camera for an orthographic (parallel) projection
    plotter.camera.parallel_projection = True
    plotter.camera.focal_point = (x_center, y_center, 0)
    # With parallel projection, the camera's z position is arbitrary; here we choose 1
    plotter.camera.position = (x_center, y_center, 1)
    plotter.camera.up = (0,1,0)

    # Set the parallel scale to half of the mesh's height
    # (This ensures the mesh fits exactly in the vertical direction)
    plotter.camera.parallel_scale = (bounds[3] - bounds[2]) / 2.0

    # Adjust the render window size to match the mesh aspect ratio exactly
    aspect_ratio = (bounds[1] - bounds[0]) / (bounds[3] - bounds[2])
    window_height = int(img_height * ((bounds[3] - bounds[2])/l_y))  # for example, choose a height in pixels
    window_width = int(window_height * aspect_ratio)
    plotter.window_size = (window_width, window_height)

    # Render and capture the screenshot
    #plotter.add_mesh(plane, style='wireframe')
    plotter.add_mesh(plane, texture=texture)
    #plotter.show_axes()
    #plotter.show()
    plotter.show(screenshot=deformed_image_dir)

    return u, new_lx, new_ly

# u, new_lx, new_ly = create_deformed_medical_image_pair("data/medical_deformed/brain.png")

def track_pixel_displacement(u, pixel_coords, img_width, img_height, new_img_width, new_img_height, l_x, l_y, new_l_x, new_l_y):
    """
    Given the displacement field `u`, track where a pixel at `pixel_coords` moves after deformation.
    
    Args:
        u: FEniCS displacement function.
        pixel_coords: Tuple (i, j) representing pixel coordinates in the original image.
        img_width: Width of the image.
        img_height: Height of the image.
        l_x: Width of the mesh domain.
        l_y: Height of the mesh domain.
    
    Returns:
        (i', j'): The new pixel coordinates in the deformed image.
    """
    # Convert pixel coordinates to physical space
    x = (pixel_coords[0] / img_width) * l_x
    y = (1-(pixel_coords[1] / img_height)) * l_y
    
    # Evaluate displacement field at (x, y)
    u_disp = u((x, y))  # Returns (u_x, u_y)
    
    # Compute deformed position
    x_new = x + u_disp[0]
    y_new = y + u_disp[1]
    
    # Convert back to pixel space
    i_new = (x_new / new_l_x) * new_img_width
    j_new = (1-(y_new / new_l_y)) * new_img_height
    
    return (i_new, j_new)

# brain = cv2.imread('data/medical_deformed/brain.png')[:, :, ::-1]
# deformed_brain = cv2.imread('data/medical_deformed/deformed_brain.png')[:,:,::-1]

# img_width, img_height = brain.shape[1], brain.shape[0]
# new_img_width, new_img_height = deformed_brain.shape[1], deformed_brain.shape[0]

# # Example usage:
# original_pixel = (1024, 1024)  # Example pixel in the original image
# deformed_pixel = track_pixel_displacement(u, original_pixel, img_width, img_height, new_img_width, new_img_height, l_x=10, l_y=10, new_l_x=new_lx, new_l_y=new_ly)
# print(f"Pixel {original_pixel} moved to {deformed_pixel}")

# cv2.imwrite('data/medical_deformed/brain_point.png', draw_keypoints(brain, np.array([original_pixel])))
# cv2.imwrite('data/medical_deformed/brain_point_deformed.png', draw_keypoints(deformed_brain, np.array([deformed_pixel])))