import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
from sympy.core.kind import RaiseNotImplementedError
import torch

from lightglue import SuperPoint
from lightglue.utils import load_image
from DeDoDe.matchers.dual_softmax_matcher import DualSoftMaxMatcher
from lightglue import LightGlue
from util.matching import draw_matches, draw_matching_comparison
import os
from models import TripleNet
from util.Affine_Transformations import generate_strain_tensors, generate_27_strain_tensors
import numpy as np
from matchers.max_similarity import StretcherDualSoftMaxMatcher
import torch

import fenics as fe
import cv2
import pyvista as pv

def get_best_device(verbose = False):
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    if verbose: print (f"Fastest device found is: {device}")
    return device

def load_images(im_path, deformed_im_path, figsize=(10, 5), titles=("Base image", "Deformed image")):
    """
    Load two images, strip alpha channels if present, display them side by side,
    and return numpy arrays (RGB) for downstream processing.
    """
    image = Image.open(im_path)
    deformed_image = Image.open(deformed_im_path)

    image_np = np.array(image)
    if image_np.ndim == 3 and image_np.shape[-1] == 4:
        image_np = image_np[:, :, :3]

    deformed_np = np.array(deformed_image)
    if deformed_np.ndim == 3 and deformed_np.shape[-1] == 4:
        deformed_np = deformed_np[:, :, :3]

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].imshow(image_np)
    axes[0].set_title(titles[0])
    axes[0].axis('off')
    axes[1].imshow(deformed_np)
    axes[1].set_title(titles[1])
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()

    return image_np, deformed_np



def extract_superpoint_keypoints(
    im_path,
    deformed_im_path,
    device,
    num_keypoints=512,
):
    """
    Extract SuperPoint keypoints/descriptors for two images and return
    features plus convenience tensors used later in matching.

    Returns (in order):
    - feats0, feats1: dictionaries from SuperPoint.extract
    - base_keypoints, base_descriptors, base_scores
    - deformed_keypoints, deformed_descriptors, deformed_scores
    - extractor, dsm_matcher (only if return_models=True)
    """
    extractor = SuperPoint(max_num_keypoints=num_keypoints).eval().to(device)

    start = time.time()
    image0 = load_image(im_path).to(device)
    image1 = load_image(deformed_im_path).to(device)
    feats0 = extractor.extract(image0)
    feats1 = extractor.extract(image1)
    end = time.time()
    print(f"Description time: {end - start:.3f}s")

    return (
        feats0,
        feats1
    )


def matching(
    matcher,
    feats0,
    feats1,
    image_np,
    deformed_np,
    device,
    inv_temp=20,
    dsm_threshold=0.01,
    save_dir='Visualisations',
    save_name='matches_baseline.png',
    image = True,
):
    """
    Step 4 baseline matching controlled by `matcher` ('dsm' or 'lightglue').

    Returns:
      baseline_img (PIL.Image.Image), base_matches (torch.Tensor), def_matches (torch.Tensor)
    """
    os.makedirs(save_dir, exist_ok=True)

    if matcher == 'dsm':
        dsm_matcher = DualSoftMaxMatcher()

        base_keypoints = feats0['keypoints']
        base_descriptors = feats0['descriptors']
        base_scores = feats0['keypoint_scores'][0]

        deformed_keypoints = feats1['keypoints']
        deformed_descriptors = feats1['descriptors']
        deformed_scores = feats1['keypoint_scores'][0]

        start = time.time()
        base_matches, def_matches, _ = dsm_matcher.match(
            base_keypoints.to(device), base_descriptors.to(device),
            deformed_keypoints.to(device), deformed_descriptors.to(device),
            P_A=base_scores, P_B=deformed_scores,
            normalize=True, inv_temp=inv_temp, threshold=dsm_threshold,
        )
        end = time.time()
        print(f"Baseline DSM matching time: {end - start:.3f}s")

    elif matcher == 'lightglue':
        lg = LightGlue(features='superpoint').eval().to(device)
        start = time.time()
        out = lg({'image0': feats0, 'image1': feats1})
        matches = out['matches'][0]
        base_matches = feats0['keypoints'][0][matches[:, 0]]
        def_matches = feats1['keypoints'][0][matches[:, 1]]
        end = time.time()
        print(f"Baseline LightGlue matching time: {end - start:.3f}s")

    else:
        print('Matcher not implemented')

    baseline_img = None
    if image:
        from PIL import Image  # local import to avoid circulars
        baseline_img = Image.fromarray(
            draw_matches(image_np, base_matches.cpu(), deformed_np, def_matches.cpu())
        )

        # Show and save
        plt.figure(figsize=(10, 5))
        plt.imshow(baseline_img)
        plt.title('Baseline matches')
        plt.axis('off')
        plt.show()

        baseline_path = os.path.join(save_dir, save_name)
        baseline_img.save(baseline_path)

    return baseline_img, base_matches, def_matches


def stretch_descriptions(
    features,
    device,
    model_path,
    hidden_dim=2048,
    num_layers=2,
):
    """
    Step 5: load the stretcher and produce stretched descriptors for feats0.

    Inputs:
      feats0: SuperPoint features dict for image0
      device: torch device
      model_path: path to stretcher weights
      hidden_dim, num_layers: TripleNet config
      stretch_type: 'normal' or 'only27' (controls strain tensors)

    Returns:
      stretched (torch.Tensor on device) with shape [num_strains, N, D]
    """
    tensors = generate_strain_tensors()

    stretcher = TripleNet(256, 3, hidden_dim=hidden_dim, num_layers=num_layers).float().to(device)
    stretcher.load_state_dict(torch.load(model_path, map_location=device))
    stretcher.eval()

    start = time.time()
    base_descriptions = features['descriptors'][0].cpu()
    with torch.no_grad():
        stretched_descriptions = np.array([stretcher(base_descriptions.to(torch.float32).to(device), torch.tensor(tensor).to(torch.float32).to(device).repeat(len(base_descriptions),1)).cpu() for tensor in tensors])
    stretched_descriptions = torch.tensor(stretched_descriptions).to(device)
    end = time.time()
    print(f"Stretching time: {end - start:.3f}s")

    return stretched_descriptions


def stretched_matching(
    matcher,
    feats0,
    feats1,
    stretched_descriptions,
    image0,
    image1,
    device,
    inv_temp=20,
    dsm_threshold=0.03,
    topk=500,
    baseline_img=None,
    save_dir='Visualisations',
    save_name='stretched_matches.png',
    comparison_name='matches_comparison.png',
    image = True,
):
    """
    Step 6: run matching using stretched descriptors and optionally save a comparison.

    Returns:
      stretched_img (PIL.Image.Image), stretched_matches (torch.Tensor), def_matches_st (torch.Tensor)
    """
    os.makedirs(save_dir, exist_ok=True)

    from PIL import Image  # local import

    if matcher == 'dsm':
        base_keypoints = feats0['keypoints']
        deformed_keypoints = feats1['keypoints']
        deformed_descriptors = feats1['descriptors']
        base_scores = feats0['keypoint_scores'][0]
        deformed_scores = feats1['keypoint_scores'][0]

        matcher_obj = StretcherDualSoftMaxMatcher()
        start = time.time()
        stretched_matches, def_matches_st, _ = matcher_obj.match(
            base_keypoints.to(device), stretched_descriptions.to(device),
            deformed_keypoints.to(device), deformed_descriptors.to(device),
            P_A=base_scores, P_B=deformed_scores,
            normalize=True, inv_temp=inv_temp, threshold=dsm_threshold,
        )
        end = time.time()
        print(f"Stretched DSM matching time: {end - start:.3f}s")

    else:
        lg = LightGlue(features='superpoint').eval().to(device)

        start = time.time()
        best_matches = {}
        best_scores = {}
        base_carrier = {}

        num_strains = stretched_descriptions.shape[0]
        for i in range(num_strains):
            feats0['descriptors'] = stretched_descriptions[i][None].to(device)
            out = lg({'image0': feats0, 'image1': feats1})
            matches = out['matches'][0]
            scores = out['scores'][0]

            base_indices = matches[:, 0]
            base_kps = feats0['keypoints'][0][matches[:, 0]]
            def_kps = feats1['keypoints'][0][matches[:, 1]]

            for j in range(len(base_indices)):
                idx = int(base_indices[j].item())
                score = float(scores[j].item())
                if idx not in best_matches or score > best_scores[idx]:
                    best_matches[idx] = def_kps[j]
                    base_carrier[idx] = base_kps[j]
                    best_scores[idx] = score

        if len(best_matches) == 0:
            raise RuntimeError("No LightGlue matches found across stretched descriptors.")

        import torch
        unique_base = torch.stack(list(base_carrier.values()))
        unique_def = torch.stack(list(best_matches.values()))
        unique_scores = torch.tensor(list(best_scores.values()), device=device)

        k = min(topk, unique_scores.numel())
        top_idx = unique_scores.argsort(descending=True)[:k]
        stretched_matches = unique_base[top_idx]
        def_matches_st = unique_def[top_idx]

        end = time.time()
        print(f"Stretched LightGlue matching time: {end - start:.3f}s")


    stretched_img = None
    if image:
        stretched_img = Image.fromarray(
            draw_matches(image0, stretched_matches.cpu(), image1, def_matches_st.cpu())
        )

        # Visualize and save comparison if provided
        plt.figure(figsize=(10, 5))
        plt.imshow(stretched_img)
        plt.title('Stretched matches')
        plt.axis('off')
        plt.show()

        stretched_path = os.path.join(save_dir, save_name)
        stretched_img.save(stretched_path)

        if baseline_img is not None:
            comparison_path = os.path.join(save_dir, comparison_name)
            draw_matching_comparison(baseline_img, stretched_img, comparison_path)
            print(f"Saved comparison to: {comparison_path}")

    return stretched_img, stretched_matches, def_matches_st


def save_dataset(
    non_deformed_descriptors,
    deformed_descriptors,
    deformation_idx,
    output_path='data/SuperPoint_Descriptors_Dataset_Test.pth',
):
    """
    Convert numpy arrays to torch tensors and save to a single .pth file.
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    torch_descriptors = torch.tensor(non_deformed_descriptors)
    torch_deformed_descriptors = torch.tensor(deformed_descriptors)
    torch_deformations = torch.tensor(deformation_idx)
    torch.save(
        {
            'descriptors': torch_descriptors,
            'deformed_descriptors': torch_deformed_descriptors,
            'transformations': torch_deformations,
        },
        output_path,
    )
    print(f'Saved dataset to: {output_path}')


def create_dataset(
    image_dir,
    device,
    deformations_per_image,
    kp_per_deformation,
    max_images=None
):
    """
    Encapsulate the Step 2 image loop for building SuperPoint descriptor arrays.

    Returns:
      non_deformed_descriptors (np.ndarray), deformed_descriptors (np.ndarray)
    """
    from PIL import Image
    from util.superpoint import sp_detect_and_describe, custom_sample_descriptors
    from util.Affine_Transformations import apply_corotated_strain_with_keypoints

    kp_per_image = deformations_per_image*kp_per_deformation

    image_names = os.listdir(image_dir)
    num_images = len(image_names) if max_images is None else max_images
    print(f'Number of images: {num_images}')
    non_deformed_descriptors = np.zeros((num_images*deformations_per_image*kp_per_deformation, 256))
    deformed_descriptors = np.zeros((num_images*deformations_per_image*kp_per_deformation, 256))

    # Prepare list of deformations
    deformation_grid = generate_strain_tensors()
    deformation_idx = np.tile(np.arange(len(deformation_grid)), (num_images, 1))

    for i, image_name in enumerate(image_names[:max_images]):
        image_path = os.path.join(image_dir, image_name)
        image = Image.open(image_path)
        W, H = image.size
        image = image.resize((W // 2, H // 2))
        image = np.array(image, dtype=np.uint8)

        keypoints, scores, descriptions, dense_descriptions, scales, img_size = sp_detect_and_describe(image, device, kp_per_image)
        top_indices = scores.argsort(descending=True)[:kp_per_image]
        keypoints = keypoints[top_indices]
        descriptions = descriptions[top_indices]

        non_deformed_descriptors[i*kp_per_deformation*deformations_per_image:(i+1)*kp_per_deformation*deformations_per_image] = descriptions.cpu()

        deformations = np.array(deformation_grid)[deformation_idx[i, :]]
        keypoints = keypoints.reshape(deformations_per_image, kp_per_deformation, 2)

        for j, (deformation, keypoint_set) in enumerate(zip(deformations, keypoints)):
            deformed_image, deformed_keypoint_set = apply_corotated_strain_with_keypoints(image, keypoint_set, deformation, dataset_mode=False)

            _, _, _, deformed_dense_descriptions, scales, _ = sp_detect_and_describe(deformed_image, device, 10000)
            sp_keypoints = (torch.tensor(deformed_keypoint_set[0]) + 0.5) * scales - 0.5
            deformed_description_set = custom_sample_descriptors(sp_keypoints.to(torch.float32), deformed_dense_descriptions.cpu()).permute(0, 2, 1)[0]

            deformed_descriptors[
                i*kp_per_deformation*deformations_per_image + j*kp_per_deformation:
                i*kp_per_deformation*deformations_per_image + (j+1)*kp_per_deformation
            ] = deformed_description_set.cpu()

        if i % 100 == 0:
            print(f"Processed {i} images")

    deformation_idx = deformation_idx.flatten()
    parameters = torch.tensor(np.array(deformation_grid)[deformation_idx])

    return non_deformed_descriptors, deformed_descriptors, parameters

def bottom(x, on_boundary):
    return (on_boundary and fe.near(x[1], 0.0))

# Strain function
def epsilon(u):
    return fe.sym(fe.grad(u))

# Stress function
def sigma(u, lambda_, mu):
    return lambda_*fe.div(u)*fe.Identity(2) + 2*mu*epsilon(u)

def von_mises_stress(sigma):
    sigma_11 = sigma[0, 0]
    sigma_22 = sigma[1, 1]
    sigma_12 = sigma[0, 1]  # Shear component

    return fe.sqrt(sigma_11**2 + sigma_22**2 - sigma_11 * sigma_22 + 3 * sigma_12**2)

def von_mises_strain(E,u):
    """Compute von Mises strain from Green-Lagrange strain tensor."""
    dim = len(u)  # Dimension (2D or 3D)
    if dim == 2:
        return fe.sqrt(0.5 * ((E[0, 0] - E[1, 1])**2 + E[0, 0]**2 + E[1, 1]**2 + 6 * E[0, 1]**2))
    elif dim == 3:
        return fe.sqrt(0.5 * ((E[0, 0] - E[1, 1])**2 + (E[1, 1] - E[2, 2])**2 + 
                              (E[2, 2] - E[0, 0])**2 + 6 * (E[0, 1]**2 + E[1, 2]**2 + E[2, 0]**2)))

def create_deformed_medical_image_pair(image_dir, deformed_image_dir, g_zy=12e6, g_zx=1e6):

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
    g_zy = g_zy
    g_zx = g_zx
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

    # --------------------
    # Post-process
    # --------------------

    F = fe.Identity(len(u)) + fe.grad(u)
    E = 0.5 * (F.T * F - fe.Identity(len(u)))
    V_scalar = fe.FunctionSpace(mesh, "P", 1)  # Scalar function space
    strain_vm = fe.project(von_mises_strain(E,u), V_scalar)

    displacements_at_vertices = np.array([u(x) for x in mesh.coordinates()])

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

    for i, point in enumerate(plane_points):
        point[:2] += displacements_at_vertices[i]  # Apply x and y displacements

    plane.points = plane_points

    # Load a texture image
    texture = pv.read_texture(image_dir)

    # Extract the image size
    img = cv2.imread(image_dir)
    img_height, img_width = img.shape[:2]



    # Apply the texture and visualize
    plotter = pv.Plotter(off_screen=True)
    plotter.enable_anti_aliasing()
    plotter.enable_image_style()


    # Calculate mesh bounds (for your PyVista plane, which represents the mesh)
    bounds = plane.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
    x_center = (bounds[0] + bounds[1]) / 2.0
    y_center = (bounds[2] + bounds[3]) / 2.0

    bottom_left = (bounds[0], bounds[2])

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
    plotter.add_mesh(plane, texture=texture, interpolate_before_map=True)
    cv2.imwrite(deformed_image_dir, cv2.cvtColor(plotter.screenshot(), cv2.COLOR_RGB2BGR))

    base_image = Image.open(image_dir)
    deformed_image = Image.open(deformed_image_dir)
    W, H = base_image.size
    # print(W,H)
    dW, dH = deformed_image.size
    # print(dW,dH)
    base_image = np.array(base_image, dtype=np.uint8)
    if base_image.shape[-1] == 4:
        base_image = base_image[:,:,:3]
    deformed_image = np.array(deformed_image, dtype=np.uint8)
    if deformed_image.shape[-1] == 4:
        deformed_image = deformed_image[:,:,:3]

    fig, axes = plt.subplots(1, 2, figsize=(10,5))
    axes[0].imshow(base_image)
    axes[0].set_title("Base Image")
    axes[0].axis('off')
    axes[1].imshow(deformed_image)
    axes[1].set_title("Deformed Image")
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()

    deformation_info = {
        'W': W,
        'H': H,
        'dW': dW,
        'dH': dH,
        'u': u,
        'new_lx': new_lx,
        'new_ly': new_ly,
        'bottom_left': bottom_left,
    }

    return base_image, deformed_image, deformation_info

def track_pixel_displacement(pixel_coords, deformation_info):
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
    # Unpack deformation info
    img_width = deformation_info['W']
    img_height = deformation_info['H']
    new_img_width = deformation_info['dW']
    new_img_height = deformation_info['dH']
    u = deformation_info['u']
    new_l_x = deformation_info['new_lx']
    new_l_y = deformation_info['new_ly']
    bottom_left = deformation_info['bottom_left']
    l_x = 10
    l_y = 10

    # Convert pixel coordinates to physical space
    x = (pixel_coords[0] / img_width) * l_x
    y = (1-(pixel_coords[1] / img_height)) * l_y
    
    # Evaluate displacement field at (x, y)
    u_disp = u((x, y))  # Returns (u_x, u_y)
    
    # Compute deformed position
    x_new = x + u_disp[0]
    y_new = y + u_disp[1]
    
    x_new -= bottom_left[0]
    y_new -= bottom_left[1]

    # Convert back to pixel space
    i_new = (x_new / new_l_x) * new_img_width
    j_new = (1-(y_new / new_l_y)) * new_img_height
    
    return (i_new, j_new)

def evaluate_matches(base_matches, deformed_matches, deformation_info):
    gt_pixel_coords = np.array([track_pixel_displacement(pixel, deformation_info) for pixel in base_matches.cpu()])
    distances = (deformed_matches.cpu() - gt_pixel_coords).norm(dim=1)
    good = (distances < 5).sum().item()
    total = len(distances)
    accuracy = good / total
    print(f'Baseline Accuracy: {accuracy} ({good} / {total})')
    return distances