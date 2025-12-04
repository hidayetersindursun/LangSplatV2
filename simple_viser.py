import argparse
import time
import viser
import numpy as np
import os
from plyfile import PlyData

def load_ply(path, max_sh_degree=3):
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    return {
        "xyz": xyz,
        "opacity": opacities,
        "features_dc": features_dc,
        "features_rest": features_extra,
        "scaling": scales,
        "rotation": rots,
        "sh_degree": max_sh_degree
    }

def build_rotation(r):
    norm = np.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])
    q = r / norm[:, None]
    R = np.zeros((r.shape[0], 3, 3))
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
    L = np.zeros((scaling.shape[0], 3, 3))
    R = build_rotation(rotation)
    s = scaling_modifier * scaling
    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]
    L = R @ L
    return L @ L.transpose(0, 2, 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply_path", type=str, required=True, help="Path to the .ply file")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    print(f"Loading PLY from {args.ply_path}...")
    data = load_ply(args.ply_path)
    
    server = viser.ViserServer(port=args.port)
    print(f"Viser server started at http://localhost:{args.port}")

    # Add Gaussian Splats to the scene
    # Using server.scene as per modern Viser API
    # We need to compute covariances manually as Viser expects them
    
    print("Computing covariances...")
    scales = np.exp(data["scaling"])
    rots = data["rotation"]
    covs = build_covariance_from_scaling_rotation(scales, 1.0, rots)
    
    # Viser expects upper triangle of covariance matrix or full 3x3?
    # The signature says 'covariances: np.ndarray'. Usually 3x3 or flattened.
    # Let's assume (N, 3, 3).
    
    # Also, Viser expects colors in [0, 1] or [0, 255]? Usually [0, 1] for float.
    # SH DC is roughly RGB. We might need to sigmoid or clip.
    # SH_0 factor:
    SH_C0 = 0.28209479177387814
    rgbs = data["features_dc"][:, :, 0] * SH_C0 + 0.5
    rgbs = np.clip(rgbs, 0, 1)

    server.scene.add_gaussian_splats(
        "/gaussians",
        centers=data["xyz"],
        rgbs=rgbs,
        opacities=data["opacity"].reshape(-1, 1),
        covariances=covs
    )

    print("Scene loaded! You can now navigate in the browser.")
    
    while True:
        time.sleep(1.0)

if __name__ == "__main__":
    main()
