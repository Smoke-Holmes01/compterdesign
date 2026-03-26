import os
import json
import math

if "PYOPENGL_PLATFORM" not in os.environ and not os.getenv("DISPLAY"):
    os.environ["PYOPENGL_PLATFORM"] = "egl"

import cv2
import trimesh
import numpy as np
import pyrender


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def normalize_mesh(mesh: trimesh.Trimesh):
    """
    居中并统一缩放。
    """
    mesh = mesh.copy()

    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2.0
    mesh.vertices -= center

    extent = bounds[1] - bounds[0]
    scale = np.max(extent)
    if scale > 0:
        mesh.vertices /= scale

    return mesh


def look_at(camera_position, target=np.array([0, 0, 0]), up=np.array([0, 0, 1])):
    camera_position = np.array(camera_position, dtype=np.float64)
    target = np.array(target, dtype=np.float64)
    up = np.array(up, dtype=np.float64)

    forward = target - camera_position
    norm_forward = np.linalg.norm(forward)
    if norm_forward < 1e-8:
        raise ValueError("Invalid camera position: same as target")
    forward /= norm_forward

    right = np.cross(forward, up)
    norm_right = np.linalg.norm(right)
    if norm_right < 1e-8:
        up = np.array([0, 1, 0], dtype=np.float64)
        right = np.cross(forward, up)
        norm_right = np.linalg.norm(right)
    right /= norm_right

    true_up = np.cross(right, forward)
    true_up /= np.linalg.norm(true_up)

    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = -forward
    pose[:3, 3] = camera_position
    return pose


def render_mask(mesh: trimesh.Trimesh, azimuth_deg: float, elevation_deg: float,
                image_size=512, distance=2.5):
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.3, 0.3, 0.3])

    render_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene.add(render_mesh)

    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)

    x = distance * math.cos(el) * math.cos(az)
    y = distance * math.cos(el) * math.sin(az)
    z = distance * math.sin(el)

    cam_pose = look_at([x, y, z])

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    scene.add(camera, pose=cam_pose)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=cam_pose)

    renderer = pyrender.OffscreenRenderer(viewport_width=image_size, viewport_height=image_size)
    try:
        _, depth = renderer.render(scene)
    finally:
        renderer.delete()

    mask = (depth > 0).astype(np.uint8) * 255
    return mask


def extract_edge(mask: np.ndarray):
    return cv2.Canny(mask, 50, 150)


def load_mesh(model_path: str):
    mesh_or_scene = trimesh.load(model_path, force="mesh")

    if isinstance(mesh_or_scene, trimesh.Scene):
        geometries = []
        for geom in mesh_or_scene.geometry.values():
            if isinstance(geom, trimesh.Trimesh):
                geometries.append(geom)
        if len(geometries) == 0:
            raise ValueError(f"No valid mesh in scene: {model_path}")
        mesh = trimesh.util.concatenate(geometries)
    else:
        mesh = mesh_or_scene

    return normalize_mesh(mesh)


def render_views_for_model(model_path: str, output_dir: str,
                           azimuth_list=None, elevation_list=None,
                           image_size=512, distance=2.5):
    ensure_dir(output_dir)

    if azimuth_list is None:
        azimuth_list = list(range(0, 360, 10))
    if elevation_list is None:
        elevation_list = [0, 10, 20, 30]

    mesh = load_mesh(model_path)

    metadata = []
    view_id = 0

    for az in azimuth_list:
        for el in elevation_list:
            try:
                mask = render_mask(
                    mesh=mesh,
                    azimuth_deg=az,
                    elevation_deg=el,
                    image_size=image_size,
                    distance=distance
                )
                edge = extract_edge(mask)

                mask_name = f"view_{view_id:04d}_az{az}_el{el}_mask.png"
                edge_name = f"view_{view_id:04d}_az{az}_el{el}_edge.png"

                mask_path = os.path.join(output_dir, mask_name)
                edge_path = os.path.join(output_dir, edge_name)

                cv2.imwrite(mask_path, mask)
                cv2.imwrite(edge_path, edge)

                metadata.append({
                    "view_id": view_id,
                    "azimuth": az,
                    "elevation": el,
                    "mask_path": mask_path,
                    "edge_path": edge_path
                })
                view_id += 1
            except Exception as e:
                print(f"[WARN] Render failed | model={model_path} az={az} el={el} | {e}")

    with open(os.path.join(output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return metadata


def render_all_models(model_dir: str, output_root: str,
                      azimuth_list=None, elevation_list=None,
                      image_size=512, distance=2.5):
    ensure_dir(output_root)

    print(
        "Render backend:"
        f" PYOPENGL_PLATFORM={os.getenv('PYOPENGL_PLATFORM', 'default')}"
        f" DISPLAY={os.getenv('DISPLAY') or 'none'}"
    )

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model dir not found: {model_dir}")

    obj_files = [f for f in sorted(os.listdir(model_dir)) if f.lower().endswith(".obj")]
    if len(obj_files) == 0:
        raise ValueError(f"No OBJ files found under: {model_dir}")

    all_info = {}

    for idx, fn in enumerate(obj_files, start=1):
        model_name = os.path.splitext(fn)[0]
        model_path = os.path.join(model_dir, fn)
        out_dir = os.path.join(output_root, model_name)

        print(f"Rendering [{idx}/{len(obj_files)}]: {model_name}")
        try:
            meta = render_views_for_model(
                model_path=model_path,
                output_dir=out_dir,
                azimuth_list=azimuth_list,
                elevation_list=elevation_list,
                image_size=image_size,
                distance=distance
            )
            all_info[model_name] = meta
        except Exception as e:
            print(f"[WARN] Skip model {model_name}: {e}")

    with open(os.path.join(output_root, "all_models_meta.json"), "w", encoding="utf-8") as f:
        json.dump(all_info, f, ensure_ascii=False, indent=2)

    return all_info


if __name__ == "__main__":
    model_dir = "data/models"
    output_root = "outputs/renders"
    result = render_all_models(model_dir, output_root)
    print(f"Done. Total models: {len(result)}")
