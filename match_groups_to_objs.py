import os
import json
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_binary(path: str):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return (img > 0).astype(np.uint8)


def resize_binary(img: np.ndarray, size=(512, 512)):
    out = cv2.resize((img * 255).astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
    return (out > 0).astype(np.uint8)


def compute_iou(a: np.ndarray, b: np.ndarray):
    inter = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    if union == 0:
        return 0.0
    return inter / union


def extract_primary_contour(binary: np.ndarray):
    cnts, _ = cv2.findContours((binary * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return None
    return max(cnts, key=cv2.contourArea)


def safe_hu_distance(contour_a, contour_b):
    if contour_a is None or contour_b is None:
        return 1e6
    return cv2.matchShapes(contour_a, contour_b, cv2.CONTOURS_MATCH_I1, 0.0)


def hu_to_similarity(hu_distance: float):
    return 1.0 / (1.0 + hu_distance)


def load_group_meta(preprocessed_root: str):
    groups = {}
    for group_name in sorted(os.listdir(preprocessed_root)):
        group_dir = os.path.join(preprocessed_root, group_name)
        meta_path = os.path.join(group_dir, "meta.json")
        if os.path.isdir(group_dir) and os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                groups[group_name] = json.load(f)
    return groups


def load_render_meta(render_root: str):
    models = {}
    for model_name in sorted(os.listdir(render_root)):
        model_dir = os.path.join(render_root, model_name)
        meta_path = os.path.join(model_dir, "meta.json")
        if os.path.isdir(model_dir) and os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                models[model_name] = json.load(f)
    return models


def load_group_records(group_meta):
    loaded_groups = {}
    for group_name, image_records in group_meta.items():
        loaded_records = []
        for image_record in image_records:
            mask = load_binary(image_record["mask_path"])
            edge = load_binary(image_record["edge_path"])
            loaded_records.append(
                {
                    "name": image_record["name"],
                    "mask": mask,
                    "edge": edge,
                    "contour": extract_primary_contour(mask),
                }
            )
        loaded_groups[group_name] = loaded_records
    return loaded_groups


def load_model_views(model_views):
    loaded_views = []
    for view in model_views:
        mask = load_binary(view["mask_path"])
        edge = load_binary(view["edge_path"])
        loaded_views.append(
            {
                "view_id": view["view_id"],
                "azimuth": view["azimuth"],
                "elevation": view["elevation"],
                "mask": mask,
                "edge": edge,
                "contour": extract_primary_contour(mask),
            }
        )
    return loaded_views


def ensure_same_shape(img: np.ndarray, ref: np.ndarray):
    if img.shape == ref.shape:
        return img
    return resize_binary(img, size=(ref.shape[1], ref.shape[0]))


def match_one_image_to_one_view(image_record, view_record):
    img_mask = ensure_same_shape(image_record["mask"], view_record["mask"])
    img_edge = ensure_same_shape(image_record["edge"], view_record["edge"])
    view_mask = view_record["mask"]
    view_edge = view_record["edge"]

    shape_iou = compute_iou(img_mask, view_mask)
    edge_iou = compute_iou(img_edge, view_edge)
    hu_dist = safe_hu_distance(image_record["contour"], view_record["contour"])
    hu_score = hu_to_similarity(hu_dist)

    total_score = 0.45 * shape_iou + 0.30 * edge_iou + 0.25 * hu_score

    return {
        "shape_iou": float(shape_iou),
        "edge_iou": float(edge_iou),
        "hu_score": float(hu_score),
        "total_score": float(total_score)
    }


def score_one_image_against_one_model(image_record, model_views):
    best_view = None
    for view in model_views:
        try:
            score_info = match_one_image_to_one_view(image_record, view)

            record = {
                "view_id": view["view_id"],
                "azimuth": view["azimuth"],
                "elevation": view["elevation"],
                **score_info
            }

            if best_view is None or record["total_score"] > best_view["total_score"]:
                best_view = record
        except Exception as e:
            print(f"[WARN] view compare failed | view_id={view.get('view_id', -1)} | {e}")

    if best_view is None:
        best_view = {
            "view_id": -1,
            "azimuth": None,
            "elevation": None,
            "shape_iou": 0.0,
            "edge_iou": 0.0,
            "hu_score": 0.0,
            "total_score": 0.0
        }

    return best_view


def score_one_group_against_one_model(group_records, model_views):
    per_image_best = []
    image_scores = []

    for image_record in group_records:
        best_view = score_one_image_against_one_model(image_record, model_views)
        per_image_best.append({
            "image_name": image_record["name"],
            "best_view": best_view
        })
        image_scores.append(best_view["total_score"])

    if len(image_scores) == 0:
        mean_score = 0.0
        min_score = 0.0
        group_score = 0.0
    else:
        mean_score = float(np.mean(image_scores))
        min_score = float(np.min(image_scores))
        group_score = 0.7 * mean_score + 0.3 * min_score

    return {
        "group_score": group_score,
        "mean_score": mean_score,
        "min_score": min_score,
        "per_image_best": per_image_best
    }


def build_score_matrix(preprocessed_root: str, render_root: str, output_dir: str):
    ensure_dir(output_dir)

    group_meta = load_group_meta(preprocessed_root)
    model_meta = load_render_meta(render_root)

    group_names = sorted(group_meta.keys())
    model_names = sorted(model_meta.keys())

    if len(group_names) == 0:
        raise ValueError("No preprocessed groups found.")
    if len(model_names) == 0:
        raise ValueError("No rendered models found.")

    print("Loading preprocessed group features into memory...")
    groups = load_group_records(group_meta)

    score_matrix = np.zeros((len(group_names), len(model_names)), dtype=np.float32)
    detailed_scores = {}

    total_tasks = len(group_names) * len(model_names)
    task_id = 0

    for group_name in group_names:
        detailed_scores[group_name] = {}

    for j, model_name in enumerate(model_names):
        print(f"Loading rendered views for model {model_name}...")
        model_views = load_model_views(model_meta[model_name])

        for i, group_name in enumerate(group_names):
            task_id += 1
            print(f"Scoring [{task_id}/{total_tasks}] {group_name} vs {model_name}")

            group_records = groups[group_name]
            score_info = score_one_group_against_one_model(group_records, model_views)

            score_matrix[i, j] = score_info["group_score"]
            detailed_scores[group_name][model_name] = score_info

    np.save(os.path.join(output_dir, "score_matrix.npy"), score_matrix)

    with open(os.path.join(output_dir, "group_names.json"), "w", encoding="utf-8") as f:
        json.dump(group_names, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, "model_names.json"), "w", encoding="utf-8") as f:
        json.dump(model_names, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, "detailed_scores.json"), "w", encoding="utf-8") as f:
        json.dump(detailed_scores, f, ensure_ascii=False, indent=2)

    return group_names, model_names, score_matrix, detailed_scores


def hungarian_assign(group_names, model_names, score_matrix, detailed_scores, output_dir: str):
    ensure_dir(output_dir)

    cost_matrix = -score_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    assignments = []
    for r, c in zip(row_ind, col_ind):
        group_name = group_names[r]
        model_name = model_names[c]
        score = float(score_matrix[r, c])

        candidates = []
        for j, m in enumerate(model_names):
            candidates.append({
                "model_name": m,
                "score": float(score_matrix[r, j])
            })
        candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)

        assignments.append({
            "group_name": group_name,
            "assigned_model": model_name,
            "score": score,
            "top3_candidates": candidates[:3],
            "details": detailed_scores[group_name][model_name]
        })

    final_result = {"assignments": assignments}

    with open(os.path.join(output_dir, "final_assignments.json"), "w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)

    return final_result
