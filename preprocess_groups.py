import os
import cv2
import json
import numpy as np


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_png_and_mask(image_path: str):
    """
    读取 PNG，并尽量提取前景 mask。
    优先使用 alpha 通道；没有 alpha 时做简单灰度阈值兜底。
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # PNG 带 alpha
    if img.ndim == 3 and img.shape[2] == 4:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
        mask = (alpha > 10).astype(np.uint8) * 255
    else:
        # 普通 RGB / 灰度图
        if img.ndim == 2:
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            bgr = img[:, :, :3]

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    return bgr, mask


def clean_mask(mask: np.ndarray):
    """
    去噪并保留最大连通区域。
    """
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    out = np.zeros_like(mask)
    out[labels == largest_idx] = 255
    return out


def crop_to_mask(image: np.ndarray, mask: np.ndarray, out_size=512, pad=20):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Mask is empty")

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(image.shape[1] - 1, x2 + pad)
    y2 = min(image.shape[0] - 1, y2 + pad)

    crop_img = image[y1:y2 + 1, x1:x2 + 1]
    crop_mask = mask[y1:y2 + 1, x1:x2 + 1]

    crop_img = cv2.resize(crop_img, (out_size, out_size), interpolation=cv2.INTER_AREA)
    crop_mask = cv2.resize(crop_mask, (out_size, out_size), interpolation=cv2.INTER_NEAREST)

    return crop_img, crop_mask


def extract_edge(mask: np.ndarray):
    return cv2.Canny(mask, 50, 150)


def preprocess_one_image(image_path: str, out_dir: str, out_size=512):
    ensure_dir(out_dir)

    name = os.path.splitext(os.path.basename(image_path))[0]

    bgr, mask = load_png_and_mask(image_path)
    mask = clean_mask(mask)
    crop_img, crop_mask = crop_to_mask(bgr, mask, out_size=out_size)
    edge = extract_edge(crop_mask)

    rgb_path = os.path.join(out_dir, f"{name}_rgb.png")
    mask_path = os.path.join(out_dir, f"{name}_mask.png")
    edge_path = os.path.join(out_dir, f"{name}_edge.png")

    cv2.imwrite(rgb_path, crop_img)
    cv2.imwrite(mask_path, crop_mask)
    cv2.imwrite(edge_path, edge)

    return {
        "name": name,
        "rgb_path": rgb_path,
        "mask_path": mask_path,
        "edge_path": edge_path
    }


def preprocess_group(group_dir: str, output_root: str, out_size=512):
    group_name = os.path.basename(group_dir)
    out_dir = os.path.join(output_root, group_name)
    ensure_dir(out_dir)

    png_files = [f for f in sorted(os.listdir(group_dir)) if f.lower().endswith(".png")]
    if len(png_files) == 0:
        print(f"[WARN] {group_name} 里没有 PNG，已跳过")
        return []

    records = []
    for fn in png_files:
        image_path = os.path.join(group_dir, fn)
        try:
            rec = preprocess_one_image(image_path, out_dir, out_size=out_size)
            records.append(rec)
        except Exception as e:
            print(f"[WARN] 处理失败: {image_path} | {e}")

    meta_path = os.path.join(out_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    return records


def preprocess_all_groups(groups_root: str, output_root: str, out_size=512):
    """
    这个函数就是 main.py 里要 import 的函数。
    """
    ensure_dir(output_root)

    if not os.path.exists(groups_root):
        raise FileNotFoundError(f"Image groups root not found: {groups_root}")

    all_info = {}
    group_names = [g for g in sorted(os.listdir(groups_root)) if os.path.isdir(os.path.join(groups_root, g))]

    if len(group_names) == 0:
        raise ValueError(f"No group folders found under: {groups_root}")

    for group_name in group_names:
        group_dir = os.path.join(groups_root, group_name)
        print(f"Preprocessing group: {group_name}")
        all_info[group_name] = preprocess_group(group_dir, output_root, out_size=out_size)

    summary_path = os.path.join(output_root, "all_groups_meta.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_info, f, ensure_ascii=False, indent=2)

    return all_info


if __name__ == "__main__":
    groups_root = "data/image_groups"
    output_root = "outputs/preprocessed_groups"
    result = preprocess_all_groups(groups_root, output_root, out_size=512)
    print(f"Done. Total groups: {len(result)}")