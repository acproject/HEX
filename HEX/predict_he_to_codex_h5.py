import argparse
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch
from PIL import Image
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from torchvision import transforms
import cv2
import scipy.ndimage as ndimage

def _import_custom_model():
    try:
        from hex_architecture import CustomModel  # type: ignore
    except Exception:
        from .hex_architecture import CustomModel  # type: ignore
    return CustomModel


def _auto_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    cap = torch.cuda.get_device_capability(0)
    if cap[0] < 7.5:
        return "cpu"
    return "cuda"


def _transform():
    return transforms.Compose(
        [
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
        ]
    )


def _iter_grid_coords(width: int, height: int, patch_size: int, stride: int) -> Iterable[Tuple[int, int]]:
    if width <= 0 or height <= 0 or patch_size <= 0 or stride <= 0:
        return
    if width <= patch_size:
        xs = [0]
    else:
        xs = list(range(0, width - patch_size + 1, stride))
        last = width - patch_size
        if xs[-1] != last:
            xs.append(last)

    if height <= patch_size:
        ys = [0]
    else:
        ys = list(range(0, height - patch_size + 1, stride))
        last = height - patch_size
        if ys[-1] != last:
            ys.append(last)

    for y in ys:
        for x in xs:
            yield int(x), int(y)


BIOMARKER_NAMES = {
    1: "DAPI", 2: "CD8", 3: "Pan-Cytokeratin", 4: "CD3e", 5: "CD163",
    6: "CD20", 7: "CD4", 8: "FAP", 9: "CD138", 10: "CD11c",
    11: "CD66b", 12: "aSMA", 13: "CD68", 14: "Ki67", 15: "CD31",
    16: "Collagen IV", 17: "Granzyme B", 18: "MMP9", 19: "PD-1", 20: "CD44",
    21: "PD-L1", 22: "E-cadherin", 23: "LAG3", 24: "Mac2/Galectin-3", 25: "FOXP3",
    26: "CD14", 27: "EpCAM", 28: "CD21", 29: "CD45", 30: "MPO",
    31: "TCF-1", 32: "ICOS", 33: "Bcl-2", 34: "HLA-E", 35: "CD45RO",
    36: "VISTA", 37: "HIF1A", 38: "CD39", 39: "CD40", 40: "HLA-DR"
}

FLUORESCENT_COLORS = {
    'DAPI': (0, 0, 255),
    'CD8': (0, 255, 0),
    'CD3e': (0, 255, 128),
    'CD4': (128, 255, 0),
    'CD20': (255, 255, 0),
    'CD45': (255, 200, 0),
    'Pan-Cytokeratin': (255, 0, 128),
    'EpCAM': (255, 0, 200),
    'E-cadherin': (200, 0, 255),
    'CD31': (0, 128, 255),
    'CD34': (0, 200, 255),
    'FAP': (255, 128, 0),
    'aSMA': (255, 100, 50),
    'Collagen IV': (150, 100, 50),
    'CD68': (0, 255, 255),
    'CD163': (50, 200, 200),
    'CD11c': (100, 150, 255),
    'CD66b': (255, 150, 100),
    'MPO': (255, 100, 100),
    'Ki67': (255, 0, 0),
    'PD-1': (255, 50, 150),
    'PD-L1': (200, 50, 200),
    'Granzyme B': (255, 80, 80),
    'FOXP3': (180, 80, 255),
    'LAG3': (100, 100, 255),
    'TIM3': (150, 100, 200),
    'VISTA': (80, 150, 255),
    'CD39': (100, 200, 150),
    'HLA-E': (150, 150, 100),
    'HLA-DR': (180, 150, 100),
    'CD44': (200, 100, 150),
    'CD138': (100, 200, 100),
    'MMP9': (200, 150, 50),
    'HIF1A': (50, 100, 200),
    'Bcl-2': (150, 200, 100),
    'TCF-1': (100, 255, 150),
    'ICOS': (200, 255, 100),
    'CD45RO': (255, 200, 100),
    'CD14': (100, 180, 180),
    'CD21': (180, 100, 180),
    'CD40': (200, 150, 150),
}


def _compute_tissue_mask(img: Image.Image, white_thresh: float = 0.92):
    arr_u8 = np.asarray(img, dtype=np.uint8)
    if arr_u8.ndim != 3 or arr_u8.shape[2] != 3:
        return None
    hsv = cv2.cvtColor(arr_u8, cv2.COLOR_RGB2HSV)
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    v = hsv[:, :, 2].astype(np.float32) / 255.0
    mask = ((v < white_thresh) & (s > 0.04)) | (v < 0.85)
    mask = ndimage.binary_closing(mask, structure=np.ones((5, 5), dtype=bool))
    mask = ndimage.binary_fill_holes(mask)
    return mask


def _render_single_marker(spatial_map: np.ndarray, marker: str, tissue_mask: np.ndarray | None, threshold_percentile: float):
    spatial_dist = np.clip(spatial_map.astype(np.float32, copy=False), 0, 1)
    if tissue_mask is not None:
        spatial_dist = spatial_dist * tissue_mask.astype(np.float32, copy=False)

    mask_vals = spatial_dist[tissue_mask] if tissue_mask is not None else spatial_dist[spatial_dist > 0]
    if mask_vals.size:
        p = float(np.clip(threshold_percentile, 0.0, 99.9))
        t = float(np.percentile(mask_vals, p))
        spatial_dist = np.clip(spatial_dist - t, 0, None)

    spatial_dist = np.power(spatial_dist, 0.6)
    spatial_dist = spatial_dist ** 0.8
    spatial_dist = ndimage.gaussian_filter(spatial_dist, sigma=1.2)
    if tissue_mask is not None:
        spatial_dist = spatial_dist * tissue_mask.astype(np.float32, copy=False)

    color = FLUORESCENT_COLORS.get(marker, (255, 255, 255))
    r, g, b = color[2], color[1], color[0]
    brightness = 1.6

    rgb = np.zeros((spatial_dist.shape[0], spatial_dist.shape[1], 3), dtype=np.float32)
    rgb[:, :, 0] = spatial_dist * r * brightness / 255.0
    rgb[:, :, 1] = spatial_dist * g * brightness / 255.0
    rgb[:, :, 2] = spatial_dist * b * brightness / 255.0

    bloom = np.zeros_like(rgb)
    for c in range(3):
        bloom[:, :, c] = ndimage.gaussian_filter(rgb[:, :, c], sigma=5)
    rgb = rgb + bloom * 0.3

    max_val = float(np.max(rgb))
    if max_val > 0:
        for c in range(3):
            channel = np.clip(rgb[:, :, c], 0, max_val)
            nonzero = channel[channel > 0]
            min_v = float(np.percentile(nonzero, 5)) if nonzero.size else 0.0
            max_v = float(np.percentile(channel, 99.5))
            if max_v > min_v:
                channel = (channel - min_v) / (max_v - min_v)
            rgb[:, :, c] = channel

    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb, "RGB")


def _render_fluorescent(img: Image.Image, spatial_maps: dict[str, np.ndarray], selected_markers: list[str], alpha: float, tissue_mask, threshold_percentile: float):
    width, height = img.size
    if tissue_mask is None:
        tissue_mask = _compute_tissue_mask(img, white_thresh=0.92)

    fluorescent_rgba = np.zeros((height, width, 4), dtype=np.float32)

    for marker in selected_markers:
        if marker not in spatial_maps:
            continue

        spatial_dist = spatial_maps[marker].astype(np.float32, copy=False)
        spatial_dist = np.clip(spatial_dist, 0, 1)
        if tissue_mask is not None:
            spatial_dist = spatial_dist * tissue_mask.astype(np.float32, copy=False)

        mask_vals = spatial_dist[tissue_mask] if tissue_mask is not None else spatial_dist[spatial_dist > 0]
        if mask_vals.size:
            p = float(np.clip(threshold_percentile, 0.0, 99.9))
            t = float(np.percentile(mask_vals, p))
            spatial_dist = np.clip(spatial_dist - t, 0, None)

        color = FLUORESCENT_COLORS.get(marker, (255, 255, 255))
        r, g, b = color[2], color[1], color[0]

        spatial_dist = np.power(spatial_dist, 0.6)
        spatial_dist = spatial_dist ** 0.8
        spatial_dist = ndimage.gaussian_filter(spatial_dist, sigma=1.2)
        if tissue_mask is not None:
            spatial_dist = spatial_dist * tissue_mask.astype(np.float32, copy=False)

        brightness = 1.5
        fluorescent_rgba[:, :, 0] += spatial_dist * r * brightness
        fluorescent_rgba[:, :, 1] += spatial_dist * g * brightness
        fluorescent_rgba[:, :, 2] += spatial_dist * b * brightness
        fluorescent_rgba[:, :, 3] += spatial_dist * 255.0

    for c in range(3):
        fluorescent_rgba[:, :, c] = np.clip(fluorescent_rgba[:, :, c], 0, 255)
    fluorescent_rgba[:, :, 3] = np.clip(fluorescent_rgba[:, :, 3], 0, 255)

    rgb_float = fluorescent_rgba[:, :, :3] / 255.0
    bloom = np.zeros_like(rgb_float)
    for c in range(3):
        bloom[:, :, c] = ndimage.gaussian_filter(rgb_float[:, :, c], sigma=8)
    rgb_float = rgb_float + bloom * 0.3

    max_val = float(np.max(rgb_float))
    if max_val > 0:
        rgb_float = rgb_float / max_val
        rgb_float = np.clip(rgb_float * 1.2, 0, 1)

    if tissue_mask is not None:
        m = tissue_mask.astype(np.float32, copy=False)
        rgb_float[:, :, 0] *= m
        rgb_float[:, :, 1] *= m
        rgb_float[:, :, 2] *= m

    fluorescent_only = (rgb_float * 255).astype(np.uint8)
    fluorescent_only_img = Image.fromarray(fluorescent_only, "RGB")

    he_rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    fl_rgb = fluorescent_only.astype(np.float32)
    overlay = (he_rgb * (1.0 - alpha) + fl_rgb * alpha).astype(np.uint8)
    overlay_img = Image.fromarray(overlay, "RGB")
    return overlay_img, fluorescent_only_img


def _is_background(pil_img: Image.Image, white_thresh: float) -> bool:
    if white_thresh >= 1.0:
        return False
    arr = np.asarray(pil_img, dtype=np.float32) / 255.0
    if arr.ndim != 3 or arr.shape[2] != 3:
        return False
    gray = (arr[:, :, 0] + arr[:, :, 1] + arr[:, :, 2]) / 3.0
    return float(gray.mean()) >= white_thresh


def _load_hex_model(
    checkpoint_path: str,
    musk_ckpt_path: str | None,
    device: str,
) -> object:
    CustomModel = _import_custom_model()
    model = CustomModel(visual_output_dim=1024, num_outputs=40, ckpt_path=musk_ckpt_path)
    if checkpoint_path and Path(checkpoint_path).exists():
        sd = torch.load(checkpoint_path, map_location="cpu")
        incompat = model.load_state_dict(sd, strict=False)
        print(f"[load_state_dict] missing_keys={len(incompat.missing_keys)} unexpected_keys={len(incompat.unexpected_keys)}")
    else:
        print(f"[warn] HEX checkpoint not found: {checkpoint_path}")
    model = model.to(device)
    model.eval()
    return model


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _require_h5py():
    try:
        import h5py  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: h5py. Please run this in your project environment where h5py is installed."
        ) from e
    return h5py


def _require_openslide():
    try:
        import openslide  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: openslide-python. Please run this in your project environment where openslide-python is installed."
        ) from e
    return openslide


def predict_to_h5_from_pil(
    img: Image.Image,
    output_h5: Path,
    model: object,
    device: str,
    patch_size: int,
    stride: int,
    batch_size: int,
    white_thresh: float,
    clip_01: bool,
    max_patches: int | None,
    export_png_dir: Path | None,
    export_markers: list[str] | None,
    export_alpha: float,
    export_threshold_percentile: float,
) -> None:
    h5py = _require_h5py()
    width, height = img.size
    coords_iter = _iter_grid_coords(width, height, patch_size, stride)
    tfm = _transform()

    marker_to_idx = {BIOMARKER_NAMES[i]: i - 1 for i in range(1, 41)}
    export_markers = list(export_markers) if export_markers else []
    spatial_maps = {m: np.zeros((height, width), dtype=np.float32) for m in export_markers} if export_png_dir and export_markers else None
    weight_map = np.zeros((height, width), dtype=np.float32) if spatial_maps is not None else None
    tissue_mask = _compute_tissue_mask(img, white_thresh=0.92) if spatial_maps is not None else None

    if spatial_maps is not None:
        yy, xx = np.mgrid[0:patch_size, 0:patch_size]
        cy = (patch_size - 1) / 2.0
        cx = (patch_size - 1) / 2.0
        sigma = max(1.0, patch_size / 3.0)
        weight_full = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma**2)).astype(np.float32)

    _ensure_parent_dir(output_h5)
    with h5py.File(str(output_h5), "w") as f:
        coords_ds = f.create_dataset(
            "coords",
            shape=(0, 2),
            maxshape=(None, 2),
            dtype=np.int32,
            chunks=(4096, 2),
        )
        pred_ds = f.create_dataset(
            "codex_prediction",
            shape=(0, 40),
            maxshape=(None, 40),
            dtype=np.float16,
            chunks=(4096, 40),
        )
        coords_ds.attrs["patch_level"] = 0
        coords_ds.attrs["patch_size"] = int(patch_size)
        coords_ds.attrs["stride"] = int(stride)
        f.attrs["image_size"] = np.asarray([width, height], dtype=np.int32)

        batch_tensors = []
        batch_coords = []
        total_written = 0

        def flush():
            nonlocal total_written, batch_tensors, batch_coords
            if not batch_tensors:
                return
            x = torch.stack(batch_tensors, dim=0).to(device)
            with torch.no_grad(), torch.autocast(device_type=device if device == "cuda" else "cpu", dtype=torch.float16, enabled=(device == "cuda")):
                preds, _ = model(x)
            preds_np = preds.detach().to(torch.float32).cpu().numpy()
            if clip_01:
                preds_np = np.clip(preds_np, 0.0, 1.0)
            preds_np = preds_np.astype(np.float16, copy=False)
            coords_np = np.asarray(batch_coords, dtype=np.int32)

            if spatial_maps is not None and weight_map is not None:
                preds32 = preds.detach().to(torch.float32).cpu().numpy().astype(np.float32, copy=False)
                for (x0, y0), v in zip(batch_coords, preds32):
                    w = weight_full
                    x1 = x0 + patch_size
                    y1 = y0 + patch_size
                    if x1 > width or y1 > height:
                        pw = min(patch_size, width - x0)
                        ph = min(patch_size, height - y0)
                        w = weight_full[:ph, :pw]
                        x1 = x0 + pw
                        y1 = y0 + ph
                    for m in export_markers:
                        idx = marker_to_idx.get(m, None)
                        if idx is None:
                            continue
                        spatial_maps[m][y0:y1, x0:x1] += float(v[idx]) * w
                    weight_map[y0:y1, x0:x1] += w

            n0 = coords_ds.shape[0]
            n1 = n0 + coords_np.shape[0]
            coords_ds.resize((n1, 2))
            pred_ds.resize((n1, 40))
            coords_ds[n0:n1] = coords_np
            pred_ds[n0:n1] = preds_np
            total_written += coords_np.shape[0]
            batch_tensors = []
            batch_coords = []

        for i, (x0, y0) in enumerate(coords_iter):
            if max_patches is not None and total_written + len(batch_coords) >= max_patches:
                break
            patch = img.crop((x0, y0, x0 + patch_size, y0 + patch_size)).convert("RGB")
            if _is_background(patch, white_thresh):
                continue
            batch_tensors.append(tfm(patch))
            batch_coords.append((int(x0), int(y0)))
            if len(batch_tensors) >= batch_size:
                flush()
            if (i + 1) % 2000 == 0:
                print(f"[progress] scanned={i+1} written={total_written}")

        flush()
        print(f"[done] saved={total_written} to {output_h5}")

    if spatial_maps is not None and weight_map is not None and export_png_dir is not None:
        export_png_dir.mkdir(parents=True, exist_ok=True)
        weight_map = np.maximum(weight_map, 1e-8)
        for m in spatial_maps:
            spatial_maps[m] = spatial_maps[m] / weight_map
        overlay_img, fluorescent_only_img = _render_fluorescent(
            img=img,
            spatial_maps=spatial_maps,
            selected_markers=export_markers,
            alpha=float(export_alpha),
            tissue_mask=tissue_mask,
            threshold_percentile=float(export_threshold_percentile),
        )
        stem = output_h5.stem.replace("_pred", "")
        overlay_img.save(str(export_png_dir / f"{stem}_fluorescent_overlay.png"))
        fluorescent_only_img.save(str(export_png_dir / f"{stem}_fluorescent_only.png"))
        marker_dir = export_png_dir / f"{stem}_markers"
        marker_dir.mkdir(parents=True, exist_ok=True)
        for m in export_markers:
            im = _render_single_marker(spatial_maps[m], m, tissue_mask=tissue_mask, threshold_percentile=float(export_threshold_percentile))
            im.save(str(marker_dir / f"{m}.png"))


def predict_to_npz_from_pil(
    img: Image.Image,
    output_npz: Path,
    model: object,
    device: str,
    patch_size: int,
    stride: int,
    batch_size: int,
    white_thresh: float,
    clip_01: bool,
    max_patches: int | None,
) -> None:
    width, height = img.size
    coords_iter = _iter_grid_coords(width, height, patch_size, stride)
    tfm = _transform()

    coords_out = []
    preds_out = []

    batch_tensors = []
    batch_coords = []

    def flush():
        nonlocal batch_tensors, batch_coords
        if not batch_tensors:
            return
        x = torch.stack(batch_tensors, dim=0).to(device)
        with torch.no_grad(), torch.autocast(device_type=device if device == "cuda" else "cpu", dtype=torch.float16, enabled=(device == "cuda")):
            preds, _ = model(x)
        preds_np = preds.detach().to(torch.float32).cpu().numpy()
        if clip_01:
            preds_np = np.clip(preds_np, 0.0, 1.0)
        preds_out.append(preds_np.astype(np.float16, copy=False))
        coords_out.append(np.asarray(batch_coords, dtype=np.int32))
        batch_tensors = []
        batch_coords = []

    total_scanned = 0
    total_written = 0
    for x0, y0 in coords_iter:
        if max_patches is not None and total_written + len(batch_coords) >= max_patches:
            break
        patch = img.crop((x0, y0, x0 + patch_size, y0 + patch_size)).convert("RGB")
        total_scanned += 1
        if _is_background(patch, white_thresh):
            continue
        batch_tensors.append(tfm(patch))
        batch_coords.append((int(x0), int(y0)))
        if len(batch_tensors) >= batch_size:
            flush()
            total_written = sum(x.shape[0] for x in coords_out)
        if total_scanned % 2000 == 0:
            total_written = sum(x.shape[0] for x in coords_out) + len(batch_coords)
            print(f"[progress] scanned={total_scanned} written={total_written}")

    flush()
    coords_np = np.concatenate(coords_out, axis=0) if coords_out else np.zeros((0, 2), dtype=np.int32)
    preds_np = np.concatenate(preds_out, axis=0) if preds_out else np.zeros((0, 40), dtype=np.float16)

    _ensure_parent_dir(output_npz)
    np.savez_compressed(
        str(output_npz),
        coords=coords_np,
        codex_prediction=preds_np,
        patch_level=np.int32(0),
        patch_size=np.int32(patch_size),
        stride=np.int32(stride),
        image_size=np.asarray([width, height], dtype=np.int32),
    )
    print(f"[done] saved={coords_np.shape[0]} to {output_npz}")


def h5_to_grid_npy(h5_path: Path, output_npy: Path) -> None:
    h5py = _require_h5py()
    with h5py.File(str(h5_path), "r") as f:
        coords = f["coords"][:].astype(np.int64, copy=False)
        preds = f["codex_prediction"][:].astype(np.float32, copy=False)
        patch_size = int(f["coords"].attrs.get("patch_size", 224))
        stride = int(f["coords"].attrs.get("stride", patch_size))
        img_size = f.attrs.get("image_size", None)
        if img_size is None:
            raise RuntimeError("Missing image_size attr in h5; this conversion is intended for non-WSI images.")
        width = int(img_size[0])
        height = int(img_size[1])

    x_stops = max(1, (width - patch_size) // stride + 1)
    y_stops = max(1, (height - patch_size) // stride + 1)

    grid_sum = np.zeros((y_stops, x_stops, preds.shape[1]), dtype=np.float32)
    grid_cnt = np.zeros((y_stops, x_stops, 1), dtype=np.float32)

    for (x, y), v in zip(coords, preds):
        xi = int(round(x / stride)) if stride > 0 else 0
        yi = int(round(y / stride)) if stride > 0 else 0
        xi = max(0, min(x_stops - 1, xi))
        yi = max(0, min(y_stops - 1, yi))
        grid_sum[yi, xi] += v
        grid_cnt[yi, xi] += 1.0

    grid = grid_sum / np.maximum(grid_cnt, 1.0)

    _ensure_parent_dir(output_npy)
    np.save(str(output_npy), grid.astype(np.float16))
    print(f"[done] grid saved: {output_npy} shape={grid.shape}")


def predict_to_h5_from_wsi(
    wsi_path: Path,
    output_h5: Path,
    model: object,
    device: str,
    patch_size: int,
    stride: int,
    batch_size: int,
    white_thresh: float,
    clip_01: bool,
    max_patches: int | None,
    export_png_dir: Path | None,
    export_markers: list[str] | None,
    export_alpha: float,
    export_threshold_percentile: float,
) -> None:
    h5py = _require_h5py()
    openslide = _require_openslide()
    slide = openslide.open_slide(str(wsi_path))
    width, height = slide.dimensions
    coords_iter = _iter_grid_coords(width, height, patch_size, stride)
    tfm = _transform()

    _ensure_parent_dir(output_h5)
    with h5py.File(str(output_h5), "w") as f:
        coords_ds = f.create_dataset(
            "coords",
            shape=(0, 2),
            maxshape=(None, 2),
            dtype=np.int32,
            chunks=(4096, 2),
        )
        pred_ds = f.create_dataset(
            "codex_prediction",
            shape=(0, 40),
            maxshape=(None, 40),
            dtype=np.float16,
            chunks=(4096, 40),
        )
        coords_ds.attrs["patch_level"] = 0
        coords_ds.attrs["patch_size"] = int(patch_size)
        coords_ds.attrs["stride"] = int(stride)
        f.attrs["wsi_path"] = str(wsi_path)

        batch_tensors = []
        batch_coords = []
        total_written = 0

        def flush():
            nonlocal total_written, batch_tensors, batch_coords
            if not batch_tensors:
                return
            x = torch.stack(batch_tensors, dim=0).to(device)
            with torch.no_grad(), torch.autocast(device_type=device if device == "cuda" else "cpu", dtype=torch.float16, enabled=(device == "cuda")):
                preds, _ = model(x)
            preds_np = preds.detach().to(torch.float32).cpu().numpy()
            if clip_01:
                preds_np = np.clip(preds_np, 0.0, 1.0)
            preds_np = preds_np.astype(np.float16, copy=False)
            coords_np = np.asarray(batch_coords, dtype=np.int32)

            n0 = coords_ds.shape[0]
            n1 = n0 + coords_np.shape[0]
            coords_ds.resize((n1, 2))
            pred_ds.resize((n1, 40))
            coords_ds[n0:n1] = coords_np
            pred_ds[n0:n1] = preds_np
            total_written += coords_np.shape[0]
            batch_tensors = []
            batch_coords = []

        for i, (x0, y0) in enumerate(coords_iter):
            if max_patches is not None and total_written + len(batch_coords) >= max_patches:
                break
            region = slide.read_region((x0, y0), 0, (patch_size, patch_size)).convert("RGB")
            if _is_background(region, white_thresh):
                continue
            batch_tensors.append(tfm(region))
            batch_coords.append((int(x0), int(y0)))
            if len(batch_tensors) >= batch_size:
                flush()
            if (i + 1) % 2000 == 0:
                print(f"[progress] scanned={i+1} written={total_written}")

        flush()
        slide.close()
        print(f"[done] saved={total_written} to {output_h5}")

    if export_png_dir and export_markers:
        export_png_dir.mkdir(parents=True, exist_ok=True)
        slide = openslide.open_slide(str(wsi_path))
        mag = 40
        try:
            mpp = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X, "0.25"))
            if mpp < 0.2:
                mag = 80
            elif 0.2 <= mpp < 0.3:
                mag = 40
            elif 0.4 <= mpp < 0.6:
                mag = 20
        except Exception:
            pass
        scale_down_factor = int(224 / (40 / mag))
        out_w = width // scale_down_factor + 1
        out_h = height // scale_down_factor + 1
        thumb = slide.get_thumbnail((out_w, out_h)).convert("RGB")
        slide.close()

        with h5py.File(str(output_h5), "r") as f:
            codex_prediction = f["codex_prediction"][:].astype(np.float32, copy=False)
            coords = f["coords"][:].astype(np.int64, copy=False)

        marker_to_idx = {BIOMARKER_NAMES[i]: i - 1 for i in range(1, 41)}
        spatial_maps = {m: np.zeros((out_h, out_w), dtype=np.float32) for m in export_markers}
        for (x, y), v in zip(coords, codex_prediction):
            xi = int(x / scale_down_factor)
            yi = int(y / scale_down_factor)
            if 0 <= xi < out_w and 0 <= yi < out_h:
                for m in export_markers:
                    idx = marker_to_idx.get(m, None)
                    if idx is None:
                        continue
                    spatial_maps[m][yi, xi] = float(v[idx])
        for m in spatial_maps:
            spatial_maps[m] = ndimage.gaussian_filter(spatial_maps[m], sigma=1.0)

        tissue_mask = _compute_tissue_mask(thumb, white_thresh=0.92)
        overlay_img, fluorescent_only_img = _render_fluorescent(
            img=thumb,
            spatial_maps=spatial_maps,
            selected_markers=list(export_markers),
            alpha=float(export_alpha),
            tissue_mask=tissue_mask,
            threshold_percentile=float(export_threshold_percentile),
        )
        stem = output_h5.stem.replace("_pred", "")
        overlay_img.save(str(export_png_dir / f"{stem}_fluorescent_overlay.png"))
        fluorescent_only_img.save(str(export_png_dir / f"{stem}_fluorescent_only.png"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="WSI (.svs) or normal image file (png/jpg/tif)")
    parser.add_argument("--output_h5", default=None, help="Output .h5 path containing coords and codex_prediction")
    parser.add_argument("--output_npz", default=None, help="Output .npz path containing coords and codex_prediction")
    parser.add_argument("--output_npy", default=None, help="Output .npy grid (Y, X, 40) for normal images")
    parser.add_argument("--output_dir", default=None, help="If set and --input is a directory, write outputs into this directory")
    parser.add_argument("--hex_ckpt", default=str(Path(__file__).resolve().parent / "checkpoint.pth"))
    parser.add_argument("--musk_ckpt", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--stride", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--white_thresh", type=float, default=0.92)
    parser.add_argument("--clip_01", action="store_true")
    parser.add_argument("--max_patches", type=int, default=None)
    parser.add_argument("--export_png_dir", default=None, help="If set, also export fluorescent PNGs into this directory")
    parser.add_argument("--export_markers", default=None, help="Comma-separated markers to render, e.g. DAPI,CD8,Pan-Cytokeratin")
    parser.add_argument("--export_alpha", type=float, default=0.7)
    parser.add_argument("--export_sparsity_percentile", type=float, default=80.0)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_h5 = Path(args.output_h5) if args.output_h5 else None
    output_npz = Path(args.output_npz) if args.output_npz else None
    output_npy = Path(args.output_npy) if args.output_npy else None
    output_dir = Path(args.output_dir) if args.output_dir else None
    export_png_dir = Path(args.export_png_dir) if args.export_png_dir else None
    export_markers = [s.strip() for s in args.export_markers.split(",")] if args.export_markers else None

    if input_path.is_dir():
        if output_dir is None:
            output_dir = input_path
        if output_h5 is not None or output_npz is not None or output_npy is not None:
            raise SystemExit("Directory mode: do not pass --output_h5/--output_npz/--output_npy; use --output_dir only.")

    if output_h5 is None and output_npz is None:
        if output_npy is None and not input_path.is_dir():
            raise SystemExit("Please set at least one output: --output_h5 or --output_npz or --output_npy")

    if args.device == "auto":
        device = _auto_device()
    else:
        device = args.device

    model = _load_hex_model(args.hex_ckpt, args.musk_ckpt, device)

    if input_path.is_dir():
        img_files = []
        for ext in ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"):
            img_files.extend(sorted(input_path.glob(ext)))
        if not img_files:
            raise SystemExit(f"No images found in {input_path}")
        for img_path in img_files:
            stem = img_path.stem
            h5_path = output_dir / f"{stem}_pred.h5"
            npy_path = output_dir / f"{stem}_pred_grid.npy"
            img = Image.open(str(img_path)).convert("RGB")
            predict_to_h5_from_pil(
                img=img,
                output_h5=h5_path,
                model=model,
                device=device,
                patch_size=args.patch_size,
                stride=args.stride,
                batch_size=args.batch_size,
                white_thresh=args.white_thresh,
                clip_01=args.clip_01,
                max_patches=args.max_patches,
                export_png_dir=export_png_dir or output_dir,
                export_markers=export_markers,
                export_alpha=args.export_alpha,
                export_threshold_percentile=args.export_sparsity_percentile,
            )
            h5_to_grid_npy(h5_path, npy_path)
        return

    is_wsi = input_path.suffix.lower() in {".svs", ".mrxs", ".ndpi", ".scn"}
    if is_wsi:
        if output_h5 is None:
            raise SystemExit("WSI mode requires --output_h5")
        predict_to_h5_from_wsi(
            wsi_path=input_path,
            output_h5=output_h5,
            model=model,
            device=device,
            patch_size=args.patch_size,
            stride=args.stride,
            batch_size=args.batch_size,
            white_thresh=args.white_thresh,
            clip_01=args.clip_01,
            max_patches=args.max_patches,
            export_png_dir=export_png_dir or (output_h5.parent if output_h5 else None),
            export_markers=export_markers,
            export_alpha=args.export_alpha,
            export_threshold_percentile=args.export_sparsity_percentile,
        )
        return

    img = Image.open(str(input_path)).convert("RGB")
    if output_h5 is not None:
        predict_to_h5_from_pil(
            img=img,
            output_h5=output_h5,
            model=model,
            device=device,
            patch_size=args.patch_size,
            stride=args.stride,
            batch_size=args.batch_size,
            white_thresh=args.white_thresh,
            clip_01=args.clip_01,
            max_patches=args.max_patches,
            export_png_dir=export_png_dir or output_h5.parent,
            export_markers=export_markers,
            export_alpha=args.export_alpha,
            export_threshold_percentile=args.export_sparsity_percentile,
        )
    if output_npz is not None:
        predict_to_npz_from_pil(
            img=img,
            output_npz=output_npz,
            model=model,
            device=device,
            patch_size=args.patch_size,
            stride=args.stride,
            batch_size=args.batch_size,
            white_thresh=args.white_thresh,
            clip_01=args.clip_01,
            max_patches=args.max_patches,
        )
    if output_npy is not None:
        if output_h5 is None:
            tmp_h5 = output_npy.with_suffix(".h5")
            predict_to_h5_from_pil(
                img=img,
                output_h5=tmp_h5,
                model=model,
                device=device,
                patch_size=args.patch_size,
                stride=args.stride,
                batch_size=args.batch_size,
                white_thresh=args.white_thresh,
                clip_01=args.clip_01,
                max_patches=args.max_patches,
                export_png_dir=export_png_dir or output_npy.parent,
                export_markers=export_markers,
                export_alpha=args.export_alpha,
                export_threshold_percentile=args.export_sparsity_percentile,
            )
            h5_to_grid_npy(tmp_h5, output_npy)
        else:
            h5_to_grid_npy(output_h5, output_npy)


if __name__ == "__main__":
    main()
