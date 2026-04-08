import argparse
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch
from PIL import Image
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from torchvision import transforms

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
    if width <= 0 or height <= 0:
        return
    x_stops = max(1, (width - patch_size) // stride + 1)
    y_stops = max(1, (height - patch_size) // stride + 1)
    for yi in range(y_stops):
        for xi in range(x_stops):
            x = min(xi * stride, width - patch_size)
            y = min(yi * stride, height - patch_size)
            yield x, y


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
) -> None:
    h5py = _require_h5py()
    width, height = img.size
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
    args = parser.parse_args()

    input_path = Path(args.input)
    output_h5 = Path(args.output_h5) if args.output_h5 else None
    output_npz = Path(args.output_npz) if args.output_npz else None
    output_npy = Path(args.output_npy) if args.output_npy else None
    output_dir = Path(args.output_dir) if args.output_dir else None

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
            )
            h5_to_grid_npy(tmp_h5, output_npy)
        else:
            h5_to_grid_npy(output_h5, output_npy)


if __name__ == "__main__":
    main()
