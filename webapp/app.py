"""
HEX/MUSK 可视化Web应用
用于展示MUSK模型处理肺癌类器官H&E图像的过程
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'MUSK'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'hex'))

import torch
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_from_directory, make_response
from pathlib import Path
import base64
from io import BytesIO
import json
import cv2
import colorsys
import threading
import tempfile
import uuid

# MUSK/HEX相关导入
from musk import utils, modeling
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
import torchvision.transforms as transforms
from hex_architecture import CustomModel

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

# 全局变量存储模型
model = None
device = None
MODEL_LOAD_LOCK = threading.Lock()
_VIRTUAL_W = None
_VIRTUAL_B = None

# 生物标志物名称
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

# 生物标志物分类
BIOMARKER_CATEGORIES = {
    "免疫细胞标记": ["CD8", "CD3e", "CD4", "CD20", "CD45", "CD45RO", "FOXP3", "Granzyme B", "PD-1", "LAG3", "ICOS", "TCF-1"],
    "巨噬细胞/髓系": ["CD163", "CD68", "CD11c", "CD14", "CD66b", "MPO", "Mac2/Galectin-3"],
    "肿瘤/上皮标记": ["Pan-Cytokeratin", "EpCAM", "E-cadherin"],
    "基质/血管": ["FAP", "aSMA", "CD31", "Collagen IV", "MMP9"],
    "免疫检查点": ["PD-L1", "PD-1", "LAG3", "VISTA", "CD39", "HLA-E", "HLA-DR"],
    "增殖/凋亡": ["Ki67", "Bcl-2"],
    "其他": ["DAPI", "CD138", "CD44", "CD21", "CD40", "HIF1A"]
}


def load_models(hex_ckpt_path: str | None = None):
    """加载MUSK和HEX模型"""
    global model, device
    with MODEL_LOAD_LOCK:
        if model is not None:
            return

        print("正在加载模型...")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {device}")

        model = CustomModel(visual_output_dim=1024, num_outputs=40)

        if hex_ckpt_path is None:
            hex_ckpt_path = "/home/acproject/workspace/python_projects/HEX/hex/checkpoint.pth"

        if os.path.exists(hex_ckpt_path):
            print(f"加载HEX checkpoint: {hex_ckpt_path}")
            sd = torch.load(hex_ckpt_path, map_location="cpu")
            incompat = model.load_state_dict(sd, strict=False)
            print(f"  missing_keys={len(incompat.missing_keys)} unexpected_keys={len(incompat.unexpected_keys)}")
        else:
            print("⚠️  警告: 未找到HEX checkpoint，使用随机初始化的回归头")

        model.eval()
        if device == "cuda":
            try:
                model = model.to(device)
                with torch.no_grad():
                    _ = model(torch.zeros((1, 3, 384, 384), dtype=torch.float32, device=device))
            except Exception as e:
                if "GET was unable to find an engine to execute this computation" in str(e):
                    torch.backends.cudnn.enabled = False
                    try:
                        model = model.to(device)
                        with torch.no_grad():
                            _ = model(torch.zeros((1, 3, 384, 384), dtype=torch.float32, device=device))
                        print("已禁用 cuDNN（V100/SM70 兼容模式）")
                    except Exception as e2:
                        if os.environ.get("HEX_FORCE_CUDA", "").strip() in ("1", "true", "True"):
                            raise
                        print(f"CUDA 运行不稳定，回退到CPU: {e2}")
                        device = "cpu"
                        model = model.to(device)
                else:
                    if os.environ.get("HEX_FORCE_CUDA", "").strip() in ("1", "true", "True"):
                        raise
                    print(f"CUDA 运行不稳定，回退到CPU: {e}")
                    device = "cpu"
                    model = model.to(device)
        else:
            model = model.to(device)

        print("HEX模型加载完成!")


def get_transform():
    """获取图像变换"""
    return transforms.Compose([
        transforms.Resize((384, 384), interpolation=3, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
    ])


def image_to_base64(img):
    """将PIL图像转换为base64字符串"""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def _rgb_black_to_transparent_rgba(
    img_rgb: Image.Image,
    alpha_floor: float = 0.0,
    alpha_gamma: float = 1.0,
    alpha_mask=None,
) -> Image.Image:
    arr = np.asarray(img_rgb.convert("RGB"), dtype=np.uint8)
    intensity = np.max(arr, axis=2).astype(np.float32) / 255.0
    intensity = np.clip(intensity, 0.0, 1.0)
    if alpha_gamma != 1.0:
        intensity = np.power(intensity, float(alpha_gamma))

    alpha_floor = float(np.clip(alpha_floor, 0.0, 1.0))
    alpha = intensity
    if alpha_floor > 0.0:
        alpha = np.maximum(alpha, alpha_floor)

    if alpha_mask is not None:
        m = np.asarray(alpha_mask)
        if m.shape != alpha.shape:
            m_img = Image.fromarray((m.astype(np.uint8) * 255), mode="L")
            m_img = m_img.resize((alpha.shape[1], alpha.shape[0]), Image.NEAREST)
            m = (np.asarray(m_img, dtype=np.uint8) > 0)
        else:
            m = m.astype(bool, copy=False)
        alpha = alpha * m.astype(np.float32, copy=False)

    alpha_u8 = (np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8)

    rgba = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
    rgba[:, :, :3] = arr
    rgba[:, :, 3] = alpha_u8
    return Image.fromarray(rgba, "RGBA")


def _apply_psf_and_noise_rgb(
    img_rgb: Image.Image,
    psf_sigma: float = 0.0,
    poisson_scale: float = 0.0,
    background_noise_sigma: float = 0.0,
    seed: int | None = None,
) -> Image.Image:
    import scipy.ndimage as ndimage

    arr = np.asarray(img_rgb.convert("RGB"), dtype=np.float32) / 255.0

    psf_sigma = float(psf_sigma)
    if psf_sigma > 0:
        for c in range(3):
            arr[:, :, c] = ndimage.gaussian_filter(arr[:, :, c], sigma=psf_sigma)

    if seed is not None:
        rng = np.random.RandomState(int(seed))
    else:
        rng = np.random

    poisson_scale = float(poisson_scale)
    if poisson_scale > 0:
        x = np.clip(arr * poisson_scale, 0.0, None)
        x = rng.poisson(x).astype(np.float32) / poisson_scale
        arr = np.clip(x, 0.0, 1.0)

    background_noise_sigma = float(background_noise_sigma)
    if background_noise_sigma > 0:
        arr = arr + rng.normal(0.0, background_noise_sigma, size=arr.shape).astype(np.float32)
        arr = np.clip(arr, 0.0, 1.0)

    out = (arr * 255.0).astype(np.uint8)
    return Image.fromarray(out, "RGB")


def _normalize_spatial_maps_zscore(
    spatial_maps: dict,
    tissue_mask,
    ref_stats: dict | None = None,
    z_clip: float = 3.0,
) -> dict:
    z_clip = float(z_clip)
    if z_clip <= 0:
        return spatial_maps

    out = {}
    for name, m in spatial_maps.items():
        x = m.astype(np.float32, copy=False)
        if tissue_mask is not None:
            vals = x[tissue_mask]
        else:
            vals = x.reshape(-1)

        stats = ref_stats.get(name) if isinstance(ref_stats, dict) else None
        if stats and "mean" in stats and "std" in stats and float(stats["std"]) > 0:
            mu = float(stats["mean"])
            sd = float(stats["std"])
        else:
            if vals.size:
                mu = float(np.mean(vals))
                sd = float(np.std(vals))
            else:
                mu = 0.0
                sd = 1.0
            if sd <= 1e-8:
                sd = 1.0

        z = (x - mu) / sd
        z = np.clip(z, -z_clip, z_clip)
        y = (z + z_clip) / (2.0 * z_clip)
        out[name] = y.astype(np.float32, copy=False)

    return out


def _compute_tissue_mask(img, white_thresh: float = 0.92):
    import scipy.ndimage as ndimage

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


def _compute_cell_mask_weak(
    img,
    tissue_mask=None,
    dilate_radius: int = 6,
    min_size: int = 64,
):
    try:
        from skimage.color import rgb2hed
        from skimage.filters import threshold_otsu
        from skimage.morphology import remove_small_objects, closing, opening, disk, dilation
        import scipy.ndimage as ndimage
    except Exception:
        return None

    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    hed = rgb2hed(arr)
    h = hed[:, :, 0].astype(np.float32, copy=False)
    h = h - float(np.min(h))
    denom = float(np.max(h))
    if denom > 1e-8:
        h = h / denom
    else:
        return None

    try:
        t = float(threshold_otsu(h))
    except Exception:
        t = 0.5

    nuclei = h > t
    nuclei = remove_small_objects(nuclei, max_size=int(max(1, min_size)))
    nuclei = opening(nuclei, footprint=disk(1))
    nuclei = closing(nuclei, footprint=disk(2))
    nuclei = ndimage.binary_fill_holes(nuclei)

    r = int(max(0, dilate_radius))
    if r > 0:
        cells = dilation(nuclei, footprint=disk(r))
    else:
        cells = nuclei

    if tissue_mask is not None:
        cells = cells & tissue_mask.astype(bool, copy=False)

    return cells.astype(bool, copy=False)


def extract_features(img_tensor):
    """使用MUSK提取图像特征"""
    global model, device
    
    with torch.no_grad():
        features = model(
            image=img_tensor.to(device),
            with_head=False,
            out_norm=False,
            ms_aug=True,
            return_global=True
        )[0]
    
    return features.cpu().numpy()


def predict_global_hex(img, clip_01=True):
    global model, device
    transform = get_transform()
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        out = model(img_tensor.to(device))
    if isinstance(out, (tuple, list)):
        preds = out[0]
    else:
        preds = out
    preds = preds[0].detach().cpu().numpy().astype(np.float32, copy=False)
    if clip_01:
        preds = np.clip(preds, 0.0, 1.0)
    return {BIOMARKER_NAMES[i]: float(preds[i - 1]) for i in range(1, 41)}


def generate_virtual_proteomics(features):
    """
    基于特征生成虚拟蛋白质组学预测
    这里使用简化的线性映射作为演示
    实际应用中需要加载训练好的HEX模型
    """
    global _VIRTUAL_W, _VIRTUAL_B
    x = np.asarray(features, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    x = x.reshape(x.shape[0], -1)
    x = np.tanh(x)

    in_dim = x.shape[1]
    out_dim = 40
    if _VIRTUAL_W is None or _VIRTUAL_W.shape[0] != in_dim:
        rng = np.random.RandomState(42)
        _VIRTUAL_W = rng.normal(0.0, 1.0 / np.sqrt(max(1, in_dim)), size=(in_dim, out_dim)).astype(np.float32)
        _VIRTUAL_B = rng.normal(0.0, 0.05, size=(out_dim,)).astype(np.float32)

    logits = x @ _VIRTUAL_W + _VIRTUAL_B
    probs = 1.0 / (1.0 + np.exp(-logits))
    probs = np.clip(probs[0], 0.0, 1.0)
    return {BIOMARKER_NAMES[i]: float(probs[i - 1]) for i in range(1, 41)}


def generate_attention_map(img_tensor, features):
    """生成注意力热图（简化版本）"""
    # 获取patch级别的特征
    with torch.no_grad():
        patch_features = model(
            image=img_tensor.to(device),
            with_head=False,
            out_norm=False,
            ms_aug=False,
            return_global=False
        )[0]
    
    # 计算注意力权重
    attention = torch.softmax(torch.sum(patch_features, dim=-1), dim=-1)
    attention = attention.cpu().numpy()
    
    # 重塑为2D热图
    num_patches = int(np.sqrt(attention.shape[1] - 1))  # -1 for CLS token
    if num_patches * num_patches < attention.shape[1] - 1:
        num_patches += 1
    
    heatmap = np.zeros((num_patches, num_patches))
    for i in range(min(attention.shape[1] - 1, num_patches * num_patches)):
        row = i // num_patches
        col = i % num_patches
        heatmap[row, col] = attention[0, i + 1]  # +1 to skip CLS token
    
    return heatmap


@app.route('/')
def index():
    """主页"""
    # 获取示例图片列表
    sample_dir = Path('/home/acproject/workspace/python_projects/HEX/lung_cancer_organoid_HE')
    sample_images = []
    if sample_dir.exists():
        for f in sample_dir.glob('*.tif'):
            sample_images.append({
                'name': f.name,
                'path': str(f)
            })
    
    resp = make_response(render_template('index.html', sample_images=sample_images))
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.route('/static/<path:filename>')
def serve_static(filename):
    resp = make_response(send_from_directory('static', filename))
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.route('/analyze', methods=['POST'])
def analyze():
    """分析图像"""
    try:
        # 确保模型已加载
        load_models()
        
        saved_path = None
        
        # 获取图像
        if 'file' in request.files:
            file = request.files['file']
            img = Image.open(file.stream).convert('RGB')
            
            # 保存上传的文件到临时目录（用于荧光层生成）
            import tempfile
            temp_dir = Path('/home/acproject/workspace/python_projects/HEX/webapp/static/temp')
            temp_dir.mkdir(exist_ok=True)
            temp_path = temp_dir / f"temp_{np.random.randint(100000)}.png"
            img.save(temp_path)
            saved_path = str(temp_path)
            
        elif 'image_path' in request.json:
            img_path = request.json['image_path']
            img = Image.open(img_path).convert('RGB')
            saved_path = img_path
        else:
            return jsonify({'error': '未提供图像'}), 400
        
        # 保存原始图像尺寸
        original_size = img.size
        
        # 预处理
        transform = get_transform()
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            out = model(img_tensor.to(device))
        if isinstance(out, (tuple, list)):
            preds = out[0]
        else:
            preds = out
        preds = preds[0].detach().cpu().numpy().astype(np.float32, copy=False)
        preds = np.clip(preds, 0.0, 1.0)
        predictions = {BIOMARKER_NAMES[i]: float(preds[i - 1]) for i in range(1, 41)}

        # 提取特征
        features = extract_features(img_tensor)
        
        # 生成缩略图
        thumbnail = img.copy()
        thumbnail.thumbnail((400, 400))
        img_base64 = image_to_base64(thumbnail)
        
        # 按类别组织结果
        categorized_predictions = {}
        for category, markers in BIOMARKER_CATEGORIES.items():
            categorized_predictions[category] = {
                marker: predictions[marker] 
                for marker in markers 
                if marker in predictions
            }
        
        # 计算统计信息
        all_values = list(predictions.values())
        stats = {
            'mean': float(np.mean(all_values)),
            'std': float(np.std(all_values)),
            'min': float(np.min(all_values)),
            'max': float(np.max(all_values))
        }
        
        return jsonify({
            'success': True,
            'image': img_base64,
            'predictions': predictions,
            'categorized_predictions': categorized_predictions,
            'stats': stats,
            'feature_shape': features.shape,
            'device': device,
            'saved_path': saved_path
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/batch_analyze', methods=['POST'])
def batch_analyze():
    """批量分析多张图像"""
    try:
        load_models()
        
        results = []
        sample_dir = Path('/home/acproject/workspace/python_projects/HEX/lung_cancer_organoid_HE')
        
        if not sample_dir.exists():
            return jsonify({'error': '示例图片目录不存在'}), 400
        
        for img_path in sorted(sample_dir.glob('*.tif'))[:4]:  # 限制4张
            try:
                img = Image.open(img_path).convert('RGB')
                
                transform = get_transform()
                img_tensor = transform(img).unsqueeze(0)
                
                features = extract_features(img_tensor)
                with torch.no_grad():
                    out = model(img_tensor.to(device))
                if isinstance(out, (tuple, list)):
                    preds = out[0]
                else:
                    preds = out
                preds = preds[0].detach().cpu().numpy().astype(np.float32, copy=False)
                preds = np.clip(preds, 0.0, 1.0)
                predictions = {BIOMARKER_NAMES[i]: float(preds[i - 1]) for i in range(1, 41)}
                
                thumbnail = img.copy()
                thumbnail.thumbnail((200, 200))
                img_base64 = image_to_base64(thumbnail)
                
                results.append({
                    'name': img_path.name,
                    'image': img_base64,
                    'predictions': predictions,
                    'stats': {
                        'mean': float(np.mean(list(predictions.values()))),
                        'std': float(np.std(list(predictions.values())))
                    }
                })
            except Exception as e:
                results.append({
                    'name': img_path.name,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# 荧光颜色映射（模拟真实荧光染料颜色）
FLUORESCENT_COLORS = {
    'DAPI': (0, 0, 255),           # 蓝色 - 核染色
    'CD8': (0, 255, 0),             # 绿色 - 细胞毒性T细胞
    'CD3e': (0, 255, 128),          # 青绿色 - T细胞
    'CD4': (128, 255, 0),           # 黄绿色 - 辅助T细胞
    'CD20': (255, 255, 0),          # 黄色 - B细胞
    'CD45': (255, 200, 0),          # 橙黄色 - 白细胞
    'Pan-Cytokeratin': (255, 0, 128), # 粉红色 - 上皮细胞
    'EpCAM': (255, 0, 200),        # 品红色 - 上皮细胞
    'E-cadherin': (200, 0, 255),   # 紫色 - 上皮标记
    'CD31': (0, 128, 255),         # 天蓝色 - 血管内皮
    'CD34': (0, 200, 255),         # 浅蓝色 - 血管
    'FAP': (255, 128, 0),          # 橙色 - 成纤维细胞
    'aSMA': (255, 100, 50),        # 橙红色 - 平滑肌
    'Collagen IV': (150, 100, 50), # 棕色 - 基底膜
    'CD68': (0, 255, 255),         # 青色 - 巨噬细胞
    'CD163': (50, 200, 200),       # 深青色 - M2巨噬细胞
    'CD11c': (100, 150, 255),      # 浅蓝色 - 树突细胞
    'CD66b': (255, 150, 100),      # 浅橙色 - 中性粒细胞
    'MPO': (255, 100, 100),        # 红色 - 髓过氧化物酶
    'Ki67': (255, 0, 0),           # 红色 - 增殖标记
    'PD-1': (255, 50, 150),        # 粉色 - 免疫检查点
    'PD-L1': (200, 50, 200),       # 紫红色 - 免疫检查点
    'Granzyme B': (255, 80, 80),   # 红色 - 细胞毒性
    'FOXP3': (180, 80, 255),       # 紫色 - Treg
    'LAG3': (100, 100, 255),       # 蓝紫色 - 免疫检查点
    'TIM3': (150, 100, 200),       # 紫色 - 免疫检查点
    'VISTA': (80, 150, 255),       # 浅蓝色 - 免疫检查点
    'CD39': (100, 200, 150),       # 青绿色 - 免疫检查点
    'HLA-E': (150, 150, 100),      # 黄褐色 - MHC
    'HLA-DR': (180, 150, 100),     # 黄褐色 - MHC II
    'CD44': (200, 100, 150),       # 粉紫色 - 干细胞标记
    'CD138': (100, 200, 100),      # 绿色 - 浆细胞
    'MMP9': (200, 150, 50),        # 黄橙色 - 基质金属蛋白酶
    'HIF1A': (50, 100, 200),       # 蓝色 - 缺氧标记
    'Bcl-2': (150, 200, 100),      # 黄绿色 - 抗凋亡
    'TCF-1': (100, 255, 150),      # 青绿色 - T细胞标记
    'ICOS': (200, 255, 100),       # 黄绿色 - 共刺激分子
    'CD45RO': (255, 200, 100),     # 橙黄色 - 记忆T细胞
    'CD14': (100, 180, 180),       # 青色 - 单核细胞
    'CD21': (180, 100, 180),       # 紫色 - B细胞标记
    'CD40': (200, 150, 150),       # 浅红色 - 共刺激分子
}


def generate_fluorescent_layer(img_size, predictions, selected_markers=None, intensity_scale=1.0):
    """
    生成荧光层图像（基于全局预测值，使用随机分布）
    
    改进版本：更真实的多层荧光效果
    
    Args:
        img_size: 图像尺寸 (width, height)
        predictions: 生物标志物预测值字典
        selected_markers: 选中的生物标志物列表，None表示全部
        intensity_scale: 强度缩放因子
    
    Returns:
        荧光层图像 (PIL Image)
    """
    import scipy.ndimage as ndimage
    
    width, height = img_size
    
    # 创建RGBA图像（黑色背景）
    fluorescent_img = np.zeros((height, width, 4), dtype=np.float32)
    
    # 选择要显示的标记物
    if selected_markers is None:
        selected_markers = list(predictions.keys())
    
    # 为每个标记物生成伪随机分布的荧光点
    np.random.seed(42)
    
    for marker in selected_markers:
        if marker not in predictions:
            continue
        
        value = predictions[marker] * intensity_scale
        if value < 0.1:  # 跳过低表达
            continue
        
        # 获取颜色 (BGR -> RGB)
        color = FLUORESCENT_COLORS.get(marker, (255, 255, 255))
        r, g, b = color[2], color[1], color[0]
        
        # 生成模拟的荧光分布
        num_points = int(value * 8000)  # 增加点的数量
        
        # 随机生成点的位置
        points_x = np.random.randint(0, width, num_points)
        points_y = np.random.randint(0, height, num_points)
        
        # 创建密度图
        density = np.zeros((height, width), dtype=np.float32)
        for px, py in zip(points_x, points_y):
            if 0 <= py < height and 0 <= px < width:
                density[py, px] += 1
        
        # 高斯模糊模拟荧光扩散
        density = ndimage.gaussian_filter(density, sigma=8)
        
        # 归一化
        if density.max() > 0:
            density = density / density.max()
        
        # 增强对比度
        density = np.power(density, 0.5) * value
        
        # 累加颜色（加法混合）
        brightness = 1.8
        fluorescent_img[:, :, 0] += density * r * brightness / 255.0  # R
        fluorescent_img[:, :, 1] += density * g * brightness / 255.0  # G
        fluorescent_img[:, :, 2] += density * b * brightness / 255.0  # B
        fluorescent_img[:, :, 3] += density * 255  # Alpha
    
    # 应用发光效果（bloom）
    bloom = np.zeros((height, width, 3), dtype=np.float32)
    for c in range(3):
        bloom[:, :, c] = ndimage.gaussian_filter(fluorescent_img[:, :, c], sigma=10)
    
    # 将发光效果加回原图
    for c in range(3):
        fluorescent_img[:, :, c] = fluorescent_img[:, :, c] + bloom[:, :, c] * 0.4
    
    # 归一化并增强对比度
    max_val = np.max(fluorescent_img[:, :, :3])
    if max_val > 0:
        for c in range(3):
            channel = fluorescent_img[:, :, c]
            min_v = np.percentile(channel[channel > 0], 2) if np.any(channel > 0) else 0
            max_v = np.percentile(channel, 98)
            if max_v > min_v:
                channel = (channel - min_v) / (max_v - min_v)
            fluorescent_img[:, :, c] = channel
    
    # 转换为8位图像
    fluorescent_img[:, :, :3] = np.clip(fluorescent_img[:, :, :3] * 255, 0, 255)
    fluorescent_img[:, :, 3] = np.clip(fluorescent_img[:, :, 3], 0, 255)
    fluorescent_img = fluorescent_img.astype(np.uint8)
    
    return Image.fromarray(fluorescent_img, 'RGBA')


def predict_spatial_distribution(img, patch_size=224, stride=112):
    """
    将图像分成多个patch，对每个patch预测表达值，重建空间分布图
    
    Args:
        img: PIL图像
        patch_size: patch大小 (默认224)
        stride: 滑动窗口步长 (默认112，即50%重叠)
    
    Returns:
        spatial_maps: 字典 {marker_name: 2D numpy array}
        global_predictions: 全局平均预测值
    """
    global model, device
    
    width, height = img.size
    
    # 计算patch网格
    if width <= patch_size:
        patches_x = 1
    else:
        patches_x = (width - patch_size + stride - 1) // stride + 1

    if height <= patch_size:
        patches_y = 1
    else:
        patches_y = (height - patch_size + stride - 1) // stride + 1
    
    # 存储每个patch的预测和坐标
    all_predictions = []
    all_coords = []
    
    # 预处理
    transform = get_transform()
    
    # 对每个patch进行预测
    for py in range(patches_y):
        for px in range(patches_x):
            # 计算patch位置
            x = min(px * stride, width - patch_size)
            y = min(py * stride, height - patch_size)
            
            # 如果图像太小，调整
            actual_patch_w = min(patch_size, width - x)
            actual_patch_h = min(patch_size, height - y)
            
            # 裁剪patch
            patch = img.crop((x, y, x + actual_patch_w, y + actual_patch_h))
            
            # 如果patch太小，填充到patch_size
            if actual_patch_w < patch_size or actual_patch_h < patch_size:
                new_patch = Image.new('RGB', (patch_size, patch_size), (0, 0, 0))
                new_patch.paste(patch, (0, 0))
                patch = new_patch
            
            # 预处理并预测
            patch_tensor = transform(patch).unsqueeze(0)
            
            with torch.no_grad():
                features = model(
                    image=patch_tensor.to(device),
                    with_head=False,
                    out_norm=False,
                    ms_aug=True,
                    return_global=True
                )[0]
            
            predictions = generate_virtual_proteomics(features.cpu().numpy())
            all_predictions.append(predictions)
            all_coords.append((x, y, actual_patch_w, actual_patch_h))
    
    # 重建空间分布图
    # 使用加权平均处理重叠区域
    spatial_maps = {name: np.zeros((height, width), dtype=np.float32) 
                    for name in BIOMARKER_NAMES.values()}
    weight_map = np.zeros((height, width), dtype=np.float32)
    
    for pred, (x, y, pw, ph) in zip(all_predictions, all_coords):
        # 创建权重mask（中心权重高，边缘权重低）
        weight = np.ones((ph, pw), dtype=np.float32)
        
        # 应用高斯权重
        center_y, center_x = ph // 2, pw // 2
        for i in range(ph):
            for j in range(pw):
                dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
                weight[i, j] = np.exp(-dist**2 / (2 * (min(ph, pw) // 3)**2))
        
        # 累加预测值
        for name, value in pred.items():
            spatial_maps[name][y:y+ph, x:x+pw] += value * weight
        
        weight_map[y:y+ph, x:x+pw] += weight
    
    # 归一化
    weight_map = np.maximum(weight_map, 1e-8)  # 避免除零
    for name in spatial_maps:
        spatial_maps[name] = spatial_maps[name] / weight_map
    
    # 计算全局平均预测值
    global_predictions = {}
    for name in spatial_maps:
        global_predictions[name] = float(np.mean(spatial_maps[name]))
    
    return spatial_maps, global_predictions


def _is_background_patch(pil_img, white_thresh: float) -> bool:
    if white_thresh >= 1.0:
        return False
    arr_u8 = np.asarray(pil_img, dtype=np.uint8)
    if arr_u8.ndim != 3 or arr_u8.shape[2] != 3:
        return False
    hsv = cv2.cvtColor(arr_u8, cv2.COLOR_RGB2HSV)
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    v = hsv[:, :, 2].astype(np.float32) / 255.0
    return (float(v.mean()) >= white_thresh) and (float(s.mean()) <= 0.05)


def predict_spatial_distribution_hex(
    img,
    patch_size=224,
    stride=112,
    selected_markers=None,
    white_thresh=0.95,
    clip_01=True,
):
    global model, device
    width, height = img.size

    if selected_markers is None:
        selected_markers = [BIOMARKER_NAMES[i] for i in range(1, 11)]

    marker_to_idx = {BIOMARKER_NAMES[i]: i - 1 for i in range(1, 41)}

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

    transform = get_transform()

    yy, xx = np.mgrid[0:patch_size, 0:patch_size]
    cy = (patch_size - 1) / 2.0
    cx = (patch_size - 1) / 2.0
    sigma = max(1.0, patch_size / 3.0)
    weight_full = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma**2)).astype(np.float32)

    spatial_maps = {m: np.zeros((height, width), dtype=np.float32) for m in selected_markers}
    weight_map = np.zeros((height, width), dtype=np.float32)

    for y in ys:
        for x in xs:

            pw = min(patch_size, width - x)
            ph = min(patch_size, height - y)

            patch = img.crop((x, y, x + pw, y + ph))
            if _is_background_patch(patch, white_thresh):
                continue

            if pw < patch_size or ph < patch_size:
                new_patch = Image.new("RGB", (patch_size, patch_size), (255, 255, 255))
                new_patch.paste(patch, (0, 0))
                patch = new_patch

            patch_tensor = transform(patch).unsqueeze(0)

            with torch.no_grad():
                out = model(patch_tensor.to(device))

            if isinstance(out, (tuple, list)):
                preds = out[0]
            else:
                preds = out

            preds = preds[0].detach().cpu().numpy().astype(np.float32, copy=False)
            if clip_01:
                preds = np.clip(preds, 0.0, 1.0)

            w = weight_full[:ph, :pw]
            for m in selected_markers:
                idx = marker_to_idx.get(m, None)
                if idx is None:
                    continue
                spatial_maps[m][y : y + ph, x : x + pw] += preds[idx] * w
            weight_map[y : y + ph, x : x + pw] += w

    weight_map = np.maximum(weight_map, 1e-8)
    for m in spatial_maps:
        spatial_maps[m] = spatial_maps[m] / weight_map

    global_predictions = {m: float(np.mean(spatial_maps[m])) for m in spatial_maps}
    return spatial_maps, global_predictions


def render_single_marker_fluorescent(spatial_map, marker, tissue_mask=None, threshold_percentile: float = 80.0):
    """
    渲染单个标志物的荧光图像 - 高对比度CODEX风格
    
    特点：
    - 纯黑背景
    - 高亮度的稀疏信号点
    - 真实的发光效果
    """
    import scipy.ndimage as ndimage
    
    spatial_dist = np.clip(spatial_map.astype(np.float32, copy=False), 0, 1)
    if tissue_mask is not None:
        spatial_dist = spatial_dist * tissue_mask.astype(np.float32, copy=False)

    # 更激进的阈值处理 - 只保留高表达区域
    mask_vals = spatial_dist[tissue_mask] if tissue_mask is not None else spatial_dist[spatial_dist > 0]
    if mask_vals.size:
        p = float(np.clip(threshold_percentile, 0.0, 99.9))
        t = float(np.percentile(mask_vals, p))
        # 使用软阈值过渡
        spatial_dist = np.clip((spatial_dist - t) / (1 - t + 1e-8), 0, 1)

    # 激进的gamma校正 - 增强亮点、压暗低值
    spatial_dist = np.power(spatial_dist, 0.4)  # 更激进的gamma
    
    # 高斯模糊模拟荧光扩散 - 更小的sigma保持信号点清晰
    spatial_dist = ndimage.gaussian_filter(spatial_dist, sigma=0.8)
    
    # 二次gamma增强对比度
    spatial_dist = np.power(spatial_dist, 0.7)
    
    if tissue_mask is not None:
        spatial_dist = spatial_dist * tissue_mask.astype(np.float32, copy=False)

    # 获取颜色
    color = FLUORESCENT_COLORS.get(marker, (255, 255, 255))
    r, g, b = color[2], color[1], color[0]
    
    # 高亮度因子 - 模拟真实荧光的明亮度
    brightness = 2.5

    # 创建RGB图像
    rgb = np.zeros((spatial_dist.shape[0], spatial_dist.shape[1], 3), dtype=np.float32)
    rgb[:, :, 0] = spatial_dist * r * brightness / 255.0
    rgb[:, :, 1] = spatial_dist * g * brightness / 255.0
    rgb[:, :, 2] = spatial_dist * b * brightness / 255.0

    # 发光效果 (Bloom) - 小范围强发光
    bloom_small = np.zeros_like(rgb)
    for c in range(3):
        bloom_small[:, :, c] = ndimage.gaussian_filter(rgb[:, :, c], sigma=2)
    
    # 大范围柔光
    bloom_large = np.zeros_like(rgb)
    for c in range(3):
        bloom_large[:, :, c] = ndimage.gaussian_filter(rgb[:, :, c], sigma=8)
    
    # 混合发光效果
    rgb = rgb + bloom_small * 0.5 + bloom_large * 0.15

    # 归一化 - 确保黑背景和高亮前景
    max_val = float(np.max(rgb))
    if max_val > 0:
        # 非线性映射 - 增强对比度
        for c in range(3):
            channel = rgb[:, :, c]
            # 使用gamma映射增强
            channel = np.power(channel / max_val, 0.8)
            rgb[:, :, c] = np.clip(channel, 0, 1)

    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb, "RGB")


def generate_spatial_fluorescent(img, spatial_maps, selected_markers=None, alpha=0.6, tissue_mask=None, threshold_percentile: float = 80.0):
    """
    基于空间分布图生成真实感荧光叠加图像 - 高对比度CODEX风格
    
    模拟真实CODEX多通道免疫荧光成像效果：
    - 纯黑色背景（无信号区域）
    - 高对比度明亮荧光信号
    - 稀疏的亮点分布
    - 真实的发光效果
    """
    import scipy.ndimage as ndimage
    
    width, height = img.size
    
    if selected_markers is None:
        selected_markers = list(spatial_maps.keys())[:10]

    if tissue_mask is None:
        tissue_mask = _compute_tissue_mask(img, white_thresh=0.92)
    
    # 创建RGB荧光图像（黑色背景）
    fluorescent_rgb = np.zeros((height, width, 3), dtype=np.float32)
    
    # 为每个标志物创建独立的荧光通道并累加
    for marker in selected_markers:
        if marker not in spatial_maps:
            continue
        
        spatial_dist = spatial_maps[marker].astype(np.float32, copy=False)
        spatial_dist = np.clip(spatial_dist, 0, 1)
        if tissue_mask is not None:
            spatial_dist = spatial_dist * tissue_mask.astype(np.float32, copy=False)

        # 激进的阈值处理 - 只保留高表达区域
        mask_vals = spatial_dist[tissue_mask] if tissue_mask is not None else spatial_dist[spatial_dist > 0]
        if mask_vals.size:
            t = float(np.percentile(mask_vals, threshold_percentile))
            spatial_dist = np.clip((spatial_dist - t) / (1 - t + 1e-8), 0, 1)
        
        # 激进的gamma校正 - 增强亮点、压暗低值
        spatial_dist = np.power(spatial_dist, 0.4)
        
        # 小范围模糊保持信号点清晰
        spatial_dist = ndimage.gaussian_filter(spatial_dist, sigma=0.8)
        
        # 二次增强
        spatial_dist = np.power(spatial_dist, 0.7)
        
        if tissue_mask is not None:
            spatial_dist = spatial_dist * tissue_mask.astype(np.float32, copy=False)
        
        # 获取该标志物的荧光颜色
        color = FLUORESCENT_COLORS.get(marker, (255, 255, 255))
        r, g, b = color[2], color[1], color[0]  # 转换为RGB
        
        # 高亮度因子
        brightness = 2.5
        
        # 累加到荧光图像（加法混合模式 - 模拟真实荧光叠加）
        fluorescent_rgb[:, :, 0] += spatial_dist * r * brightness / 255.0
        fluorescent_rgb[:, :, 1] += spatial_dist * g * brightness / 255.0
        fluorescent_rgb[:, :, 2] += spatial_dist * b * brightness / 255.0
    
    # 发光效果 (Bloom)
    # 小范围强发光
    bloom_small = np.zeros_like(fluorescent_rgb)
    for c in range(3):
        bloom_small[:, :, c] = ndimage.gaussian_filter(fluorescent_rgb[:, :, c], sigma=2)
    
    # 大范围柔光
    bloom_large = np.zeros_like(fluorescent_rgb)
    for c in range(3):
        bloom_large[:, :, c] = ndimage.gaussian_filter(fluorescent_rgb[:, :, c], sigma=8)
    
    # 混合发光效果
    fluorescent_rgb = fluorescent_rgb + bloom_small * 0.5 + bloom_large * 0.15

    # 应用组织mask
    if tissue_mask is not None:
        m = tissue_mask.astype(np.float32, copy=False)
        for c in range(3):
            fluorescent_rgb[:, :, c] *= m

    # 归一化 - 确保黑背景和高亮前景
    max_val = float(np.max(fluorescent_rgb))
    if max_val > 0:
        for c in range(3):
            channel = fluorescent_rgb[:, :, c]
            # 使用gamma映射增强对比度
            channel = np.power(channel / max_val, 0.8)
            fluorescent_rgb[:, :, c] = np.clip(channel, 0, 1)
    
    # 转换为8位图像
    fluorescent_rgb = np.clip(fluorescent_rgb * 255, 0, 255).astype(np.uint8)
    fluorescent_only = Image.fromarray(fluorescent_rgb, 'RGB')

    # 创建叠加图像 - 使用屏幕混合模式
    he_array = np.array(img).astype(np.float32) / 255.0
    
    # 将H&E转换为灰度作为背景
    he_gray = np.mean(he_array, axis=2, keepdims=True)
    he_tinted = np.concatenate([he_gray * 0.9, he_gray * 0.85, he_gray * 0.95], axis=2)
    
    # 叠加荧光
    fl_normalized = fluorescent_rgb.astype(np.float32) / 255.0
    
    # 使用屏幕混合模式（Screen blend mode）- 更适合发光效果
    overlay = 1 - (1 - he_tinted) * (1 - fl_normalized * alpha)
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)
    
    overlay_img = Image.fromarray(overlay, 'RGB')
    
    return overlay_img, fluorescent_only


def generate_spatial_heatmap(img_size, predictions, selected_markers=None):
    """
    生成空间热力图（模拟CODEX多通道图像）
    
    Returns:
        热力图图像和颜色图例
    """
    width, height = img_size
    
    # 创建多通道图像
    heatmap = np.zeros((height, width, 3), dtype=np.float32)
    
    if selected_markers is None:
        selected_markers = list(predictions.keys())[:10]  # 默认显示前10个
    
    # 为每个标记物分配一个空间区域
    np.random.seed(42)
    
    # 生成平滑的空间分布
    for i, marker in enumerate(selected_markers):
        if marker not in predictions:
            continue
        
        value = predictions[marker]
        if value < 0.1:
            continue
        
        # 使用柏林噪声生成平滑分布
        scale = 0.05 + 0.02 * i
        noise = np.random.randn(height // 10, width // 10)
        # 上采样
        from scipy.ndimage import zoom
        try:
            spatial_dist = zoom(noise, (height / (height // 10), width / (width // 10)), order=1)
        except:
            spatial_dist = cv2.resize(noise, (width, height))
        
        spatial_dist = (spatial_dist - spatial_dist.min()) / (spatial_dist.max() - spatial_dist.min() + 1e-8)
        spatial_dist = spatial_dist * value
        
        # 获取颜色并添加到热力图
        color = FLUORESCENT_COLORS.get(marker, (255, 255, 255))
        heatmap[:, :, 0] += spatial_dist * color[2]
        heatmap[:, :, 1] += spatial_dist * color[1]
        heatmap[:, :, 2] += spatial_dist * color[0]
    
    # 归一化
    heatmap = np.clip(heatmap / heatmap.max() * 255, 0, 255).astype(np.uint8)
    
    return Image.fromarray(heatmap, 'RGB')


def overlay_fluorescent_on_he(he_img, fluorescent_layer, alpha=0.5):
    """
    将荧光层叠加到H&E图像上
    
    使用屏幕混合模式（Screen Blend）实现更真实的发光效果
    
    Args:
        he_img: H&E图像 (PIL Image, RGB)
        fluorescent_layer: 荧光层 (PIL Image, RGBA)
        alpha: 叠加透明度
    
    Returns:
        叠加后的图像
    """
    # 确保尺寸匹配
    if he_img.size != fluorescent_layer.size:
        fluorescent_layer = fluorescent_layer.resize(he_img.size, Image.LANCZOS)
    
    # 转换为numpy数组
    he_array = np.array(he_img).astype(np.float32) / 255.0
    fl_array = np.array(fluorescent_layer).astype(np.float32) / 255.0
    
    # 提取荧光层的RGB和Alpha通道
    fl_rgb = fl_array[:, :, :3]
    fl_alpha = fl_array[:, :, 3:4] / 255.0 * alpha
    
    # 将H&E转为灰度作为底色
    he_gray = np.mean(he_array, axis=2, keepdims=True)
    he_tinted = np.concatenate([he_gray * 0.95, he_gray * 0.9, he_gray * 0.98], axis=2)
    
    # 屏幕混合模式: result = 1 - (1 - a) * (1 - b)
    # 更适合发光效果的叠加
    fl_with_alpha = fl_rgb * fl_alpha
    overlay = 1 - (1 - he_tinted) * (1 - fl_with_alpha)
    
    # 混合原始H&E
    result = he_array * (1 - alpha * 0.5) + overlay * alpha * 0.5 + fl_with_alpha * 0.3
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    
    return Image.fromarray(result, 'RGB')


@app.route('/generate_fluorescent', methods=['POST'])
def generate_fluorescent():
    """
    生成荧光层叠加图像（基于空间分布预测）
    """
    try:
        data = request.json
        image_path = data.get('image_path')
        selected_markers = data.get('selected_markers')
        alpha = data.get('alpha', 0.6)
        threshold_percentile = float(data.get('sparsity_percentile', 80))
        mode = data.get('mode', 'spatial')
        output_dir = data.get('output_dir', None)
        job_id = data.get('job_id', None)
        transparent_alpha_floor = float(data.get('transparent_alpha_floor', 0.0))
        psf_sigma = float(data.get('psf_sigma', 0.0))
        poisson_scale = float(data.get('poisson_scale', 0.0))
        background_noise_sigma = float(data.get('background_noise_sigma', 0.0))
        noise_seed = data.get('noise_seed', None)
        channel_norm = data.get('channel_norm', 'none')
        channel_ref_stats = data.get('channel_ref_stats', None)
        z_clip = float(data.get('z_clip', 3.0))
        segmentation_mode = str(data.get('segmentation_mode', 'none')).strip().lower()
        seg_dilate = int(data.get('seg_dilate', 6))
        seg_min_size = int(data.get('seg_min_size', 64))

        if mode == "hex" and (output_dir is None or str(output_dir).strip() == ""):
            output_dir = "/home/acproject/workspace/python_projects/HEX/bridge_out_web"
        
        saved_files = None

        # 加载图像
        img = Image.open(image_path).convert('RGB')
        original_size = img.size
        
        max_display_size = 800
        if max(original_size) > max_display_size:
            ratio = max_display_size / max(original_size)
            display_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
            img_display = img.resize(display_size, Image.LANCZOS)
        else:
            img_display = img
            display_size = original_size

        max_infer_size = 4096
        if max(original_size) > max_infer_size:
            ratio = max_infer_size / max(original_size)
            infer_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
            img_infer = img.resize(infer_size, Image.LANCZOS)
        else:
            img_infer = img
            infer_size = original_size
        
        load_models()
        if not selected_markers:
            selected_markers = [BIOMARKER_NAMES[i] for i in range(1, 11)]
        
        if mode == 'spatial':
            print(f"正在进行空间分布预测 (推理尺寸: {infer_size}, 显示尺寸: {display_size})...")
            spatial_maps, predictions = predict_spatial_distribution_hex(
                img_infer,
                patch_size=192,
                stride=48,
                selected_markers=selected_markers,
                white_thresh=0.95,
                clip_01=True,
            )
            print(f"空间预测完成，共 {len(spatial_maps)} 个标志物")
            
            # 生成荧光叠加
            tissue_mask = _compute_tissue_mask(img_infer, white_thresh=0.92)
            cell_mask = None
            if segmentation_mode == "weak":
                cell_mask = _compute_cell_mask_weak(
                    img_infer,
                    tissue_mask=tissue_mask,
                    dilate_radius=seg_dilate,
                    min_size=seg_min_size,
                )
            combined_mask = cell_mask if cell_mask is not None else tissue_mask
            if channel_norm == "zscore":
                spatial_maps = _normalize_spatial_maps_zscore(
                    spatial_maps,
                    tissue_mask=combined_mask,
                    ref_stats=channel_ref_stats,
                    z_clip=z_clip,
                )
            if combined_mask is not None and np.any(combined_mask):
                predictions = {m: float(np.mean(spatial_maps[m][combined_mask])) for m in spatial_maps}
            result, fluorescent_only = generate_spatial_fluorescent(
                img_infer,
                spatial_maps, 
                selected_markers, 
                alpha,
                tissue_mask=combined_mask,
                threshold_percentile=threshold_percentile,
            )
            per_marker_images_pil = {m: render_single_marker_fluorescent(spatial_maps[m], m, tissue_mask=combined_mask, threshold_percentile=threshold_percentile) for m in selected_markers if m in spatial_maps}

            if result.size != display_size:
                result = result.resize(display_size, Image.LANCZOS)
                fluorescent_only = fluorescent_only.resize(display_size, Image.LANCZOS)
                per_marker_images_pil = {m: im.resize(display_size, Image.LANCZOS) for m, im in per_marker_images_pil.items()}

            tissue_mask = _compute_tissue_mask(img_display, white_thresh=0.92)
            cell_mask = None
            if segmentation_mode == "weak":
                cell_mask = _compute_cell_mask_weak(
                    img_display,
                    tissue_mask=tissue_mask,
                    dilate_radius=seg_dilate,
                    min_size=seg_min_size,
                )
            combined_mask = cell_mask if cell_mask is not None else tissue_mask

            fluorescent_only = _apply_psf_and_noise_rgb(
                fluorescent_only,
                psf_sigma=psf_sigma,
                poisson_scale=poisson_scale,
                background_noise_sigma=background_noise_sigma,
                seed=int(noise_seed) if noise_seed is not None else None,
            )
            per_marker_images_pil = {
                m: _apply_psf_and_noise_rgb(
                    im,
                    psf_sigma=psf_sigma,
                    poisson_scale=poisson_scale,
                    background_noise_sigma=background_noise_sigma,
                    seed=int(noise_seed) if noise_seed is not None else None,
                )
                for m, im in per_marker_images_pil.items()
            }

            fluorescent_only_rgba = _rgb_black_to_transparent_rgba(
                fluorescent_only,
                alpha_floor=transparent_alpha_floor,
                alpha_gamma=1.0,
                alpha_mask=combined_mask,
            )
            per_marker_images_pil = {
                m: _rgb_black_to_transparent_rgba(im, alpha_floor=transparent_alpha_floor, alpha_gamma=1.0, alpha_mask=combined_mask)
                for m, im in per_marker_images_pil.items()
            }

            fl_only_base64 = image_to_base64(fluorescent_only_rgba)
            per_marker_images = {m: image_to_base64(im) for m, im in per_marker_images_pil.items()}
        elif mode == "hex":
            import h5py  # type: ignore
            import predict_he_to_codex_h5 as hex_h5  # type: ignore

            if output_dir:
                out_root = Path(output_dir).expanduser().resolve()
                project_root = Path("/home/acproject/workspace/python_projects/HEX").resolve()
                if project_root not in out_root.parents and out_root != project_root:
                    raise ValueError("output_dir must be inside /home/acproject/workspace/python_projects/HEX")
                out_root.mkdir(parents=True, exist_ok=True)
                run_dir = out_root
                managed_tmp = None
            else:
                managed_tmp = tempfile.TemporaryDirectory(prefix="hex_fluor_")
                run_dir = Path(managed_tmp.name)

            try:
                name = str(job_id).strip() if job_id else Path(image_path).stem
                if not name:
                    name = uuid.uuid4().hex

                pred_h5_dir = run_dir / "he2codex" / "pred_h5"
                codex_npy_dir = run_dir / "he2codex" / "codex_npy"
                png_dir = run_dir / "he2codex" / "fluorescent_png"
                pred_h5_dir.mkdir(parents=True, exist_ok=True)
                codex_npy_dir.mkdir(parents=True, exist_ok=True)
                png_dir.mkdir(parents=True, exist_ok=True)

                tmp_h5 = pred_h5_dir / f"{name}.h5"
                tmp_npy = codex_npy_dir / f"{name}.npy"

                hex_h5.predict_to_h5_from_pil(
                    img=img_infer,
                    output_h5=tmp_h5,
                    model=model,
                    device=device,
                    patch_size=224,
                    stride=112,
                    batch_size=64,
                    white_thresh=0.92,
                    clip_01=True,
                    max_patches=None,
                    export_png_dir=png_dir,
                    export_markers=selected_markers,
                    export_alpha=float(alpha),
                    export_threshold_percentile=float(threshold_percentile),
                )

                stem = tmp_h5.stem.replace("_pred", "")
                overlay_path = png_dir / f"{stem}_fluorescent_overlay.png"
                fl_only_path = png_dir / f"{stem}_fluorescent_only.png"
                marker_dir = png_dir / f"{stem}_markers"

                result = Image.open(str(overlay_path)).convert("RGB")
                fluorescent_only = Image.open(str(fl_only_path)).convert("RGB")

                per_marker_images_pil = {}
                for m in selected_markers:
                    p = marker_dir / f"{m}.png"
                    if p.exists():
                        per_marker_images_pil[m] = Image.open(str(p)).convert("RGB")

                hex_h5.h5_to_grid_npy(tmp_h5, tmp_npy)
                saved_files = {
                    "pred_h5": str(tmp_h5),
                    "codex_npy": str(tmp_npy),
                    "fluorescent_overlay_png": str(overlay_path),
                    "fluorescent_only_png": str(fl_only_path),
                    "marker_dir": str(marker_dir),
                }

                with h5py.File(str(tmp_h5), "r") as f:
                    pred = f["codex_prediction"][:]
                pred = pred.astype(np.float32, copy=False)
                pred = np.clip(pred, 0.0, 1.0)
                marker_to_idx = {BIOMARKER_NAMES[i]: i - 1 for i in range(1, 41)}
                predictions = {}
                for m in selected_markers:
                    idx = marker_to_idx.get(m, None)
                    if idx is None:
                        continue
                    predictions[m] = float(np.mean(pred[:, idx])) if pred.shape[0] else 0.0
            finally:
                if managed_tmp is not None:
                    managed_tmp.cleanup()

            if result.size != display_size:
                result = result.resize(display_size, Image.LANCZOS)
                fluorescent_only = fluorescent_only.resize(display_size, Image.LANCZOS)
                per_marker_images_pil = {m: im.resize(display_size, Image.LANCZOS) for m, im in per_marker_images_pil.items()}

            fluorescent_only = _apply_psf_and_noise_rgb(
                fluorescent_only,
                psf_sigma=psf_sigma,
                poisson_scale=poisson_scale,
                background_noise_sigma=background_noise_sigma,
                seed=int(noise_seed) if noise_seed is not None else None,
            )
            per_marker_images_pil = {
                m: _apply_psf_and_noise_rgb(
                    im,
                    psf_sigma=psf_sigma,
                    poisson_scale=poisson_scale,
                    background_noise_sigma=background_noise_sigma,
                    seed=int(noise_seed) if noise_seed is not None else None,
                )
                for m, im in per_marker_images_pil.items()
            }

            fluorescent_only_rgba = _rgb_black_to_transparent_rgba(
                fluorescent_only,
                alpha_floor=transparent_alpha_floor,
                alpha_gamma=1.0,
                alpha_mask=combined_mask,
            )
            per_marker_images_pil = {
                m: _rgb_black_to_transparent_rgba(im, alpha_floor=transparent_alpha_floor, alpha_gamma=1.0, alpha_mask=combined_mask)
                for m, im in per_marker_images_pil.items()
            }

            fl_only_base64 = image_to_base64(fluorescent_only_rgba)
            per_marker_images = {m: image_to_base64(im) for m, im in per_marker_images_pil.items()}
        else:
            # 旧方法：随机分布（更快但无空间信息）
            transform = get_transform()
            img_tensor = transform(img_display).unsqueeze(0)
            features = extract_features(img_tensor)
            predictions = generate_virtual_proteomics(features)
            
            fluorescent = generate_fluorescent_layer(display_size, predictions, selected_markers)
            result = overlay_fluorescent_on_he(img_display, fluorescent, alpha)
            fl_only_base64 = image_to_base64(fluorescent.convert('RGB'))
            per_marker_images = {}
        
        # 转换为base64
        result_base64 = image_to_base64(result)
        original_resized_base64 = image_to_base64(img_display)
        
        return jsonify({
            'success': True,
            'overlay_image': result_base64,
            'fluorescent_only': fl_only_base64,
            'predictions': predictions,
            'per_marker_images': per_marker_images,
            'original_resized': original_resized_base64,
            'image_size': display_size,
            'mode': mode,
            'saved_files': saved_files,
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/get_marker_colors', methods=['GET'])
def get_marker_colors():
    """获取所有标记物的颜色信息"""
    colors_info = []
    for marker, color in FLUORESCENT_COLORS.items():
        # 转换为hex颜色
        hex_color = '#{:02x}{:02x}{:02x}'.format(color[2], color[1], color[0])
        colors_info.append({
            'name': marker,
            'color': hex_color,
            'rgb': list(color)
        })
    return jsonify(colors_info)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--hex_ckpt", default="/home/acproject/workspace/python_projects/HEX/hex/checkpoint.pth")
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--stride", type=int, default=112)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--white_thresh", type=float, default=0.95)
    parser.add_argument("--max_size", type=int, default=0)
    parser.add_argument("--markers", default="")
    args = parser.parse_args()

    if args.offline:
        if args.input is None or args.output_dir is None:
            raise SystemExit("--offline requires --input and --output_dir")

        input_path = Path(args.input)
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        selected_markers = None
        if args.markers.strip():
            selected_markers = [m.strip() for m in args.markers.split(",") if m.strip()]

        load_models(hex_ckpt_path=args.hex_ckpt)

        def run_one(img_path: Path):
            img = Image.open(str(img_path)).convert("RGB")
            if args.max_size and max(img.size) > args.max_size:
                ratio = args.max_size / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.LANCZOS)

            spatial_maps, predictions = predict_spatial_distribution_hex(
                img,
                patch_size=args.patch_size,
                stride=args.stride,
                selected_markers=selected_markers,
                white_thresh=args.white_thresh,
                clip_01=True,
            )

            overlay, fl_only = generate_spatial_fluorescent(
                img,
                spatial_maps,
                selected_markers=selected_markers,
                alpha=args.alpha,
            )

            stem = img_path.stem
            overlay_path = out_dir / f"{stem}_overlay.png"
            fl_path = out_dir / f"{stem}_fluorescent.png"
            pred_path = out_dir / f"{stem}_predictions.json"

            overlay.save(str(overlay_path))
            fl_only.save(str(fl_path))
            pred_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")

            print(f"[done] {img_path.name} -> {overlay_path.name}, {fl_path.name}, {pred_path.name}")

        if input_path.is_dir():
            img_files = []
            for ext in ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"):
                img_files.extend(sorted(input_path.glob(ext)))
            if not img_files:
                raise SystemExit(f"No images found in {input_path}")
            for p in img_files:
                run_one(p)
        else:
            run_one(input_path)
    else:
        print("=" * 60)
        print("HEX/MUSK 可视化Web应用")
        print("=" * 60)
        print()
        print("预加载模型...")
        load_models()
        print()
        print("启动Web服务器...")
        print("请在浏览器中访问: http://localhost:5000")
        print()
        app.run(host='0.0.0.0', port=5000, debug=True)
