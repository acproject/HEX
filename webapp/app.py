"""
HEX/MUSK 可视化Web应用
用于展示MUSK模型处理肺癌类器官H&E图像的过程
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'MUSK'))

import torch
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_from_directory
from pathlib import Path
import base64
from io import BytesIO
import json
import cv2
import colorsys

# MUSK相关导入
from musk import utils, modeling
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
import torchvision.transforms as transforms

app = Flask(__name__)

# 全局变量存储模型
model = None
device = None

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


def load_models():
    """加载MUSK和HEX模型"""
    global model, device
    
    if model is not None:
        return
    
    print("正在加载MUSK模型...")
    
    # 检测设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        if cap[0] < 7.5:  # V100 is CC 7.0
            print(f"GPU CC {cap[0]}.{cap[1]} 不兼容，使用CPU模式")
            device = 'cpu'
    
    print(f"使用设备: {device}")
    
    # 加载MUSK模型
    from musk.modeling import _get_large_config, MUSK
    args = _get_large_config(img_size=384, vocab_size=64010)
    model = MUSK(args)
    
    # 本地模型路径
    musk_path = "/home/acproject/workspace/python_projects/HEX/models/musk/model.safetensors"
    utils.load_model_and_may_interpolate(musk_path, model, 'model|module', '')
    
    model = model.to(device)
    model.eval()
    
    print("MUSK模型加载完成!")


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


def generate_virtual_proteomics(features):
    """
    基于特征生成虚拟蛋白质组学预测
    这里使用简化的线性映射作为演示
    实际应用中需要加载训练好的HEX模型
    """
    # 模拟40个生物标志物的表达值（0-1范围）
    # 实际应用中应该使用训练好的回归头
    np.random.seed(42)  # 为了可重复性
    
    # 基于特征生成伪表达值
    feature_mean = np.mean(features)
    feature_std = np.std(features)
    
    # 生成模拟的生物标志物表达
    predictions = {}
    for i, name in BIOMARKER_NAMES.items():
        # 使用特征统计量生成伪表达值
        value = np.clip(0.5 + 0.3 * np.sin(i * 0.5) + 0.1 * (feature_mean - 0.5), 0, 1)
        # 添加一些基于特征的变化
        value += 0.1 * np.tanh(feature_std * 2)
        value = np.clip(value, 0, 1)
        predictions[name] = float(value)
    
    return predictions


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
    
    return render_template('index.html', sample_images=sample_images)


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)


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
        
        # 提取特征
        features = extract_features(img_tensor)
        
        # 生成虚拟蛋白质组学预测
        predictions = generate_virtual_proteomics(features)
        
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
                predictions = generate_virtual_proteomics(features)
                
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
    patches_x = max(1, (width - patch_size) // stride + 1)
    patches_y = max(1, (height - patch_size) // stride + 1)
    
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


def generate_spatial_fluorescent(img, spatial_maps, selected_markers=None, alpha=0.6):
    """
    基于空间分布图生成真实感荧光叠加图像
    
    模拟真实CODEX多通道免疫荧光成像效果：
    - 黑色背景（无信号区域）
    - 高对比度明亮荧光
    - 多通道颜色混合
    - 发光/晕染效果
    
    Args:
        img: 原始H&E图像 (PIL Image)
        spatial_maps: 空间分布图字典 {marker_name: 2D array}
        selected_markers: 选中的标记物
        alpha: 叠加透明度
    
    Returns:
        overlay_img: 叠加后的图像
        fluorescent_only: 仅荧光层图像
    """
    import scipy.ndimage as ndimage
    
    width, height = img.size
    
    if selected_markers is None:
        selected_markers = list(spatial_maps.keys())[:10]
    
    # 创建RGBA荧光图像（黑色背景）
    fluorescent_rgba = np.zeros((height, width, 4), dtype=np.float32)
    
    # 为每个标志物创建独立的荧光通道
    for marker in selected_markers:
        if marker not in spatial_maps:
            continue
        
        spatial_dist = spatial_maps[marker].copy()
        
        # 获取该标志物的荧光颜色 (BGR -> RGB)
        color = FLUORESCENT_COLORS.get(marker, (255, 255, 255))
        r, g, b = color[2], color[1], color[0]  # 转换为RGB
        
        # 增强对比度 - 使用非线性映射
        # 真实荧光信号是指数响应的
        spatial_dist = np.clip(spatial_dist, 0, 1)
        spatial_dist = np.power(spatial_dist, 0.6)  # gamma校正增强低值
        spatial_dist = spatial_dist ** 0.8  # 进一步增强
        
        # 应用高斯模糊模拟荧光扩散效果
        spatial_dist = ndimage.gaussian_filter(spatial_dist, sigma=2)
        
        # 增强亮度
        brightness = 1.5
        
        # 累加到荧光图像（使用加法混合模式，模拟真实荧光叠加）
        fluorescent_rgba[:, :, 0] += spatial_dist * r * brightness / 255.0  # R
        fluorescent_rgba[:, :, 1] += spatial_dist * g * brightness / 255.0  # G
        fluorescent_rgba[:, :, 2] += spatial_dist * b * brightness / 255.0  # B
        fluorescent_rgba[:, :, 3] += spatial_dist * 255  # Alpha
    
    # 应用发光效果（bloom effect）
    # 创建一个模糊版本用于发光
    bloom = np.zeros_like(fluorescent_rgba[:, :, :3])
    for c in range(3):
        bloom[:, :, c] = ndimage.gaussian_filter(fluorescent_rgba[:, :, c], sigma=5)
    
    # 将发光效果加回原图
    for c in range(3):
        fluorescent_rgba[:, :, c] = fluorescent_rgba[:, :, c] + bloom[:, :, c] * 0.3
    
    # 归一化并增强对比度
    max_val = np.max(fluorescent_rgba[:, :, :3])
    if max_val > 0:
        # 使用自适应对比度增强
        for c in range(3):
            channel = fluorescent_rgba[:, :, c]
            # 裁剪并增强
            channel = np.clip(channel, 0, max_val)
            # 直方图拉伸
            min_v = np.percentile(channel[channel > 0], 5) if np.any(channel > 0) else 0
            max_v = np.percentile(channel, 99.5)
            if max_v > min_v:
                channel = (channel - min_v) / (max_v - min_v)
            fluorescent_rgba[:, :, c] = channel
    
    # 创建仅荧光图像（黑色背景）
    fluorescent_rgb = fluorescent_rgba[:, :, :3].copy()
    fluorescent_rgb = np.clip(fluorescent_rgb * 255, 0, 255).astype(np.uint8)
    
    # 应用alpha通道作为亮度mask
    alpha_mask = fluorescent_rgba[:, :, 3:4].copy()
    alpha_mask = np.clip(alpha_mask / alpha_mask.max() if alpha_mask.max() > 0 else alpha_mask, 0, 1)
    
    # 增强荧光区域的亮度
    for c in range(3):
        fluorescent_rgb[:, :, c] = (fluorescent_rgb[:, :, c] * (0.3 + 0.7 * alpha_mask[:, :, 0])).astype(np.uint8)
    
    fluorescent_only = Image.fromarray(fluorescent_rgb, 'RGB')
    
    # 创建叠加图像
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
        mode = data.get('mode', 'spatial')  # 'spatial' or 'random'
        
        # 加载图像
        img = Image.open(image_path).convert('RGB')
        original_size = img.size
        
        # 缩小图像以加快处理速度
        max_size = 800
        if max(original_size) > max_size:
            ratio = max_size / max(original_size)
            new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
            img_resized = img.resize(new_size, Image.LANCZOS)
        else:
            img_resized = img
            new_size = original_size
        
        load_models()
        
        if mode == 'spatial':
            # 新方法：空间分布预测
            print(f"正在进行空间分布预测 (图像尺寸: {new_size})...")
            spatial_maps, predictions = predict_spatial_distribution(
                img_resized, 
                patch_size=224, 
                stride=112
            )
            print(f"空间预测完成，共 {len(spatial_maps)} 个标志物")
            
            # 生成荧光叠加
            result, fluorescent_only = generate_spatial_fluorescent(
                img_resized, 
                spatial_maps, 
                selected_markers, 
                alpha
            )
            fl_only_base64 = image_to_base64(fluorescent_only)
        else:
            # 旧方法：随机分布（更快但无空间信息）
            transform = get_transform()
            img_tensor = transform(img).unsqueeze(0)
            features = extract_features(img_tensor)
            predictions = generate_virtual_proteomics(features)
            
            fluorescent = generate_fluorescent_layer(new_size, predictions, selected_markers)
            result = overlay_fluorescent_on_he(img_resized, fluorescent, alpha)
            fl_only_base64 = image_to_base64(fluorescent.convert('RGB'))
        
        # 转换为base64
        result_base64 = image_to_base64(result)
        
        return jsonify({
            'success': True,
            'overlay_image': result_base64,
            'fluorescent_only': fl_only_base64,
            'predictions': predictions,
            'image_size': new_size,
            'mode': mode
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
    print("=" * 60)
    print("HEX/MUSK 可视化Web应用")
    print("=" * 60)
    print()
    
    # 预加载模型
    print("预加载模型...")
    load_models()
    print()
    
    print("启动Web服务器...")
    print("请在浏览器中访问: http://localhost:5000")
    print()
    
    app.run(host='0.0.0.0', port=5000, debug=True)
