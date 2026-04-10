from pathlib import Path

import torch
import torch.nn as nn
from musk import modeling, utils

MUSK_CKPT_PATH = str(Path(__file__).resolve().parents[1] / "models" / "musk" / "model.safetensors")

class CustomModel(nn.Module):
    def __init__(self, visual_output_dim, num_outputs, ckpt_path=None):
        super(CustomModel, self).__init__()
        
        # 使用MUSK类创建模型（不通过timm）
        config = modeling._get_large_config(img_size=384)
        model_musk = modeling.MUSK(config)
        
        # 加载MUSK权重
        load_path = ckpt_path or MUSK_CKPT_PATH
        print(f"Load MUSK ckpt from {load_path}")
        utils.load_model_and_may_interpolate(load_path, model_musk, 'model|module', '')
        
        self.visual = model_musk
        
        # HEX回归头：将MUSK特征映射到40个蛋白表达值
        self.regression_head = nn.Sequential(
            nn.Linear(visual_output_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )
        self.regression_head1 = nn.Sequential(
            nn.Linear(128, num_outputs),
        )

    def forward(self, x=None, image=None, with_head=False, out_norm=False, ms_aug=False, return_global=True):
        """
        兼容多种调用方式:
        1. model(x=tensor) -> 返回 (preds, features) 用于预测蛋白表达
        2. model(image=..., ms_aug=..., return_global=...) -> 返回MUSK特征用于空间预测
        """
        if x is None:
            return self.visual(
                image=image,
                with_head=with_head,
                out_norm=out_norm,
                ms_aug=ms_aug,
                return_global=return_global,
            )

        vision_cls = self.visual(
            image=x,
            with_head=False,
            out_norm=False,
            ms_aug=ms_aug,
            return_global=True,
        )[0]
        features = self.regression_head(vision_cls)
        preds = self.regression_head1(features)
        return preds, features
