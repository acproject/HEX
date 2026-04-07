# HEX
===========
## 基于人工智能的组织病理学虚拟空间蛋白质组学，用于肺癌可解释生物标志物发现

**摘要：** 空间蛋白质组学能够高分辨率地绘制蛋白质表达图谱，可以改变我们对生物学和疾病的理解。然而，临床转化仍面临重大挑战，包括成本、复杂性和可扩展性问题。在此，我们提出H&E到蛋白质表达（HEX），一种旨在从标准组织病理学切片计算生成空间蛋白质组学图谱的人工智能模型。HEX在来自382个肿瘤样本的819,000个配对组织病理学图像及其匹配蛋白质表达数据上进行训练和验证，能够准确预测涵盖免疫、结构和功能程序的40个生物标志物的表达。与从H&E图像预测蛋白质表达的替代方法相比，HEX展现出显著的性能提升。我们开发了一种多模态数据整合方法，将原始H&E图像与AI生成的虚拟空间蛋白质组学相结合，以增强结果预测。应用于总计2,298名患者的六个独立非小细胞肺癌队列，与传统临床病理学和分子生物标志物相比，HEX支持的多模态整合将预后准确率提高了22%，免疫治疗反应预测准确率提高了24-39%。生物学解释揭示了预测治疗反应的空间组织肿瘤-免疫微环境，包括应答者中辅助性T细胞和细胞毒性T细胞的共定位，以及非应答者中免疫抑制性肿瘤相关巨噬细胞和中性粒细胞聚集。HEX提供了一种低成本且可扩展的方法来研究空间生物学，并为精准医学的可解释生物标志物的发现和临床转化提供了可能。

## 依赖项：

**硬件：**
- NVIDIA GPU（在配备CUDA 11.8和cuDNN 9.1的NVIDIA L40S x8上测试）（Ubuntu 22.04）

**软件：**
- Python (3.10.15), PyTorch (2.4.0+cu118)

**其他Python库：**
- accelerate (1.2.0), captum (0.7.0), fsspec (2024.10.0), ftfy (6.3.1), gitpython (3.1.43), h5py (3.12.1), huggingface-hub (0.26.5), imageio (2.36.1), joblib (1.4.2), lifelines (0.30.0), lightning-utilities (0.11.9), lxml (5.3.0), matplotlib (3.9.3), musk (1.0.0), networkx (3.4.2), nltk (3.9.1), numpy (2.2.0), opencv-python (4.10.0.84), openslide-python (1.4.1), pandas (2.2.3), pillow (11.0.0), protobuf (5.29.1), pytorch-lightning (2.2.1), scikit-image (0.24.0), scikit-learn (1.5.2), scikit-survival (0.23.1), scipy (1.14.1), seaborn (0.13.2), tensorboardx (2.6.2.2), timm (0.9.8), torch-geometric (2.6.1), torchaudio (2.5.1), torchvision (0.20.1), tqdm (4.67.1), transformers (4.47.0), wandb (0.19.1)
* MUSK (https://github.com/lilab-stanford/MUSK)
* Palom (https://github.com/labsyspharm/palom)
* DINOv2 (https://github.com/facebookresearch/dinov2)
* CLAM (https://github.com/mahmoodlab/CLAM)
* imbalanced-regression (https://github.com/YyzHarry/imbalanced-regression)
* robust_loss_pytorch (https://github.com/jonbarron/robust_loss_pytorch)
* MCAT (https://github.com/mahmoodlab/MCAT)

### 补充说明上面库的作用
#### 视觉编码器
|库	|作用|
|---|--|
|MUSK|	核心视觉编码器，用于从H&E图像提取特征（hex_architecture.py）|
|DINOv2|	用于从CODEX图像提取特征向量（codex_h5_png2fea.py）|
|timm	|提供预训练视觉模型加载接口（create_model）|

#### 病理图像处理
|库	|作用|
|--|--|
|CLAM	  |WSI（全切片图像）预处理、切片分割、特征提取|
|OpenSlide  |读取和处理全切片病理图像（.svs格式）（virtual_codex_from_h5.py）|
|Palom	|CODEX与H&E图像配准对齐（extract_marker_info_patch.py）|

#### 多模态融合与可解释性
|库	|作用|
|--|--|
|MCAT	|多模态协同注意力Transformer，融合H&E和CODEX特征（model_coattn.py）|
|Captum	|计算积分梯度，实现模型可解释性（test_mica.py）|


#### 不平衡学习
|库	|作用|
|--|--|
|imbalanced-regression	|处理蛋白质表达值的标签分布不平衡问题（FDS特征分布平滑）|
|robust_loss_pytorch	|鲁棒损失函数，处理回归中的异常值|

#### 训练监控与实验管理
|库	|作用|
|--|--|
|wandb|	实验跟踪和可视化|

## 第一步：CODEX和H&E图像预处理
* 使用palom包对CODEX和H&E图像进行配准，获得配准后的CODEX图像。
* 运行 `extract_marker_info_patch.py` 提取每个图像块的蛋白质表达信息。
* 使用 `extract_he_patch.py` 脚本构建配对的组织病理学图像和匹配蛋白质表达数据集。
* 使用CLAM风格的分割工具（来自 `mahmoodlab/CLAM`）根据您的队列元数据为HEX和MICA创建患者级别的数据分割，然后运行 `check_splits.py` 检查分割完整性。

## 第二步：训练和测试HEX
* 使用 `torchrun --nnodes=1 --nproc-per-node=8 ./hex/train_dist_codex_lung_marker.py` 开始训练。
日志和检查点将分别保存到writer_dir和checkpoint_dir。
* 通过运行 `python test_codex_lung_marker.py` 评估模型检查点，使用checkpoint_path指定 `<save_location>/models/your_checkpoint.pth`。
输出结果将存储在 `save_dir` 中。示例数据位于 `hex/sample_data` 文件夹中。

## 第三步：训练和测试MICA
* 使用CLAM预处理WSI并生成组织学特征包（MCAT风格流程）。
* 应用训练好的HEX模型为每个WSI生成对应的CODEX图像，然后运行 `codex_h5_png2fea.py` 构建CODEX特征包（DINOv2）。
* 使用 `train_mica.py` 训练MICA，例如 `python train_mica.py --mode coattn --base_path your_path --gc 8 --project_name your_project --max_epochs 20 --lr 1e-5`。训练日志和检查点将保存在 `results_dir` 下。
示例数据位于 `mica/sample_data`。数据布局应遵循MCAT格式。
* 使用 `test_mica.py` 评估检查点（结果将保存为 `.pkl` 文件在相应的结果文件夹中）。
* 对于可解释性，`test_mica.py` 还可以计算积分梯度以可视化空间模式。


## 致谢
本项目基于许多开源仓库构建，如CLAM (https://github.com/mahmoodlab/CLAM), MCAT (https://github.com/mahmoodlab/MCAT), imbalanced-regression (https://github.com/YyzHarry/imbalanced-regression), 和Palom (https://github.com/labsyspharm/palom)。我们感谢这些仓库的作者和贡献者。

## 许可证
本仓库采用CC-BY-NC-ND 4.0许可证。本仓库包含/依赖按各自许可证授权的第三方组件。详情请参阅各项目。

## 引用
如果您觉得我们的工作对您的研究有帮助，请考虑引用：
* Li, Z., Li, Y., Xiang, J. et al. AI-enabled virtual spatial proteomics from histopathology for interpretable biomarker discovery in lung cancer. Nat Med 32, 231–244 (2026). https://doi.org/10.1038/s41591-025-04060-4
