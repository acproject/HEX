# 入门指南

<cite>
**本文档引用的文件**
- [README.md](file://README.md)
- [main.py](file://main.py)
- [hex/hex_architecture.py](file://hex/hex_architecture.py)
- [hex/utils.py](file://hex/utils.py)
- [hex/test_codex_lung_marker.py](file://hex/test_codex_lung_marker.py)
- [hex/train_dist_codex_lung_marker.py](file://hex/train_dist_codex_lung_marker.py)
- [mica/models/model_coattn.py](file://mica/models/model_coattn.py)
- [mica/train_mica.py](file://mica/train_mica.py)
- [mica/codex_h5_png2fea.py](file://mica/codex_h5_png2fea.py)
- [mica/test_mica.py](file://mica/test_mica.py)
- [extract_he_patch.py](file://extract_he_patch.py)
- [extract_marker_info_patch.py](file://extract_marker_info_patch.py)
- [check_splits.py](file://check_splits.py)
</cite>

## 目录
1. [简介](#简介)
2. [项目结构](#项目结构)
3. [核心组件](#核心组件)
4. [架构概览](#架构概览)
5. [详细组件分析](#详细组件分析)
6. [依赖关系分析](#依赖关系分析)
7. [性能考虑](#性能考虑)
8. [故障排除指南](#故障排除指南)
9. [结论](#结论)

## 简介

HEX是一个基于深度学习的虚拟空间蛋白组学项目，旨在从标准的组织病理学切片中计算生成蛋白质表达谱。该项目由斯坦福大学实验室开发，能够准确预测40种生物标志物的表达水平，包括免疫、结构和功能程序。

该项目的核心创新在于：
- 使用AI模型从H&E染色图像生成虚拟空间蛋白组学
- 提供可解释的生物标志物发现方法
- 支持多模态数据整合，结合原始H&E图像和AI生成的虚拟空间蛋白组学
- 在肺癌患者队列中验证了其在预后预测和免疫治疗反应预测方面的优越性能

## 项目结构

项目采用模块化设计，主要包含以下核心模块：

```mermaid
graph TB
subgraph "主项目"
A[HEX 主目录]
B[README.md]
C[main.py]
end
subgraph "HEX模块"
D[hex/]
D1[hex_architecture.py]
D2[utils.py]
D3[test_codex_lung_marker.py]
D4[train_dist_codex_lung_marker.py]
D5[sample_data/]
end
subgraph "MICA模块"
E[mica/]
E1[models/]
E11[model_coattn.py]
E2[train_mica.py]
E3[codex_h5_png2fea.py]
E4[test_mica.py]
E5[sample_data/]
end
subgraph "预处理工具"
F[extract_he_patch.py]
G[extract_marker_info_patch.py]
H[check_splits.py]
end
A --> D
A --> E
A --> F
A --> G
A --> H
D --> D1
D --> D2
D --> D3
D --> D4
E --> E1
E --> E2
E --> E3
E --> E4
```

**图表来源**
- [README.md:1-57](file://README.md#L1-L57)
- [hex/hex_architecture.py:1-46](file://hex/hex_architecture.py#L1-L46)
- [mica/models/model_coattn.py:1-714](file://mica/models/model_coattn.py#L1-L714)

**章节来源**
- [README.md:1-57](file://README.md#L1-L57)
- [main.py:1-7](file://main.py#L1-L7)

## 核心组件

### HEX主干架构

HEX的核心是基于MUSK（Multimodal Universal Scale Kit）的视觉编码器，专门用于处理多模态生物医学图像。

```mermaid
classDiagram
class CustomModel {
+int visual_output_dim
+int num_outputs
+FDS[] FDS
+bool training_status
+__init__(visual_output_dim, num_outputs, fds_active_markers)
+forward(x, labels, epoch)
}
class FDS {
+int feature_dim
+int bucket_num
+tensor running_mean
+tensor running_var
+smooth(features, labels, epoch)
+update_running_stats(features, labels, epoch)
}
class PatchDataset {
+list images
+ndarray labels
+__getitem__(idx)
+__len__()
}
CustomModel --> FDS : "使用"
CustomModel --> PatchDataset : "训练数据"
```

**图表来源**
- [hex/hex_architecture.py:15-46](file://hex/hex_architecture.py#L15-L46)
- [hex/utils.py:116-342](file://hex/utils.py#L116-L342)

### MICA多模态注意力模型

MICA模块实现了多模态注意力机制，用于整合H&E图像和虚拟CODEX图像特征。

```mermaid
classDiagram
class MCAT_Surv {
+string fusion
+int n_classes
+string transformer_mode
+string pooling
+Sequential wsi_net
+Sequential codex_net
+MultiheadAttention coattn
+TransformerEncoder path_transformer
+TransformerEncoder codex_transformer
+Attn_Net_Gated path_attention_head
+Sequential path_rho
+Attn_Net_Gated codex_attention_head
+Sequential mm
+Linear classifier
+forward(x_path, x_codex)
}
class MultiheadAttention {
+int embed_dim
+int num_heads
+forward(query, key, value)
}
class BilinearFusion {
+int dim1
+int dim2
+forward(vec1, vec2)
}
MCAT_Surv --> MultiheadAttention : "使用"
MCAT_Surv --> BilinearFusion : "使用"
```

**图表来源**
- [mica/models/model_coattn.py:12-124](file://mica/models/model_coattn.py#L12-L124)
- [mica/models/model_coattn.py:459-615](file://mica/models/model_coattn.py#L459-L615)

**章节来源**
- [hex/hex_architecture.py:15-46](file://hex/hex_architecture.py#L15-L46)
- [hex/utils.py:32-81](file://hex/utils.py#L32-L81)
- [mica/models/model_coattn.py:12-124](file://mica/models/model_coattn.py#L12-L124)

## 架构概览

整个系统采用分层架构设计，从数据预处理到最终的多模态分析：

```mermaid
flowchart TD
A[原始数据输入] --> B[数据预处理阶段]
B --> C[HEX模型训练]
B --> D[MICA模型训练]
C --> E[虚拟CODEX生成]
D --> F[多模态融合分析]
E --> F
F --> G[生存分析结果]
B1[CO-REGISTER H&E与CODEX] --> B
B2[提取补体信息] --> B
B3[构建数据集] --> B
C1[分布式训练] --> C
C2[特征分布平滑] --> C
D1[WSI特征提取] --> D
D2[虚拟CODEX特征] --> D
D3[多头注意力] --> D
```

**图表来源**
- [README.md:26-44](file://README.md#L26-L44)
- [hex/train_dist_codex_lung_marker.py:42-400](file://hex/train_dist_codex_lung_marker.py#L42-L400)
- [mica/train_mica.py:28-238](file://mica/train_mica.py#L28-L238)

## 详细组件分析

### 数据预处理流程

数据预处理是整个管道的关键步骤，确保输入数据的质量和一致性。

```mermaid
sequenceDiagram
participant User as 用户
participant HE as H&E图像
participant CODEX as CODEX图像
participant REG as 注册工具
participant PATCH as 裁剪工具
participant LABEL as 标签生成
User->>HE : 提供H&E SVS文件
User->>CODEX : 提供CODEX OME文件
User->>REG : 执行配准
REG->>REG : CO-REGISTER图像
REG->>PATCH : 生成注册后的CODEX
PATCH->>PATCH : 裁剪图像块
PATCH->>LABEL : 提取标记信息
LABEL->>User : 输出CSV标签文件
```

**图表来源**
- [extract_he_patch.py:9-78](file://extract_he_patch.py#L9-L78)
- [extract_marker_info_patch.py:21-74](file://extract_marker_info_patch.py#L21-L74)

#### 分割检查机制

系统内置了完整的分割检查机制，确保训练和验证集的正确性。

```mermaid
flowchart TD
A[加载分割文件] --> B{检查文件数量}
B --> |1个| C[单折验证模式]
B --> |5个| D[严格5折交叉验证]
C --> E{检查必需列}
E --> |缺失| F[错误：缺少必要列]
E --> |完整| G{检查患者重叠}
G --> |有重叠| H[错误：患者重叠]
G --> |无重叠| I[通过]
D --> J{检查每折完整性}
J --> |不完整| K[错误：折数不匹配]
J --> |完整| L{检查严格性约束}
L --> |违反| M[错误：分割不严格]
L --> |满足| N[通过]
```

**图表来源**
- [check_splits.py:72-104](file://check_splits.py#L72-L104)
- [check_splits.py:107-148](file://check_splits.py#L107-L148)

**章节来源**
- [extract_he_patch.py:9-78](file://extract_he_patch.py#L9-L78)
- [extract_marker_info_patch.py:21-74](file://extract_marker_info_patch.py#L21-L74)
- [check_splits.py:72-148](file://check_splits.py#L72-L148)

### HEX训练流程

HEX模型采用分布式训练策略，支持大规模数据集的高效训练。

```mermaid
sequenceDiagram
participant Trainer as 训练器
participant Data as 数据加载器
participant Model as 模型
participant FDS as 特征分布平滑
participant Dist as 分布式通信
Trainer->>Data : 加载训练数据
Data->>Model : 批量输入图像
Model->>FDS : 应用特征平滑
FDS->>Model : 平滑后的特征
Model->>Trainer : 预测输出
Trainer->>Dist : 计算梯度
Dist->>Trainer : 同步梯度
Trainer->>Model : 更新参数
Trainer->>Trainer : 保存检查点
```

**图表来源**
- [hex/train_dist_codex_lung_marker.py:245-396](file://hex/train_dist_codex_lung_marker.py#L245-L396)

#### 特征分布平滑机制

HEX引入了特征分布平滑（FDS）技术，有效改善了小样本情况下的模型性能。

```mermaid
flowchart TD
A[输入特征] --> B[桶化标签]
B --> C[计算运行统计]
C --> D[更新均值方差]
D --> E[应用平滑核]
E --> F[校准均值方差]
F --> G[输出平滑特征]
H[自适应平滑] --> I{检查epoch}
I --> |达到阈值| E
I --> |未达到| J[直接输出]
```

**图表来源**
- [hex/utils.py:116-326](file://hex/utils.py#L116-L326)

**章节来源**
- [hex/train_dist_codex_lung_marker.py:42-396](file://hex/train_dist_codex_lung_marker.py#L42-L396)
- [hex/utils.py:116-326](file://hex/utils.py#L116-L326)

### MICA测试流程

MICA的测试流程包含了生存分析和可解释性分析两个重要方面。

```mermaid
sequenceDiagram
participant Tester as 测试器
participant Model as MCAT模型
participant Data as 测试数据
participant IG as 集成梯度
participant Metrics as 性能指标
Tester->>Model : 加载预训练权重
Tester->>Data : 准备测试数据
Data->>Model : 输入WSI特征
Model->>Tester : 输出风险评分
Tester->>Metrics : 计算C-index
Tester->>IG : 可选：计算解释性
IG->>Tester : 返回注意力图
Tester->>Tester : 保存结果
```

**图表来源**
- [mica/test_mica.py:32-77](file://mica/test_mica.py#L32-L77)

#### 多模态融合策略

MICA实现了多种多模态融合策略，支持不同的注意力机制和池化方式。

```mermaid
classDiagram
class FusionStrategies {
<<interface>>
+concatenate()
+bilinear_pooling()
+attention_weighted()
}
class ConcatFusion {
+linear_layer
+relu_activation
+forward(h_path, h_codex)
}
class BilinearFusion {
+gated_units
+bilinear_pooling
+post_fusion_dropout
+forward(vec1, vec2)
}
class AttentionFusion {
+attention_weights
+weighted_sum
+forward(h_path, h_codex)
}
FusionStrategies <|.. ConcatFusion
FusionStrategies <|.. BilinearFusion
FusionStrategies <|.. AttentionFusion
```

**图表来源**
- [mica/models/model_coattn.py:616-680](file://mica/models/model_coattn.py#L616-L680)

**章节来源**
- [mica/test_mica.py:32-77](file://mica/test_mica.py#L32-L77)
- [mica/models/model_coattn.py:616-680](file://mica/models/model_coattn.py#L616-L680)

## 依赖关系分析

项目依赖关系复杂但结构清晰，主要依赖于多个开源框架和库。

```mermaid
graph TB
subgraph "核心框架"
A[PyTorch 2.4.0]
B[torchvision 0.20.1]
C[torchaudio 2.5.1]
D[timm 0.9.8]
end
subgraph "生物医学库"
E[openslide-python 1.4.1]
F[palom 0.1.0]
G[h5py 3.12.1]
H[nltk 3.9.1]
end
subgraph "第三方模型"
I[MUSK 1.0.0]
J[DINOv2]
K[CLAM]
L[MCAT]
end
subgraph "科学计算"
M[numpy 2.2.0]
N[pandas 2.2.3]
O[scipy 1.14.1]
P[scikit-learn 1.5.2]
end
A --> I
B --> I
D --> I
E --> F
G --> H
I --> J
I --> K
I --> L
```

**图表来源**
- [README.md:7-24](file://README.md#L7-L24)

**章节来源**
- [README.md:7-24](file://README.md#L7-L24)

## 性能考虑

### 训练优化策略

HEX采用了多种训练优化技术来提升模型性能：

1. **分布式训练**：支持多GPU并行训练，显著提升训练效率
2. **混合精度训练**：使用FP16半精度减少内存占用
3. **渐进式解冻**：从冻结所有参数到逐步解冻网络层
4. **自适应损失函数**：使用鲁棒损失函数提高模型稳定性

### 推理加速技术

在推理阶段，项目实现了多项优化措施：

1. **特征缓存**：利用FDS机制缓存历史统计信息
2. **批量处理**：优化数据加载和批处理策略
3. **内存管理**：合理管理GPU内存使用

## 故障排除指南

### 常见问题及解决方案

#### 数据预处理问题

**问题1：配准失败**
- 检查CO-REGISTER工具是否正确安装
- 确认H&E和CODEX图像格式兼容
- 验证图像分辨率匹配

**问题2：图像裁剪异常**
- 检查OpenSlide库版本兼容性
- 确认图像路径正确性
- 验证磁放大倍数设置

#### 模型训练问题

**问题3：分布式训练报错**
- 检查NCCL环境配置
- 验证GPU可见性和驱动版本
- 确认端口可用性

**问题4：内存不足**
- 调整批次大小
- 启用梯度累积
- 检查特征维度设置

#### 模型评估问题

**问题5：C-index计算异常**
- 检查生存时间数据格式
- 验证事件状态编码
- 确认风险评分范围

**章节来源**
- [check_splits.py:151-159](file://check_splits.py#L151-L159)
- [hex/train_dist_codex_lung_marker.py:28-39](file://hex/train_dist_codex_lung_marker.py#L28-L39)

## 结论

HEX项目代表了数字病理学和人工智能结合的重要进展。通过将传统的H&E染色图像转换为虚拟蛋白质表达谱，该项目为癌症研究提供了新的工具和视角。

### 主要优势

1. **临床实用性**：基于标准H&E染色，无需额外的昂贵检测
2. **可解释性**：提供空间生物学信息和可视化分析
3. **多模态整合**：支持多种生物标志物的同时分析
4. **开源生态**：基于成熟的开源框架和工具链

### 技术特色

- **特征分布平滑**：有效解决小样本学习问题
- **分布式训练**：支持大规模数据集的高效训练
- **多头注意力**：实现模态间的动态交互
- **集成梯度**：提供模型决策的可解释性

该项目为精准医学和肿瘤免疫治疗研究提供了强有力的技术支撑，具有重要的临床应用价值和科研意义。