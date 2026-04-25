# Introduction

模型的输入是全景RGB图片，输出是热力图（由原始RGB图和mask叠加得到）。基于两个项目：UAD（unsupported affordance detection）和GREAT。

UAD基于Behavior 1k、Omnigibson等数据集进行训练，输入为2D RGB常规视角图片，输出为热力图。GREAT主要使用其中的思维链CoT部分，通过调用foundaion model和预设prompt（一般为4条，对应物体的几何结构和affordance），对输入的图片进行处理，得到四条详细的描述，用于优化数据集和推理效果。

Behavior 1k数据集初始为3d，通过设定12个不同的角度得到12张同一物体、不同角度的RGB图片。通过DINOv2对这些图片进行特征提取，得到具体的物体得到不同部位（例如，cup会分为把手、杯口和杯子本身）。将得到的图片以及各部位的特征投影回3D物体，在经过聚类计算，得到各部分分类的3d物体点云。调用foundation model，通过prompt得到物体各部分的affordance（例如cup handler-handle-green）。也就是说，给物体的每个部分分配一个颜色，并且该颜色对应一种affordance。随后，经过计算选出最能代表该物体的几张图片，进行训练。

本项目所作的一个改动就是将UAD的数据集进行修改，由原先的简略的affordance改为详细的、有明确目标指向的CoT结果。另一个改动是修改UAD的特征提取部分，使得模型能够适应全景环境，方式包括滑窗分割读取、多层特征融合等。

---

# dataset
## pipeline_info.sql
| category_name | instance_name | frame_idx | | clustered_img_path | status| vlm_response | error_msg

训练所需的有以下四类:
1. processed_img. 是rgb图像经过处理后的tensor.
2. text/text embedding. text是VLM给出的针对每个部件的描述, 例如‘handle for gripping’, text embedding就是文本的嵌入向量, 可有openai text embedding model(online)或sentence transformer(local)两种模型选择
3. sim_proj. 训练 target，也就是模型要学习预测的热力图, 来自 H5 的 similarity_projections。例如, vlm_response中有"red": ["handle for gripping"], dataset 会用color_label_names 找到 "red" 对应第几个 cluster，然后从：similarity_projections[color_idx, cam_idx]取出对应的热力图作为 sim_proj。
4. mask. dataset 内部用于背景替换增强的 foreground mask，不直接作为模型输入或 loss target 返回。它由原始 RGB 中接近白色的背景区域计算出来

输入:
processed_img = 某个真实 RGB 视角
text_emb = 某句 affordance 描述的 embedding
监督:
sim_proj = 这句 affordance 对应颜色区域的热力图

# 项目文件结构

以下为上游项目 unsup-affordance (UAD) 的核心文件清单，ARLP 基于其改进。

## 一、核心源代码文件

### 主入口脚本

| 文件 | 功能 | ARLP 对应 |
|------|------|-----------|
| `src/inference.py` | 推理入口 (`AffordanceInference` 类) | `inference.py` |
| `src/train.py` | 训练入口 (`Trainer` 类) | `train.py` / `src/train.py` |
| `src/eval_agd.py` | AGD20K 数据集评估 | `eval_origin.py` / `eval_agd.py` |
| `src/eval_pano.py` | 全景滑动窗口评估 | `eval_windowslide.py` |
| `src/eval_pano_depth.py` | 带深度约束的全景评估 | 无 |
| `src/eval_pano_multiscale.py` | 多尺度评估 | 无 |

### 数据处理管线

| 文件 | 功能 | ARLP 对应 |
|------|------|-----------|
| `src/pipeline.py` | 数据策展管线 v1 | 无 |
| `src/pipelinev2.py` | 数据策展管线 v2 | 无 |
| `src/pipelinev3.py` | 数据策展管线 v3 | `pipelinev3.py` |
| `src/fusion.py` | 3D 特征融合 (多视图 RGBD → 点云) | `src/fusion.py` |
| `src/cluster.py` | K-means / PCA 聚类 | `src/cluster.py` |
| `src/get_vlm_response.py` | GPT-4o VLM 接口 (生成 affordance 描述) | 无 (但被 import) |
| `src/viz_cloud.py` | 点云可视化 | `viz_cloud.py` |

### 模型定义

| 文件 | 功能 | ARLP 对应 |
|------|------|-----------|
| `src/model/network.py` | `Conv2DFiLMNet` 网络架构 (FiLM 条件化卷积) | `model/network.py` |
| `src/model/dataset.py` | `RegionSimDataset` 数据集类 | `model/dataset.py` |

### 工具模块 (`src/utils/`)

| 文件 | 功能 | ARLP 对应 |
|------|------|-----------|
| `src/utils/__init__.py` | 包初始化 | 无 |
| `src/utils/file_utils.py` | YAML 配置加载、H5 数据存储 | `utils/file_utils.py` |
| `src/utils/img_utils.py` | 图像处理 + DINOv2 特征提取 | `utils/img_utils.py` |
| `src/utils/vlm_utils.py` | OpenAI / SentenceTransformer 文本嵌入 | `utils/vlm_utils.py` |
| `src/utils/eval_utils.py` | 评估指标 (KL, SIM, NSS) | `utils/eval_utils.py` |
| `src/utils/pcd_utils.py` | 点云操作 | `utils/pcd_utils.py` |
| `src/utils/postprocess_utils.py` | 深度过滤 + 形态学后处理 | `utils/postprocess_utils.py` |

## 二、配置文件 (`configs/`)

| 文件 | 用途 | ARLP 对应 |
|------|------|-----------|
| `configs/st_emb.yaml` | SentenceTransformer 嵌入 + ViT-S 配置 | 无 |
| `configs/oai_emb.yaml` | OpenAI 嵌入 + ViT-S 配置 | 无 |
| `configs/st_emb_vitl.yaml` | SentenceTransformer + ViT-L 配置 | 无 |
| `configs/oai_emb_vitl.yaml` | OpenAI 嵌入 + ViT-L 配置 | 无 |
| `configs/eval_agd.yaml` | AGD20K 评估专用配置 | 无 |

## 三、预训练模型 / 权重

| 文件 | 说明 | ARLP 对应 |
|------|------|-----------|
| `checkpoints/st_emb.pth` | SentenceTransformer 方案训好的模型 | 无 |
| `checkpoints/oai_emb.pth` | OpenAI 嵌入方案训好的模型 | 无 |
| `checkpoints/eval_agd.pth` | AGD20K 评估模型 (ViT-L) | 无 |
| `checkpoints/dinov2_vits14_pretrain.pth` | DINOv2 ViT-S/14 预训练权重 | 无 |
| `all-MiniLM-L6-v2/` | SentenceTransformer 本地模型 (384维) | 无 (运行时下载) |
| `models--facebook--dinov2-large/` | DINOv2-Large HuggingFace 缓存 | 无 (硬编码路径引用) |
| `dinov2/` | DINOv2 完整源码包 (含 ViT 模型定义) | 无 |
