# BBDM - Brownian Bridge Diffusion Model 代码库指南

## 项目概述
这是一个用于图像到图像转换任务的 **布朗桥扩散模型 (BBDM)** 实现，支持像素空间和潜在空间两种模式。

## BBDM vs 条件扩散模型 (核心创新)

| 特性 | 条件扩散模型 (DDPM/LDM) | BBDM (本项目) |
|------|------------------------|---------------|
| 起点 | 纯高斯噪声 $\mathcal{N}(0,I)$ | 条件图像 $x_T = y$ |
| 条件注入 | UNet 每步注入条件 $\epsilon_\theta(x_t, y, t)$ | **无条件注入** $\epsilon_\theta(x_t, t)$ |
| 理论保证 | 条件分布难以保证 | 布朗桥过程直接建模域间映射 |
| 适用场景 | 条件与输出相似度高的任务 | 跨域翻译任务 |

**关键区别**: BBDM 将 y 仅作为逆扩散起点，不在每步预测中作为条件输入。

## 核心架构

### 模型层次结构
```
BrownianBridgeModel (像素空间)
    └── LatentBrownianBridgeModel (潜在空间, 继承自 BrownianBridgeModel)
            └── 使用冻结的 VQGAN 进行编码/解码
```

### 关键组件
- **Register.py**: 全局注册器模式，用于注册 `dataset` 和 `runners`
- **runners/**: Runner 负责训练/测试流程控制
  - `BaseRunner.py`: 基础抽象类，处理 checkpoint、EMA、DDP
  - `BBDMRunner.py`: BBDM 专用 runner，处理模型初始化和优化器配置
- **model/BrownianBridge/**: 核心扩散模型实现
- **dataset/custom.py**: 数据集实现，使用 `@Registers.dataset.register_with_name()` 装饰器

## 配置系统
所有配置使用 YAML 文件，位于 `configs/` 目录：
- **Template-BBDM.yaml**: 像素空间模型
- **Template-LBBDM-f{4,8,16}.yaml**: 潜在空间模型（不同压缩比）

关键配置结构：
```yaml
runner: "BBDMRunner"           # 选择 runner
model:
  model_type: "LBBDM"          # "BBDM" 或 "LBBDM"
  BB:
    params:
      objective: 'grad'        # 'grad', 'noise', 'ysubx'
      loss_type: 'l1'          # 'l1' 或 'l2'
data:
  dataset_type: 'custom_aligned'  # 注册的数据集名称
```

## 开发者工作流

### 训练命令
```bash
python main.py --config configs/xxx.yaml --train --sample_at_start --save_top --gpu_ids 0
```

### 测试/采样
```bash
python main.py --config configs/xxx.yaml --sample_to_eval --gpu_ids 0 --resume_model path/to/ckpt
```

### 评估指标
使用 `preprocess_and_evaluation.py`:
```bash
python preprocess_and_evaluation.py -f LPIPS -s source_dir -t target_dir -n 1
python preprocess_and_evaluation.py -f diversity -s source_dir -n 1
```

## 扩展模式

### 添加新数据集
在 `dataset/custom.py` 使用装饰器注册：
```python
@Registers.dataset.register_with_name('your_dataset_name')
class YourDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        # stage: 'train', 'val', 'test'
```

### 添加新 Runner
```python
@Registers.runners.register_with_name('YourRunner')
class YourRunner(DiffusionBaseRunner):
    def initialize_model(self, config): ...
```

## 数据集目录约定
- **paired 任务**: `{path}/train/A`, `{path}/train/B`, `{path}/val/A` ...
- **colorization/inpainting**: `{path}/train`, `{path}/val`, `{path}/test`

## 像素空间 BBDM 训练与推理流程

### 训练流程 (forward)
```
输入: x (目标图像), y (条件图像)
    │
    ▼
1. 随机采样时间步 t ~ Uniform(0, T)
    │
    ▼
2. q_sample(): 前向扩散 (布朗桥)
   x_t = (1-m_t)*x + m_t*y + σ_t*noise
   objective = m_t*(y-x) + σ_t*noise  # 当 objective='grad'
    │
    ▼
3. UNet 预测: objective_recon = denoise_fn(x_t, t, context)
    │
    ▼
4. 计算损失: L1/L2(objective, objective_recon)
```

### 推理流程 (p_sample_loop)
```
输入: y (条件图像)
    │
    ▼
1. 初始化 x_T = y (从条件图像开始)
    │
    ▼
2. 循环 t = T → 0:
   ├─ UNet 预测 objective_recon
   ├─ predict_x0_from_objective(): 重建 x0
   └─ p_sample(): 计算 x_{t-1} (添加噪声的逆过程)
    │
    ▼
3. 输出: x_0 (生成的目标图像)
```

### 核心公式
- **前向过程**: `x_t = (1-m_t)*x_0 + m_t*y + σ_t*ε`
- **目标函数**: `grad` (默认), `noise`, `ysubx`
- **方差调度**: `δ_t = 2s*(m_t - m_t²)`，其中 s 控制采样多样性

### 数学原理 (来自论文)

**布朗桥边缘分布**:
$$q_{BB}(x_t | x_0, y) = \mathcal{N}(x_t; (1-m_t)x_0 + m_t y, \delta_t I)$$

**方差调度设计**:
- 原始布朗桥方差 $\frac{t(T-t)}{T}$ 会随 T 增大而爆炸
- 本项目采用 $\delta_t = 2s(m_t - m_t^2)$，保持方差在 [0, 0.5s] 范围
- 参数 s 控制多样性：s↑ 多样性↑，s=1 为默认值

**训练目标** (三种等价形式):
| objective | 预测目标 | 公式 |
|-----------|---------|------|
| `grad` (默认) | 梯度 | $m_t(y-x_0) + \sigma_t\epsilon$ |
| `noise` | 噪声 | $\epsilon$ |
| `ysubx` | 域差异 | $y - x_0$ |

## 核心模块说明

### UNetModel (`model/BrownianBridge/base/modules/diffusionmodules/openaimodel.py`)
基于 OpenAI 改进的 UNet，用于噪声/梯度预测：

| 参数 | 说明 |
|------|------|
| `image_size` | 输入图像尺寸 (像素空间=256, 潜在空间=64/32/16) |
| `model_channels` | 基础通道数 (默认128) |
| `channel_mult` | 每层通道倍数 (如 `(1,2,4,8)`) |
| `num_res_blocks` | 每个分辨率的残差块数量 |
| `attention_resolutions` | 应用注意力的分辨率 (如 `(32,16,8)`) |
| `condition_key` | 条件类型: `nocond`, `SpatialRescaler`, `first_stage` |

**结构**: Encoder → Middle → Decoder (带跳跃连接)

### Attention 模块 (`model/BrownianBridge/base/modules/attention.py`)

| 模块 | 用途 |
|------|------|
| `AttentionBlock` | QKV 自注意力，用于 UNet 中间层 |
| `SpatialTransformer` | 空间注意力 + 可选交叉注意力 (用于条件注入) |
| `CrossAttention` | 交叉注意力，query 来自特征，key/value 来自 context |
| `LinearAttention` | 线性复杂度的高效注意力 |

### ResBlock (`openaimodel.py`)
带时间步嵌入的残差块：
```python
# 时间嵌入通过 FiLM 调制 (use_scale_shift_norm=True)
h = norm(h) * (1 + scale) + shift
```

## 条件注入方式

| condition_key | 说明 | context 来源 |
|---------------|------|--------------|
| `nocond` | 无条件生成 | None |
| `SpatialRescaler` | 通过 CNN 降采样条件图像 | `CondStageParams` 配置 |
| `first_stage` | 使用 VQGAN encoder 特征 | 冻结的 VQGAN |

## 重要注意事项
- VQGAN checkpoint 路径必须在配置中指定: `model.VQGAN.params.ckpt_path`
- 支持 DDP 多 GPU 训练: `--gpu_ids 0,1,2,3`
- EMA 配置在 `model.EMA` 下，默认 30000 步后启用
- 图像默认归一化到 [-1, 1] (`to_normal: True`)
- 像素空间使用 `model_type: "BBDM"`，潜在空间使用 `model_type: "LBBDM"`
- 采样步数由 `sample_step` 控制 (默认 200)，`skip_sample: True` 启用跳步采样

## 加速采样 (DDIM-style)

基于非马尔可夫过程保持边缘分布的思想（类似 DDIM）：
- 训练时使用 T=1000 步，推理时使用 S=200 步
- 通过 `skip_sample: True` 和 `sample_step: 200` 配置
- `sample_type`: `linear` (均匀跳步) 或 `cosine` (余弦调度)
- `eta`: 控制随机性，eta=1 为完全随机，eta=0 为确定性采样

## 关键超参数

| 参数 | 位置 | 说明 |
|------|------|------|
| `max_var` | `BB.params` | 最大方差缩放因子 s，控制多样性 |
| `eta` | `BB.params` | DDIM 采样随机性，默认 1.0 |
| `mt_type` | `BB.params` | m_t 调度: `linear` (m_t=t/T) 或 `sin` |
| `num_timesteps` | `BB.params` | 训练步数 T，默认 1000 |
| `sample_step` | `BB.params` | 推理步数 S，默认 200 |
