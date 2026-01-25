# BBDM - 布朗桥扩散模型 AI 开发指南

## 项目定位
动漫线稿上色项目，基于**布朗桥扩散模型 (BBDM)** 实现图像到图像转换。当前实验阶段为 **A0-Baseline**（线稿 Concat 输入，无参考图），后续将扩展双流条件注入（A1-A4）。

## 核心架构速查


```
BrownianBridgeModel          ← 像素空间 BBDM (当前使用)
    └── LatentBrownianBridgeModel  ← 潜在空间 (继承自上面，使用 VQGAN)
```

**关键文件**:
- [Register.py](../Register.py) - 全局注册器，`Registers.dataset` / `Registers.runners`
- [dataset/custom.py](../dataset/custom.py) - 数据集实现，用 `@Registers.dataset.register_with_name()` 注册
- [runners/DiffusionBasedModelRunners/BBDMRunner.py](../runners/DiffusionBasedModelRunners/BBDMRunner.py) - 训练/推理流程
- [model/BrownianBridge/BrownianBridgeModel.py](../model/BrownianBridge/BrownianBridgeModel.py) - 扩散模型核心

## 开发工作流

### 训练/测试命令
```bash
# 训练
python main.py --config configs/A0-Baseline.yaml --train --sample_at_start --gpu_ids 0

# 推理采样
python main.py --config configs/A0-Baseline.yaml --sample_to_eval --gpu_ids 0 --resume_model path/to/ckpt

# 评估 (FID/LPIPS)
python preprocess_and_evaluation.py -f LPIPS -s source_dir -t target_dir -n 1
```

### 配置文件结构 (YAML)
```yaml
runner: "BBDMRunner"              # Runner 名称 (已注册)
model:
  model_type: "BBDM"              # "BBDM" (像素) 或 "LBBDM" (潜在)
  BB.params:
    UNetParams.in_channels: 6     # 输入通道 = x_t(3) + condition(3)
    objective: 'grad'             # 预测目标: 'grad'|'noise'|'ysubx'
data:
  dataset_type: 'anime_colorization'  # 数据集名称 (已注册)
```

## 扩展模式

### 添加新数据集
```python
# dataset/custom.py
@Registers.dataset.register_with_name('your_dataset')
class YourDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):  # stage: train/val/test
        ...
    def __getitem__(self, index):
        # 返回 dict 格式 (推荐):
        return {'GT': ..., 'lineart': ..., 'distorted': ..., 'name': ...}
        # 或 tuple 格式 (兼容旧代码):
        return (x, x_name), (x_cond, x_cond_name)
```

### 添加新 Runner
```python
# runners/DiffusionBasedModelRunners/YourRunner.py
@Registers.runners.register_with_name('YourRunner')
class YourRunner(DiffusionBaseRunner):
    def initialize_model(self, config): ...
    def loss_fn(self, net, batch, epoch, step, ...): ...
```

## 数据集目录结构
```
dataset/
├── train/
│   ├── color/    # 彩色 GT
│   └── sketch/   # 线稿
├── val/ ...
└── test/ ...
```

## BBDM 核心原理 (vs 标准扩散)

| 标准 DDPM | BBDM |
|-----------|------|
| 起点: 纯噪声 $\mathcal{N}(0,I)$ | 起点: 条件图像 $x_T = y$ |
| 逆过程需条件注入 | **无条件注入**，条件隐含在起点 |

**前向过程**: $x_t = (1-m_t)x_0 + m_t y + \sigma_t \epsilon$  
**推理**: 从 $y$ 出发，逐步预测 $x_0$

## 代码约定

1. **Batch 解包**: `BBDMRunner._unpack_batch()` 统一处理 dict/tuple 格式
2. **线稿通道**: 1 通道灰度，通过 `_prepare_cond()` repeat 为 3 通道
3. **图像归一化**: `to_normal: True` → [-1, 1]
4. **UNet 输入**: `concat(x_t, x_cond)` 通道拼接，需设 `in_channels=6`

## 常见修改点

| 任务 | 修改位置 |
|------|----------|
| 改 UNet 输入通道 | `configs/*.yaml` → `BB.params.UNetParams.in_channels` |
| 添加额外损失 | `BBDMRunner.loss_fn()` |
| 修改扩散过程 | `BrownianBridgeModel.q_sample()` / `p_sample()` |
| 添加 Cross-Attention | `model/BrownianBridge/base/modules/attention.py` |

## 调试提示
- 验证数据加载: 检查 `__getitem__` 返回的 tensor shape 和归一化范围
- 查看训练进度: TensorBoard → `results/<dataset>/<model>/logs/`
- 早停配置: `training.early_stopping.enabled/patience/min_delta`
