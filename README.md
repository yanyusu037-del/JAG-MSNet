This is the implementation of article: **"Multi-stage convolutional autoencoder network for hyperspectral unmixing"**.

**Citation**

If you find our work useful in your research or publication, please cite:

Yu Y, Ma Y, Mei X, et al. Multi-stage convolutional autoencoder network for hyperspectral unmixing[J]. International Journal of Applied Earth Observation and Geoinformation, 2022, 113: 102981.

**Usage**

Run MSNet_samson.py

If you want to run the code in your own data, you can accordingly change the input and change the parameters.

The format of input:

A represents the ground truth of abundance.

M represents the ground truth of endmembers.

M1 represents the initialization of endmembers by VCA.

Y represents the HSI.

---

# 数据集准备与运行说明

本文档记录 JAG-MSNet（MSNet，多阶段卷积自编码器高光谱解混）项目的数据集整理、格式转换与运行环境情况。

## 1. 项目概览

复现论文 **"Multi-stage convolutional autoencoder network for hyperspectral unmixing"**（Yu et al., IJAEOG 2022）。

仓库内为每个数据集一个独立脚本，结构基本一致（非模块化，代码以复制粘贴方式组织）：

| 脚本 | 数据集 | 空间尺寸 | 端元数 | 波段数 |
|------|--------|---------|--------|--------|
| `MSPNet_samson.py` | Samson | 95 × 95 | 3 | 156 |
| `MSNet_jasper.py` | Jasper Ridge | 100 × 100 | 4 | 198 |
| `MSNet-urban.py` | Urban | 307 × 307 | 5 | 162 |
| `MSNet_moni.py` | Moni（合成数据，本次新增） | 26 × 20 | 4 | 198 |

**模型结构**：`multiStageUnmixing`，三阶段由粗到细（1/4 → 1/2 → 全分辨率）。每阶段 `layer1/2/3` 卷积回归丰度，经 `Softmax` 归一化后由 `decoderlayer4/5/6`（1×1 卷积，无偏置）重建；三个 decoder 权重用 VCA 端元初始化。

**损失**：`total = SAD重构损失 + α·MSE + β·EdgeLoss`（EdgeLoss 内部使用 CharbonnierLoss）。

**评价指标**：`arange_A_E` 做索引对齐后，计算丰度 **RMSE** 与端元 **SAD**。

## 2. 运行环境

本任务全程使用 conda 环境 **vmunet**（WSL Ubuntu-22.04 内），不使用 base 环境。

```
conda 安装路径: /root/MSA/ENTER
环境路径:       /root/MSA/ENTER/envs/vmunet
解释器:         /root/MSA/ENTER/envs/vmunet/bin/python

torch 1.13.0+cu117   (cuda: True)
numpy 1.24.4
scipy 1.10.1
```

运行方式（需在项目根目录执行，脚本使用相对路径）：

```bash
cd /root/HuangXiaoYao/JAG-MSNet
/root/MSA/ENTER/envs/vmunet/bin/python MSNet_moni.py
```

## 3. 数据格式约定

每个 `.mat` 文件需包含以下键（脚本直接按此读取）：

| 键 | 形状 | 含义 |
|----|------|------|
| `Y` | (L, N) | 高光谱图像（波段 × 像素） |
| `A` | (P, N) | 丰度真值（端元数 × 像素） |
| `M` | (L, P) | 端元真值（波段 × 端元数） |
| `M1` | (L, P) | VCA 初始化的端元 |

## 4. 数据集清单

数据文件统一放在 `data/` 目录下。

| 文件 | 类型 | 说明 |
|------|------|------|
| `data/samson_dataset.mat` | MSNet 格式 | 键齐全（A/M/M1/Y），原项目自带 |
| `data/jasper.mat` | MSNet 格式 | 由 `jasper_dataset.mat` 转换而来 |
| `data/moni.mat` | MSNet 格式 | 由 `moni_dataset.mat` 转换而来 |
| `data/urban.mat` | 部分 | 仅含 Y/M1，**缺 A/M 真值** |
| `data/jasper_dataset.mat` | 原始副本 | 来自 `MBUNet-main/data` |
| `data/moni_dataset.mat` | 原始副本 | 来自 `MBUNet-main/data` |
| `data/urban_dataset.mat` | 原始副本 | 来自 `MBUNet-main/data` |

## 5. 格式转换细节

原始数据集来自 `../MBUNet-main/data`，其键名与 MSNet 脚本要求不同，转换过程如下：

- **jasper**：`A(4,10000)` 直接作为丰度真值；`GT(4,198)` 转置为 `M(198,4)`；`M1(198,4)` 由 Y 经 VCA（seed=0）提取；`Y(198,10000)`。
- **moni**：`A/M/M1/Y` 键本已齐全，统一为 float64 存入 `moni.mat`。
- **samson**：沿用项目原有 `samson_dataset.mat`，未改动（与 MBUNET 版本内容不同：端元归一化方式、像素排列有差异）。
- **urban**：原始文件仅含 `Y/SlectBands/nRow/nCol/nBand/maxValue`，**两个仓库均无 urban 丰度/端元真值**；仅能生成 `Y` 与 VCA 初始化 `M1(162,5)`。

VCA 实现复用自 `MBUNet-main/models/vca.py`（Nascimento & Dias, 2005）。

## 6. 脚本与数据对应

| 脚本 | 加载路径 |
|------|---------|
| `MSPNet_samson.py` | `data/samson_dataset.mat` |
| `MSNet_jasper.py` | `data/jasper.mat` |
| `MSNet-urban.py` | `data/urban.mat`（原写死 `urban5.mat`，已指向现有文件） |
| `MSNet_moni.py` | `data/moni.mat` |

`MSNet_moni.py` 由 `MSNet_jasper.py` 改写而来，关键改动：
- 支持**非方形输入**：`reshape` 改为 `(band, row, col)`；两处硬编码上采样尺寸改为动态获取（layer3 → `downsampling22.shape[2:]`，layer2 → `x.shape[2:]`），保证 EdgeLoss 两级尺度对齐成立。
- 移除脚本中未使用的 `torchstat` 导入。

## 7. 实测结果

`MSNet_moni.py` 在 vmunet 环境完整跑通（约 25 秒）：

```
Epoch: 0   | loss: 0.7291
Epoch: 700 | loss: 0.1293
mean_RMSE  0.0924
mean_SAD   0.0106
```

## 8. 已知问题与待办

1. **urban 缺真值**：`data/urban.mat` 无 `A/M`，`MSNet-urban.py` 结尾计算 RMSE/SAD 会报错。待获取 urban 丰度/端元真值后补入。
2. **jasper 脚本导入报错**：`MSNet_jasper.py` 第 14 行 `from torchstat import stat` 未安装且全文未使用，直接运行会在导入阶段失败（`MSNet_moni.py` 已移除该行）。
3. **脚本依赖根目录运行**：所有脚本使用相对路径（`data/...`），需在项目根目录下执行。

