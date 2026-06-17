## Gaussian Physics-Aligned Geometry Diagnosis

- **Ours PLY**: `/mnt/gxf/PanoSplat/output/inference/apartment_0604_1930_2/gaussians/gaussians.ply`
- **Baseline PLY**: `/mnt/gxf/A_work/PanoSplatt3R/result/infer/apartment/gaussians.ply`
- **Ours SHA256**: `db8c44a0e5dde34b` | **Baseline SHA256**: `4211badd6ffefee7`
- **Opacity 激活**: Ours=`sigmoid(3dgs_logit)` | Baseline=`sigmoid(3dgs_logit)`
- **Scale 激活**: Ours=`exp(3dgs_log_scale)` | Baseline=`exp(3dgs_log_scale)`
- **硬剪枝阈值**: opacity > 0.05

### 核心物理指标（显形高斯）

| 指标 | PanoSplat | PanoSplatt3R | 更优 | 说明 |
| --- | ---: | ---: | :---: | --- |
| PLY 总顶点数 | 1,925,899 | 4,836,273 | — | PLY header 原始 Gaussian 数量 |
| Invisible Zombie Points (opacity≤阈值) | 0 | 0 | tie | 不参与渲染的隐形点；阈值=0.05 |
| Active Effective Points | 1,925,899 | 4,836,273 | — | opacity > 0.05，参与全部指标 |
| Mean Visible Opacity | 0.5733 | 0.3171 | — | 显形点平均不透明度（已激活） |
| Raw Chamfer L1 ↓ | 0.292931 | 0.926257 | **ours** | 无 ICP 对齐：列分别为 Ours→Baseline / Baseline→Ours 单向均值；对称 L1=0.609594 |
| Mean Max Scale | 0.014842 | 0.017752 | — | 显形高斯最大半轴均值；过大=臃肿，过小=漏光 |
| Anisotropy Ratio (>4) ↓ | 6.86% | 0.74% | **baseline** | 最大轴/最小轴 > 阈值 的占比；量化拉丝飞线 |

### 显形高斯局部几何（辅助）

| 指标 | PanoSplat | PanoSplatt3R | 更优 | 说明 |
| --- | ---: | ---: | :---: | --- |
| Mean Roughness (visible) ↓ | 0.002383 | 0.002389 | **ours** | 显形中心 PCA 平面残差（Ours 对齐后 / Baseline 原坐标） |
| SOR Std (visible) ↓ | 0.003867 | 0.010102 | **ours** | 显形点邻居距离标准差 |
| Sparse Voxel Ratio (visible) ↓ | 6.54% | 8.74% | **ours** | 0.1 m 网格中 1–2 点体素占比 |

### 对齐信息（仅辅助局部几何行）

- **requested**: icp
- **method**: icp
- **fitness**: 0.8281437925965521
- **rmse**: 0.09222749830346204

### 诊断结论

- **全局几何漂移**：Raw Chamfer L1 较大（Ours=0.292931, Baseline=0.926257），未经对齐的空间偏差会直接拉低 2D PSNR。
- **拉丝/飞线退化更严重**：Anisotropy Ratio>4 占比高于 Baseline，高长宽比高斯会在全景渲染中产生毛刺悬浮物。

_分析耗时: 433.1s_
