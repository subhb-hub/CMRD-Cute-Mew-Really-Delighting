# SLST Direction-v2 两-fold探索指南

## 实验边界

- FACED：fold 5、10。
- SEED-IV：fold 1、8。
- 仅 seed 42、B4-SLST；训练 80 epoch，source-development 选择 checkpoint。
- 每 10 epoch 监视 source-train、source-development 和 target。target 只用于探索诊断，不参与 checkpoint、early stopping 或配置选择。
- 新结果写入 `runs/slst_direction_v2`，不会覆盖 `runs/slst_v1`。

## 新表示

| 条件 | 输入 |
| --- | --- |
| `H0_scalar_explicit` | magnitude + explicit scalar RJSD |
| `H1_raw_inner_explicit` | scalar + raw centered landmark inner products |
| `H2_pca_lowrank_explicit` | scalar + low-rank PCA projection，不除特征值 |
| `H3_hilbert_lowrank_explicit` | scalar + low-rank标准 Hilbert whitening |
| `H4_stable_hilbert_lowrank_explicit` | scalar + 带特征值阈值的低秩 Hilbert |
| `H5_hilbert_full_explicit` | scalar + full-rank Hilbert |
| `H6_stable_hilbert_lowrank_residual` | H4 + orthogonal residual |

每个方向条件保存 source-train 频带诊断：Gram 特征值、有效秩、条件数、稳定化 jitter、逐轴标准差、坐标能量/d0 和残差/d0。raw/PCA 坐标不具备 Hilbert 等距含义，因此后两个比例在对应 JSON 中为 `null`。

## 推荐执行顺序

一键完成校验、smoke 和两个数据集的 CoordinateGate：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_slst_direction_v2_one_click.ps1
```

脚本默认断点续跑，并在结束时打印任务状态。若只运行一个数据集，可增加 `-Dataset FACED` 或 `-Dataset SEEDIV`。

先校验和 smoke：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_slst_direction_v2.ps1 -Stage Validate -Dataset Both
powershell -ExecutionPolicy Bypass -File scripts/run_slst_direction_v2.ps1 -Stage Smoke -Dataset Both
```

主诊断阶段：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_slst_direction_v2.ps1 -Stage CoordinateGate -Dataset Both
```

`CoordinateGate` 每个数据集为 2 folds × 7 conditions = 14 个任务；两个数据集合计 28 个任务。先分析这一阶段，不要直接连续启动后续全部矩阵。

根据 CoordinateGate 再选择：

```powershell
# r={2,4}, tau={1e-2,1e-3,1e-4}
powershell -ExecutionPolicy Bypass -File scripts/run_slst_direction_v2.ps1 -Stage StabilityGate -Dataset Both

# K/r 覆盖筛选；可把 BestCondition 改为 CoordinateGate 最优固定坐标
powershell -ExecutionPolicy Bypass -File scripts/run_slst_direction_v2.ps1 -Stage CoverageGate -Dataset Both `
  -BestCondition H4_stable_hilbert_lowrank_explicit

# L0-L5 优化可达性；只在固定版本有继续价值时运行
powershell -ExecutionPolicy Bypass -File scripts/run_slst_direction_v2.ps1 -Stage LearnabilityGate -Dataset Both
```

所有矩阵默认断点续跑。只有明确需要覆盖重跑时才加 `-NoResume`。

## 终端状态与落盘证据

终端逐层打印：

- `MATRIX-PLAN`：fold、条件数和总任务数；
- `MATRIX-TASK i/N`、`TASK-START`、`RESUME-SKIP`；
- `EPOCH`：当前 epoch、loss、两个 LR、landmark drift/update、epoch 耗时和 ETA；
- `MONITOR`：每 10 epoch 的 source-train、source-dev、target Macro-F1、当前最佳 epoch；
- `EARLY-STOP`、`TASK-COMPLETE` 或 `TASK-FAILED`。

任务目录保存 `status.json`、`monitoring.json`、`training_history.csv`、`coordinate_diagnostics_source_train.json`、逐监视点预测/混淆矩阵、`best_model.pt` 和 `result.json`。可随时查看：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_slst_direction_v2.ps1 -Stage Status
powershell -ExecutionPolicy Bypass -File scripts/run_slst_direction_v2.ps1 -Stage Summarize
```

## 可学习 landmark 诊断

L1-L5 每个 epoch 记录 classification、anchor、diversity、coverage 对 landmark logits 的独立梯度范数，以及组合梯度、logit 更新量、概率 L1 漂移、JSD drift、pairwise JSD、Gram 最小特征值和正则项加权前后数值。前三个 epoch 冻结的配置梯度和更新应为零；解冻后若漂移仍远低于 `1e-4`，先判定为优化不可达，不作性能结论。
