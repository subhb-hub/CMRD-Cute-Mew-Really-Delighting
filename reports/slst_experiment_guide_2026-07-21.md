# SLST Landmark-JSD 实验指南

## 已实现的实验定义

主路径为：原生 PSD → magnitude/shape 分解 → source-train atlas → 可微 JSD-Hilbert 坐标 → Band–Channel–Temporal Transformer。

- `A0–A6`：输入特征消融，均使用同一套 SLST 容量。
- `B0–B4`：展平、Band、Channel、Temporal 结构消融。
- `C1–C4`：随机/固定初始化、无约束学习、anchor、anchor+diversity+coverage。
- `K=4/8/16`：独立 atlas 与独立任务目录，不会互相覆盖。
- FACED：官方 subject fold，以及三个互斥的 unseen-stimulus rotation。
- SEED-IV：沿用固定缓存中的 15-fold LOSO 和两个 source-development subjects。

`A5_centered_landmark` 明确定义为 `[standardized magnitude, d(q,a0)-d(q,ak)]`。`A6_hilbert_landmark` 为 `[standardized magnitude, z, e_perp]`，使用 `G + 1e-4 I` 的 Cholesky 坐标。

## 协议边界

当前版本是实验探索协议。每 10 epoch 会保存 source-train、source-development 和 target 的 trial-level 指标、subject-level 指标、预测与混淆矩阵；checkpoint 只由 source-development subject-averaged Macro-F1 决定，target 指标不参与梯度、early stopping 或配置选择。

因此 target 曲线只能用于诊断过拟合和域差异，不能作为无偏正式结果。正式论文阶段仍应锁定配置后关闭 target monitoring，并执行 source-dev 选 epoch、全 source refit、最后一次 target evaluation 的嵌套流程。

## 推荐执行顺序

在项目根目录运行：

```powershell
# 1. 校验两套数据、配置和已有固定缓存
powershell -ExecutionPolicy Bypass -File scripts/run_slst.ps1 -Stage Validate -Dataset Both

# 2. 纯合成 smoke；不会启动正式数据训练
powershell -ExecutionPolicy Bypass -File scripts/run_slst.ps1 -Stage Smoke -Dataset Both

# 3. 复用 runs/spectral_atlas_v1_seed42 的 base trial cache，建立 mmap 结构缓存
powershell -ExecutionPolicy Bypass -File scripts/run_slst.ps1 -Stage PreparePack -Dataset Both

# 4. 三个代表 fold、seed 42 的快速主路径
powershell -ExecutionPolicy Bypass -File scripts/run_slst.ps1 -Stage PilotQuick -Dataset Both

# 5. K、输入特征、结构、可学习 atlas 的逐组消融
powershell -ExecutionPolicy Bypass -File scripts/run_slst.ps1 -Stage KScreen -Dataset Both
powershell -ExecutionPolicy Bypass -File scripts/run_slst.ps1 -Stage FeatureA -Dataset Both
powershell -ExecutionPolicy Bypass -File scripts/run_slst.ps1 -Stage ArchitectureB -Dataset Both
powershell -ExecutionPolicy Bypass -File scripts/run_slst.ps1 -Stage LandmarkC -Dataset Both
```

确认 PilotQuick 的显存、速度、source-dev 曲线和 target gap 后，再运行：

```powershell
# 三个代表 fold、3 seeds
powershell -ExecutionPolicy Bypass -File scripts/run_slst.ps1 -Stage Pilot -Dataset Both

# 所有 FACED 10 folds / SEED-IV 15 folds，只跑主结果条件
powershell -ExecutionPolicy Bypass -File scripts/run_slst.ps1 -Stage Full -Dataset Both

# FACED 三个 unseen-stimulus rotations；只跑最终三种模型
powershell -ExecutionPolicy Bypass -File scripts/run_slst.ps1 -Stage Strict -Dataset FACED
```

所有矩阵默认恢复已完成任务。只有明确需要重跑时才加 `-NoResume`。

## 单任务与可选 V-REx

```powershell
conda run --no-capture-output -n cmrd python scripts/run_slst.py run `
  --config configs/slst/faced_v1.yaml `
  --run-root runs/slst_v1 `
  --fold 1 `
  --protocol conventional_subject_holdout `
  --condition C4_regularized `
  --architecture B4_slst `
  --seed 42
```

C5 只作为最后的可选项，用 C4 配置增加 source-subject V-REx 权重：

```powershell
conda run --no-capture-output -n cmrd python scripts/run_slst.py run `
  --config configs/slst/faced_v1.yaml `
  --run-root runs/slst_v1 `
  --set slst.vrex_weight=0.1 `
  --fold 1 `
  --protocol conventional_subject_holdout `
  --condition C4_regularized `
  --architecture B4_slst `
  --seed 42
```

V-REx 权重会进入 task signature，因此不会覆盖普通 C4 结果。

## 输出

任务目录包含 `status.json`、`monitoring.json`、`training_history.csv`、`best_model.pt`、每 10 epoch 的预测和混淆矩阵，以及 `result.json`。任务路径包含 dataset、protocol、fold、K、condition、architecture、seed 和 task signature；改变训练或 atlas 超参数不会误用旧结果。

汇总命令：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_slst.ps1 -Stage Status -RunRoot runs/slst_v1
powershell -ExecutionPolicy Bypass -File scripts/run_slst.ps1 -Stage Summarize -RunRoot runs/slst_v1
```

主比较应优先看 target 的 `subject_averaged_macro_f1` 与 `worst_quartile_subject_macro_f1`，同时检查普通 Macro-F1、balanced accuracy、逐主体结果和混淆矩阵。
