# FACED / SEED-IV Spectral-Atlas 实验运行说明

## 目的与证据边界

本实验比较单标量 RJSD、保留方向的 Landmark-JSD/Nyström 坐标、完整 ILR+功率、完整 log-PSD、DE，以及同维 PCA/随机投影。FACED 和 SEED-IV 共享表示与分类器代码，但不共享信号预处理：

- FACED 直接读取官方 `Processed_data`，不重新做 ICA、重参考或裁剪。
- SEED-IV 读取现有已验证 ICA/LOSO 缓存中的 cleaned trials。
- 所有锚点、参考、PCA 和标准化只使用 source-train。
- MLP 每 10 epoch 保存 source-train、source-development 和 target 的 Accuracy、Balanced Accuracy、Macro-F1 与混淆矩阵。
- target 曲线仅用于探索诊断，不参与停止、checkpoint、超参数或表示选择；checkpoint 固定为最终 epoch。因此这些结果不能当作独立测试证据。

## 表示矩阵

- `de`
- `log_band_power`
- `scalar_jsd_power`
- `raw_landmark_power_cap{1,2,4,8}`
- `nystrom_landmark_power_cap{1,2,4,8}`
- `ilr_power_full`
- `log_psd_full`
- `pca_ilr_power_cap{1,2,4,8}`
- `random_ilr_power_cap{1,2,4,8}`

`cap` 会按每个频带的真实自由度 `F_b-1` 截断。FACED 的每通道维数依次为 10、15、22、31，完整表示为 46；SEED-IV 的完整表示为 49。训练输入使用每个 trial 的时间均值与标准差拼接，因此结果文件中的最终维数还会乘以通道数和 2。

## 推荐执行顺序

```powershell
# 1. 只读校验真实数据和缓存
.\scripts\run_spectral_atlas.ps1 -Stage Validate -Dataset Both

# 2. 不访问真实 EEG 的合成 smoke
.\scripts\run_spectral_atlas.ps1 -Stage Smoke -Dataset Both

# 3. 生成可复用的基础谱缓存；这是耗时阶段，但不训练模型
.\scripts\run_spectral_atlas.ps1 -Stage PrepareBase -Dataset Both

# 4. fold 1 探索：两个数据集；FACED 自动同时跑两种任务
.\scripts\run_spectral_atlas.ps1 -Stage Pilot -Dataset Both

# 5. 查看状态和汇总
.\scripts\run_spectral_atlas.ps1 -Stage Status
.\scripts\run_spectral_atlas.ps1 -Stage Summarize
```

`Pilot` 使用 `pooled_mlp` 比较 DE、scalar RJSD、Nyström-cap4、完整 ILR、完整 log-PSD、PCA-cap4 和随机-cap4。确认缓存、曲线、类别覆盖与维数都正常后，再运行：

```powershell
# fold 1 的完整 cap 消融，只跑 MLP
.\scripts\run_spectral_atlas.ps1 -Stage Core -Dataset Both

# 全部 FACED 10 folds / SEED-IV 15 folds，三类浅模型；成本最高
.\scripts\run_spectral_atlas.ps1 -Stage Full -Dataset Both
```

可以用 `-Dataset FACED` 或 `-Dataset SEEDIV` 单独运行。默认使用 `conda cmrd`；可用 `-RunRoot` 改变输出目录。任务默认跳过已有完整结果；只有明确需要重跑时才加 `-NoResume`。

## FACED 协议

- `conventional_subject_holdout`：新主体，但 source/target 使用同一固定视频面板。
- `subject_and_stimulus_holdout`：source-train、source-development、target 分别使用互不重叠的视频；每个情绪内部完成 train/dev/test 划分。

两者必须分别汇报。第二项只是 FACED 小刺激样本下的严格压力测试，不能单独解释为通用未见刺激泛化。

## 主要产物

- 基础谱缓存：`runs/spectral_atlas_v1_seed42/cache/base/`
- source-only atlas/PCA 状态：`cache/folds/<dataset>/<protocol>/fold-XX/<hash>/state.*`
- pooled 表示：同目录下 `pooled_features.npz`
- 任务结果：`tasks/<dataset>/<protocol>/fold-XX/<condition>/<model>/result.json`
- 每 10 epoch 监视：任务目录中的 `monitoring.json`、`training_history.csv`
- 混淆矩阵：任务目录中的 `confusions/*.csv` 和可用时的 `*.png`
- 总表：`summary.csv`

## 进入下一阶段的门槛

只有当 Nyström/Landmark 表示同时优于 scalar RJSD、同维随机投影和同维 PCA，并在明显低于完整表示的维数下接近 ILR/log-PSD，而且在 FACED 严格协议和 SEED-IV LOSO 中方向一致，才建议继续做 V-REx、CSD 或 residual fusion。
