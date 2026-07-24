# FACED PSD-JSD 模型改进实验报告

日期：2026-07-18  
结论级别：单个预定义外层 fold 的锁后结果；不是完整多外层 fold 论文结果

## 结论先行

原 Notebook 的主要问题确实在模型与优化路径，而不是代码不能反向传播：110.66M 参数的 CNN+Transformer 在 source-dev 上退化为近单类预测，严格单批检查也没有真正过拟合。

本轮把模型改成 186,833 参数的 `NativeBandFlattenTemporalTransformer`，保持逐频点 PSD-JSD 输入完全不变，并完成了以下闭环：

1. 严格 18 样本可学习性门禁通过；
2. 全规模 source-train 可学习性从单类坍缩修复到可达 100%；
3. 三折 source-only 主体级内层交叉验证全部通过非退化门槛；
4. 仅基于 source 证据锁定候选和固定训练 epoch；
5. 在 111 个 source 主体上重训后进行锁后外层目标评估。

最终外层目标（主体 0–11，共 336 trials）：Accuracy 14.88%，Balanced Accuracy 14.81%，Macro-F1 14.51%，9 类均有预测。模型不再坍缩，但效果仍弱；当前主问题已经从“模型不会学”变为“模型记住 source，却难以跨主体泛化”。

## 改了什么

### P0：强制门禁与可审计训练

- 严格 sanity set：每类 2 个样本，共 18 个；关闭 dropout、weight decay、label smoothing 和 gradient clipping；必须达到 accuracy ≥99% 且 loss ≤0.05。
- 每 epoch 保存 source-train 与 source-dev 的 Accuracy、Balanced Accuracy、Macro-F1、预测类别数、预测直方图、预测熵、logit 标准差、裁剪前梯度范数及裁剪比例。
- 非退化选择门槛：source-train 高于机会水平、source-dev Balanced Accuracy 高于机会水平、至少预测 3 类。
- 配置、代码、协议哈希分离保存；不同协议禁止写入同一 run。
- source 选择阶段禁止读取外层目标主体，并保存每折隔离审计。

### P1：小模型和优化消融

- 把原约 110.66M 参数模型缩小到 0.18–0.41M 参数候选。
- 对比带常规正则的稳定配置与关闭正则的纯可学习性配置。
- 证实早期 gradient clipping 会大量截断梯度，但关闭裁剪仍不能解决原 CNN 的全量单类坍缩。

### P2：保留 channel×band 身份

失败的 `NativeBandChannelTemporalTransformer` 在 band 和 channel 两级都做注意力加权平均；它能记住 18 个样本和少量主体，却在 74 个主体训练时稳定停在类别先验。

新模型保持相同输入与原生频带长度切片，但结构改成：

`原生频点编码 → 拼接五个 band → 保留 30 个固定 channel 的展平投影 → 时间 Transformer → CLS 分类`

它不在时间建模前平均掉 band/channel 身份，同时仍严格忽略补零频点。单元测试验证任意修改 padding 值都不会改变输出。

## 实验矩阵

| 阶段 | 候选 | 参数量 | Sanity | Source-dev BAcc | Source-dev Macro-F1 | 预测类数 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| 原 Notebook smoke | 大 CNN+Transformer | 110.66M | 未通过严格门禁 | 未完整记录 | 2.78% | 近 1 | 淘汰 |
| 稳定正则 smoke | `cnn_tiny_stable` | 411,281 | 通过 | 12.04% | 5.47% | 2 | 坍缩，淘汰 |
| 稳定正则 smoke | `native_band_tiny_stable` | 184,325 | 通过 | 15.74% | 5.65% | 2 | 坍缩，淘汰 |
| 纯可学习性 smoke | `cnn_tiny_clean` | 411,281 | 通过 | 11.11% | 2.78% | 1 | 坍缩，淘汰 |
| 纯可学习性 smoke | `native_band_tiny_clean` | 184,325 | 通过 | 14.12% | 11.34% | 8 | 少量主体可学 |
| 原生 band/channel pooling 三折 | `native_band_tiny_clean` | 184,325 | 3/3 | 11.11% ± 0.00% | 2.78% ± 0.00% | 1 | 全规模坍缩，淘汰 |
| 保留 channel×band 单折诊断 | `native_band_flatten_clean` | 186,833 | 通过 | 11.66% | 10.55% | 9 | 全规模可学 |
| 保留 channel×band 三折 | `native_band_flatten_clean` | 186,833 | 3/3 | **12.55% ± 0.37%** | **12.22% ± 0.34%** | 9 | 3/3 通过并锁定 |

三折最佳 epoch 为 `[23, 19, 18]`，锁后最终训练长度预先定义为其中位数 19。外层目标在此规则和模型权重固定后才加载。

## 锁后外层目标结果

| 指标 | 结果 |
|---|---:|
| Source trials | 3,108 |
| Target trials | 336 |
| 最终 source Accuracy | 93.82% |
| 最终 source Macro-F1 | 93.76% |
| Target Accuracy | 14.88% |
| Target Balanced Accuracy | 14.81% |
| Target Macro-F1 | 14.51% |
| Target 预测类别数 | 9 |
| JSD 重构最大误差 | 5.96×10⁻⁸ |

九分类 Balanced Accuracy 的机会水平是 11.11%；目标 BAcc 高出 3.70 个百分点。由于 neutral 类每主体有 4 个视频、其余类为 3 个，majority-class Accuracy 基线为 14.29%，所以 14.88% Accuracy 只高 0.60 个百分点。不能把该结果表述为强分类性能。

混淆矩阵显示没有单类坍缩，但没有形成清晰对角线；`joy` 被预测 56 次，`tenderness` 仅 17 次。Source 93.8% 与 target 14.5% Macro-F1 的差距是强过拟合/跨主体分布差异的直接证据。

## 协议事件

第一次锁后目标前向已算出指标，但绘图时 Matplotlib 误选 Tk 后端并因 `init.tcl` 缺失退出，指标未写入 JSON。随后切换到 Agg，从已经保存且未改变的 source 模型恢复，重新执行同一目标前向并落盘。

因此审计中如实记录 `target_forward_passes=2`。两次之间没有修改模型、权重、训练 epoch、预处理或候选，也没有使用第一次未持久化的指标调参；但严格措辞不应声称“物理上只执行了一次目标前向”。

## 可信边界

- 可信：本轮证明了新结构能消除全量训练的单类坍缩，并在固定的三折 source-only 内层 CV 中稳定略高于机会水平。
- 可信：外层目标主体未参与参考、标准化、checkpoint、epoch 或候选选择。
- 有限：外层结果只覆盖官方 outer fold 1，不能替代完整 outer-fold 重复评估。
- 有限：目标结果弱，当前模型不能作为论文主结果。
- 禁止：不能再根据 outer fold 1 的目标混淆矩阵或指标调整当前协议；下一版必须使用新的 source-only 开发协议或新的预注册外层 fold。

## 下一轮模型侧优先级（保持特征不变）

1. **主体域不变训练**：在 source 内层 CV 中比较 subject-balanced episodic batch、subject-adversarial head 或 GroupDRO；目标是缩小 93.8%→14.5% 的泛化鸿沟。
2. **同类跨主体约束**：加入 supervised contrastive loss 或同类跨主体 prototype consistency；只用 source 主体构造正负对。
3. **受控正则回加**：在已证明可学习的 flatten 结构上，分别单独测试 dropout、weight decay、mixup，不能一次同时改变多项。
4. **通道拓扑建模**：在保留固定 channel 身份的基础上加入基于真实电极拓扑的图层或位置偏置，避免重新使用任意 channel index 卷积。
5. **完整外层协议**：若要形成论文证据，预先冻结下一版模型后跑全部官方 outer folds，并报告均值、标准差、每折混淆与失败率。

## 关键产物

- Source-only 三折锁：`runs/faced_psd_jsd_flatten_inner_cv_seed42/source_selection_lock.json`
- 三折汇总：`runs/faced_psd_jsd_flatten_inner_cv_seed42/inner_cv_summary.json`
- 每折训练曲线与隔离审计：`runs/faced_psd_jsd_flatten_inner_cv_seed42/inner_cv/`
- 锁后目标结果：`runs/faced_psd_jsd_flatten_locked_outer_test_seed42/target_result.json`
- 锁后混淆矩阵：`runs/faced_psd_jsd_flatten_locked_outer_test_seed42/target_confusion.png`
- 锁后模型：`runs/faced_psd_jsd_flatten_locked_outer_test_seed42/locked_source_model.pt`
- 目标前隔离审计：`runs/faced_psd_jsd_flatten_locked_outer_test_seed42/pre_target_isolation_audit.json`
- 模型实现：`src/cmrd/models/faced_psd_jsd.py`
- Source-only runner：`src/cmrd/faced_psd_jsd_experiment.py`
- 锁后评估器：`src/cmrd/faced_psd_jsd_locked_test.py`
- 最终 source-only 配置：`configs/faced/psd_jsd_flatten_inner_cv.yaml`

## 验证命令

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
conda run --no-capture-output -n cmrd python -m unittest discover -s tests -p 'test_faced_psd_jsd_experiment.py' -v
conda run --no-capture-output -n cmrd python scripts\run_faced_psd_jsd_recommended.py --config configs\faced\psd_jsd_flatten_inner_cv.yaml --stage status
conda run --no-capture-output -n cmrd python scripts\run_faced_psd_jsd_locked_outer_test.py --config configs\faced\psd_jsd_flatten_inner_cv.yaml
```

最后一条命令当前是幂等读取：检测到 `target_result.json` 后只打印已有结果，不会再次执行目标前向。
