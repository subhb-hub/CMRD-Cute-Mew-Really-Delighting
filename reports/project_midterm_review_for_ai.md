---
schema_version: "1.0"
document_type: "machine_readable_research_midterm_review"
project: "CMRD-Cute-Mew-Really-Delighting"
title: "EEG cross-subject emotion recognition project midterm review"
language: "zh-CN"
generated_date: "2026-07-20"
primary_metric: "Macro-F1"
metric_scale: "0_to_1"
datasets:
  - "SEED"
  - "SEED-IV"
  - "DEAP"
  - "FACED"
primary_features:
  - "DE"
  - "RJSD"
  - "sqrt-JSD"
  - "Fisher-Rao"
  - "histogram/log-PSD"
evidence_levels:
  formal: "Complete folds, fixed training policy, target excluded from model selection."
  completed_matched: "Complete matched comparison, but still internal project evidence."
  diagnostic: "Target-monitored, single-fold, oracle, post-hoc, or otherwise exploratory."
  infrastructure: "Validates preprocessing or execution, not predictive efficacy."
  incomplete: "Interrupted, pending, or missing comparison conditions."
canonical_sources:
  - "runs/fixed_protocol_seed42/"
  - "runs/faced_native_compact_base_seed42/"
  - "runs/faced_axial_no_cls_whitebox_fold1_diag10/"
  - "runs/faced_rjsd_shape_magnitude_ablation_fold1_diag10/"
  - "runs/faced_subject_reference_oracles_fold1/"
  - "runs/faced_normative_probe_fold1_seed42/"
  - "reports/deap_fold1_diagnostic_data.json"
---

# EEG 跨主体情绪识别项目中期复盘（AI 读入版）

## 1. 使用说明

本文件是一份自包含的项目状态快照，供其他 AI、代码代理或研究协作者直接读入。除非某条记录明确标记为 `formal` 或 `completed_matched`，否则不得把其中的最高目标域分数当作独立测试结果。

解析时遵守以下优先级：

1. `formal` 结果优先于所有其他结果。
2. `completed_matched` 可用于同协议内部比较，但不能自动等同于外部验证。
3. `diagnostic` 只用于判断训练健康、机制方向和后续优先级。
4. `infrastructure` 不包含模型有效性结论。
5. `incomplete` 不能按负结果解释。

## 2. Technical Summary

### 2.1 当前主要结论

项目当前并非“所有训练代码都失效”。FACED 的完整 10-fold 匹配实验表明，在完全相同的 866,953 参数 HCBT 和 100 epoch 训练条件下：

- DE：Macro-F1 `0.554800 ± 0.067235`
- native Fisher-Rao PCA：Macro-F1 `0.113564 ± 0.012088`
- native sqrt-JSD：Macro-F1 `0.047864 ± 0.026925`

因此，训练管线能够学习 DE；持续失败的是紧凑 RJSD/Fisher-Rao 表征及其跨主体迁移。

### 2.2 RJSD 的跨数据集表现不稳定

SEED/SEED-IV 固定协议矩阵包含 `600/600` 个完成任务，即：

`2 datasets × 4 representations × 5 models × 15 LOSO folds`。

在相同 Hierarchical Attention 模型下：

- SEED：RJSD-z 比 DE-z 低 `0.058322` Macro-F1，95% bootstrap CI `[−0.098914, −0.020706]`，Holm `p=0.026855`。
- SEED-IV：RJSD-z 比 DE-z 高 `0.018295`，95% bootstrap CI `[−0.023763, 0.060066]`，Holm `p=0.842407`。
- SEED-IV 最佳表征不是 RJSD，而是完整 histogram，Macro-F1 `0.856720`。

结论：RJSD 不是全局优于 DE 的稳定表征，其效果依赖数据集。

### 2.3 已定位的 RJSD 退化机制

FACED A–E 消融显示以下三个问题均真实存在：

1. 1 秒谱估计的概率谱噪声过大。
2. 标量距离压缩丢失功率/能量幅度。
3. 无符号距离丢失相对参考的方向。

将频谱支持扩展到 4 秒、加入 magnitude bypass、加入 signed direction 后，fold-1 目标监测 Macro-F1 从 `0.05483` 提高到 `0.17865`；50 epoch 的 signed shape+magnitude 达到 `0.20465`。改进有效，但仍明显低于同一简化模型的 DE `0.45198`。

### 2.4 剩余瓶颈

当前剩余问题主要是主体与刺激域错位，而不是模型容量不足：

- RJSD-E 的 source Macro-F1 可到 `0.78404`，target 只有 `0.20465`。
- 未标注 subject K27 reference oracle 仅从 global reference 的 `0.22851` 提高到 `0.24975`。
- 使用目标标签辅助构造的 class-balanced oracle 也只有 `0.26610`。
- FACED pseudo-reference 在常规主体留出下最好 `0.31819`，但 subject+stimulus 严格留出只有 `0.12664`。

因此，继续堆叠复杂参考网络、层级 Transformer 或对抗正则化的预期收益很低。

## 3. 术语与协议定义

| Field | Definition |
|---|---|
| DE | Differential entropy；保留频带能量信息的基线表征。 |
| JSD | Jensen-Shannon divergence。 |
| sqrt-JSD / RJSD | 对 JSD 开根号后的距离；本项目要求所有 RJSD 流程保留开根号。 |
| signed RJSD | 在距离幅度之外编码相对参考方向；注意部分早期 signed-sqrt 实验同时改变缩放与方向，不能单独归因为方向。 |
| target locked | 目标主体不参与特征拟合、checkpoint 选择或超参数选择。 |
| target monitored | 训练期间每隔若干 epoch 查看目标域；只属于探索诊断。 |
| conventional FACED holdout | 留出目标主体，但源域仍可能包含相同刺激视频。 |
| strict subject+stimulus holdout | 同时留出目标主体和目标刺激。 |
| final score | 最后一个 epoch 的指标。 |
| monitored peak | 在已观察目标曲线中取得的峰值；不得当作独立测试结果。 |

## 4. 正式与完整匹配结果

### 4.1 SEED/SEED-IV 固定协议矩阵

协议：15-fold LOSO，seed 42，固定 80 epoch，目标不参与选择。

| Dataset | Representation | Model | Folds | Macro-F1 mean | Subject SD | Evidence |
|---|---|---:|---:|---:|---:|---|
| SEED | DE-z | Hierarchical Attention | 15 | 0.752433 | 0.051347 | formal |
| SEED | DE-raw | Hierarchical Attention | 15 | 0.752072 | 0.039545 | formal |
| SEED | histogram | Plain Transformer | 15 | 0.738802 | 0.041090 | formal |
| SEED | histogram | Hierarchical Attention | 15 | 0.723750 | 0.026247 | formal |
| SEED | RJSD-z | Hierarchical Attention | 15 | 0.694112 | 0.057922 | formal |
| SEED-IV | histogram | Hierarchical Attention | 15 | 0.856720 | 0.008612 | formal |
| SEED-IV | RJSD-z | Hierarchical Attention | 15 | 0.803783 | 0.034092 | formal |
| SEED-IV | DE-z | Hierarchical Attention | 15 | 0.785488 | 0.068160 | formal |
| SEED-IV | DE-raw | Hierarchical Attention | 15 | 0.770930 | 0.056919 | formal |
| SEED-IV | histogram | Plain Transformer | 15 | 0.704265 | 0.036758 | formal |

Source: `runs/fixed_protocol_seed42/summary.json`, `condition_summary.csv`, `paired_statistics.csv`, and recursive `result.json` files.

### 4.2 FACED 10-fold 匹配表征矩阵

协议：10 个主体 folds、100 epoch、相同 866,953 参数 HCBT、无目标 checkpoint 选择。

| Representation | Folds | Accuracy mean | Macro-F1 mean | Macro-F1 sample SD | Min | Max | Evidence |
|---|---:|---:|---:|---:|---:|---:|---|
| DE | 10 | 0.553631 | 0.554800 | 0.067235 | 0.461819 | 0.645539 | completed_matched |
| native Fisher-Rao PCA | 10 | 0.128988 | 0.113564 | 0.012088 | 0.089371 | 0.129984 | completed_matched |
| native sqrt-JSD | 10 | 0.140179 | 0.047864 | 0.026925 | 0.027778 | 0.092080 | completed_matched |

Source: `runs/faced_native_compact_base_seed42/` recursive `result.json` files and `matrix_manifest.json`.

## 5. 完整实验台账

每条记录都使用固定键，便于 AI 抽取。

### EXP-001

- dataset: `SEED, SEED-IV`
- evidence_level: `formal`
- split: `15-fold LOSO`
- target_usage: `locked; excluded from model and epoch selection`
- experiment: `fixed protocol matrix`
- feature: `histogram, DE-raw, DE-z, RJSD-z`
- model: `logistic regression, linear SVM, small MLP, plain Transformer, Hierarchical Attention`
- training: `seed 42; fixed 80 epochs`
- result: `600/600 tasks complete; SEED best 0.752433; SEED-IV best 0.856720 Macro-F1`
- interpretation: `RJSD is dataset-dependent and not globally superior.`
- source: `runs/fixed_protocol_seed42/`

### EXP-002

- dataset: `SEED, SEED-IV`
- evidence_level: `diagnostic`
- split: `15 target subjects, target monitored`
- target_usage: `monitored`
- experiment: `early HCBT runs`
- feature: `RJSD d128`
- model: `HierarchicalChannelBandTransformer`
- result: `SEED 0.70966; SEED-IV 0.87840 Macro-F1`
- interpretation: `Historical evidence only; cannot be compared as formal target-locked results.`
- source: `runs/seed/`, `runs/seediv/`

### EXP-003

- dataset: `SEED, SEED-IV`
- evidence_level: `diagnostic`
- split: `fold-1 only`
- target_usage: `monitored`
- experiment: `native compact geometry comparison`
- feature: `sqrt-JSD, Fisher-Rao PCA, Wasserstein-1`
- model: `matched compact-feature backbone`
- result: `SEED: 0.59743, 0.72782, 0.60138; SEED-IV: 0.92989, 0.88880, 0.84637`
- interpretation: `Representation ranking reverses across datasets.`
- source: `runs/native_compact_v1_seed42/`

### EXP-004

- dataset: `SEED, SEED-IV`
- evidence_level: `diagnostic`
- split: `fold-1`
- target_usage: `monitored over 200 epochs`
- experiment: `capacity and signed-distance ablation`
- feature: `RJSD, signed sqrt-RJSD, DE-z`
- model: `base and large HCBT variants`
- result: `SEED final: RJSD base 0.66393, signed base 0.75794, RJSD large 0.75682, signed large 0.70562, DE-z large 0.86406; SEED-IV final: 0.94363, 0.90215, 0.85677, 0.93015, 0.93086`
- interpretation: `Increasing capacity and signed sqrt do not produce stable cross-dataset gains.`
- source: `runs/srjsd_large_v1_seed42/condition_summary.csv`

### EXP-005

- dataset: `DEAP`
- evidence_level: `infrastructure`
- split: `source-only fold states`
- target_usage: `not applicable to preprocessing validation`
- experiment: `official BDF preprocessing and cache validation`
- feature: `ICA-cleaned EEG cache, DE/RD prerequisites`
- model: `none`
- result: `32 subjects × 40 trials; official 512 Hz BDF; fixed 60 s video segments; BioSemi32 reorder; strict ICA cache completed and validated`
- interpretation: `Preprocessing infrastructure is usable; this does not prove model efficacy.`
- source: `DEAP preprocessing manifests and validation files in the repository`

### EXP-006

- dataset: `DEAP`
- evidence_level: `incomplete`
- split: `fold-1`
- target_usage: `monitored`
- experiment: `compact sqrt-JSD HCBT`
- feature: `sqrt-JSD`
- model: `HCBT`
- training: `interrupted after epoch 190`
- result: `best monitored target Macro-F1 0.2415 at epoch 180; last 0.18748`
- interpretation: `Diagnostic evidence of weak transfer; run interruption prevents a complete comparison.`
- source: `reports/deap_fold1_diagnostic_data.json`

### EXP-007

- dataset: `DEAP`
- evidence_level: `diagnostic`
- split: `fold-1`
- target_usage: `monitored`
- experiment: `DE baseline`
- feature: `DE`
- model: `matched baseline model`
- training: `200 epochs`
- result: `final accuracy 0.32500; balanced accuracy 0.31754; Macro-F1 0.30175; monitored best Macro-F1 about 0.32377`
- interpretation: `DE is stronger than compact sqrt-JSD on this diagnostic, but overall transfer remains weak.`
- source: `runs/deap_de_baseline_v1_seed42/`

### EXP-008

- dataset: `DEAP`
- evidence_level: `diagnostic`
- split: `fold-1 feature geometry analysis`
- target_usage: `analysis only`
- experiment: `subject-versus-emotion effect decomposition`
- feature: `sqrt-JSD and Fisher-Rao PC1`
- model: `adjusted eta-squared analysis`
- result: `sqrt-JSD subject eta2 ≈0.376 vs emotion ≈0.001; Fisher-Rao PC1 subject ≈0.510 vs emotion ≈0.005; source-video majority labels predict target labels at 0.075`
- interpretation: `Subject heterogeneity is much stronger than emotion separation in compact geometry.`
- source: `reports/deap_fold1_diagnostic_data.json`

### EXP-009

- dataset: `FACED`
- evidence_level: `completed_matched`
- split: `10 subject folds`
- target_usage: `no target model selection`
- experiment: `native compact matched matrix`
- feature: `DE, native sqrt-JSD, native Fisher-Rao PCA`
- model: `866,953-parameter HCBT`
- training: `100 epochs`
- result: `DE 0.554800; Fisher-Rao 0.113564; sqrt-JSD 0.047864 Macro-F1`
- interpretation: `The pipeline learns DE; compact RJSD collapses across folds.`
- source: `runs/faced_native_compact_base_seed42/`

### EXP-010

- dataset: `FACED`
- evidence_level: `diagnostic`
- split: `source-dev smoke`
- target_usage: `target not required for conclusion`
- experiment: `initial PSD-JSD CNN plus Transformer`
- feature: `time × channel × band × native-frequency sqrt-JSD`
- model: `Conv3D plus temporal Transformer, approximately 110.66M parameters`
- training: `50 epochs`
- result: `source-dev Macro-F1 0.027778; one-class prediction collapse`
- interpretation: `The original high-capacity architecture was not learnable under the available sample regime.`
- source: `runs/faced_sqrt_jsd_monitor/` and PSD-JSD notebook outputs

### EXP-011

- dataset: `FACED`
- evidence_level: `diagnostic`
- split: `three-fold source-only inner CV plus one locked outer target fold`
- target_usage: `locked for the final outer evaluation`
- experiment: `flattened channel-by-band repair`
- feature: `flattened structured sqrt-JSD`
- model: `nine-class flattened model`
- result: `inner source-dev Macro-F1 0.12217 ± 0.00342; source final 0.93758; locked target Macro-F1 0.14506`
- interpretation: `Optimization collapse was repaired, but cross-subject generalization remained poor.`
- source: `corresponding faced flattened-channel-band run directories`

### EXP-012

- dataset: `FACED`
- evidence_level: `diagnostic`
- split: `source-only fold-1`
- target_usage: `target not loaded`
- experiment: `legacy Train.py graph backbone`
- feature: `sqrt-JSD`
- model: `GraphBackbone, 23,931,609 parameters`
- result: `source train Macro-F1 1.00000; source-dev Macro-F1 0.15204 at epoch 100`
- interpretation: `Large capacity memorizes source training without robust source-dev generalization.`
- source: `runs/faced_sqrt_jsd_legacy_graph_seed42/source_training_summary.json`

### EXP-013

- dataset: `FACED`
- evidence_level: `diagnostic`
- split: `source-only fold-1`
- target_usage: `target not loaded`
- experiment: `hierarchical FBCT and DE-BCT`
- feature: `sqrt-JSD for FBCT; DE for BCT`
- model: `approximately 15.1M hierarchical Transformers`
- result: `FBCT final source-dev Macro-F1 0.027778; DE-BCT best 0.06869 and final 0.04904 over 10 epochs`
- interpretation: `Band/channel hierarchy and high complexity worsened learnability.`
- source: `FACED hierarchical source-only run directories`

### EXP-014

- dataset: `FACED`
- evidence_level: `diagnostic`
- split: `fold-1`
- target_usage: `monitored`
- experiment: `simple structured DE CNN`
- feature: `DE with structured input`
- model: `37,320-parameter CNN`
- training: `10 epochs`
- result: `target Macro-F1 0.28809; all nine classes predicted`
- interpretation: `A small model is more learnable than the complex hierarchical alternatives.`
- source: `runs/faced_de_simple_structured_cnn_fold1_smoke/`

### EXP-015

- dataset: `FACED`
- evidence_level: `diagnostic`
- split: `fold-1`
- target_usage: `monitored every 10 epochs`
- experiment: `axial no-CLS white-box`
- feature: `DE and RJSD`
- model: `simple axial Transformer without Channel Vote, explicit_channels, or extra regularizers`
- result: `10 epochs: DE 0.31408 vs RJSD 0.05483 Macro-F1; 50-epoch DE final 0.45198, monitored peak 0.49051 at epoch 40`
- interpretation: `The minimal model and training loop work; feature choice dominates.`
- source: `runs/faced_axial_no_cls_whitebox_fold1_diag10/`, `runs/faced_axial_no_cls_whitebox_DE_fold1_50ep/`

### EXP-016

- dataset: `FACED`
- evidence_level: `diagnostic`
- split: `fold-1`
- target_usage: `monitored`
- experiment: `RJSD representation ablation A to E`
- feature: `A=1s unsigned raw; B=4s Welch unsigned; C=1s normalized shape+magnitude; D=4s shape+magnitude; E=4s signed shape+magnitude`
- model: `same axial no-CLS white-box`
- result: `10 epochs A 0.05483, B 0.08661, C 0.09253, D 0.15673, E 0.17865; 50 epochs D 0.18800, E 0.20465 Macro-F1`
- interpretation: `Longer spectral support, magnitude bypass, and direction all help, but the absolute level remains weak.`
- source: `runs/faced_rjsd_shape_magnitude_ablation_fold1_diag10/`, `runs/faced_rjsd_shape_magnitude_DE_fold1_50ep/`

### EXP-017

- dataset: `FACED`
- evidence_level: `diagnostic`
- split: `fold-1`
- target_usage: `monitored`
- experiment: `DE plus signed-RJSD fusion`
- feature: `DE plus E-type signed shape+magnitude RJSD`
- model: `two-branch fusion with DE+DE capacity control`
- training: `50 epochs`
- result: `DE+E 0.31840; DE+DE control 0.35464 Macro-F1`
- interpretation: `Naive feature fusion hurts; added branch capacity is not the solution.`
- source: `runs/faced_de_signed_rjsd_fusion_fold1_50ep/comparison.csv`

### EXP-018

- dataset: `FACED`
- evidence_level: `diagnostic`
- split: `conventional subject holdout and strict subject+stimulus holdout`
- target_usage: `diagnostic probes`
- experiment: `pseudo/normative reference probe`
- feature: `A0 absolute DE; A1 pseudo-relative; A5 shrink pseudo; N1 source-video reference; D1 absolute plus A5`
- model: `linear/probe classifier`
- result: `conventional Macro-F1: 0.20747, 0.29127, 0.24274, 0.31041, 0.31819; strict: 0.10820, 0.08479, 0.07774, 0.11867, 0.12664`
- interpretation: `A large fraction of the apparent gain follows shared stimulus temporal traces.`
- source: `runs/faced_normative_probe_fold1_seed42/condition_metrics.csv`

### EXP-019

- dataset: `FACED`
- evidence_level: `diagnostic`
- split: `conventional and strict subject+stimulus holdout`
- target_usage: `diagnostic`
- experiment: `subject-adversarial training`
- feature: `DE-like emotion features`
- model: `emotion-only baseline, marginal GRL, conditional GRL, dual GRL`
- result: `conventional Macro-F1: baseline 0.34562, marginal 0.33414, conditional 0.32818, dual 0.15687; strict: 0.10058, 0.08170, 0.05627, 0.06804`
- interpretation: `No useful subject-invariance versus emotion-performance Pareto improvement.`
- source: `runs/faced_subject_adversarial_fold1_light_v2_seed42/matrix_results.csv`

### EXP-020

- dataset: `FACED`
- evidence_level: `diagnostic`
- split: `registered 5s and post-hoc 1s; fold-1`
- target_usage: `diagnostic`
- experiment: `STR-JSD C0/C4/C5/C6`
- feature: `absolute DE; DE+delta+signed pointwise; log ratio; quality gate`
- model: `matched light classifier`
- result: `5s conventional Macro-F1 C0 0.25445, C4 0.26520, C5 0.21752, C6 0.25305; strict C0 0.10030, C4 0.11806, C5 0.071996, C6 0.10414; 1s strict post-hoc C5 0.14812`
- interpretation: `No reliable registered improvement; quality/FCCA did not help. The 1s C5 signal is post-hoc.`
- source: `runs/faced_str_jsd_fold1_light_seed42/matrix_results.csv`

### EXP-021

- dataset: `FACED`
- evidence_level: `diagnostic`
- split: `fold-1`
- target_usage: `monitored`
- experiment: `vector-preserving compact representations`
- feature: `frequency-vector RJSD and full-vector Fisher-Rao`
- model: `deep model plus linear probe`
- result: `deep Macro-F1 0.12579 and 0.12786; target linear probe about 0.16497 and 0.16145`
- interpretation: `Retaining a longer vector alone is insufficient; source-to-target mismatch remains large.`
- source: `runs/faced_vector_preserving_monitor_seed42/condition_summary.csv`

### EXP-022

- dataset: `FACED`
- evidence_level: `diagnostic`
- split: `fold-1`
- target_usage: `monitored`
- experiment: `relative supervised compression`
- feature: `frequency-point RJSD and supervised Fisher-Rao LDA2`
- model: `deep classifier and linear probe`
- result: `deep target Macro-F1 0.12723 and 0.06557; best linear Fisher-Rao probe 0.18108`
- interpretation: `Supervised low-dimensional compression did not repair transfer.`
- source: `runs/faced_relative_supervised_monitor_seed42/`

### EXP-023

- dataset: `FACED`
- evidence_level: `diagnostic`
- split: `fold-1 oracle analysis`
- target_usage: `target subject reference oracle; monitored`
- experiment: `subject-reference oracle`
- feature: `global, K-shot, shrinkage, and class-balanced references`
- model: `same 64,188-parameter E-type model for deep comparison`
- result: `global 0.22851; subject K27 0.24975; shrink64 0.23074; class-balanced label-assisted oracle 0.26610 Macro-F1`
- interpretation: `Removing subject identity helps only modestly; the oracle ceiling is too low to justify a complex amortized reference network.`
- source: `runs/faced_subject_reference_oracles_fold1/`

## 6. 被证伪、暂停或降级的方向

```yaml
directions:
  - name: "Channel Vote and explicit_channels"
    decision: "removed"
    reason: "No observed benefit; also creates an avoidable information bottleneck."
  - name: "bias term"
    decision: "omit"
    reason: "No observed effect in current exploration."
  - name: "quality/FCCA"
    decision: "pause"
    reason: "No observed benefit in the registered comparisons."
  - name: "deeper or wider hierarchical Transformers"
    decision: "stop scaling"
    reason: "Repeated collapse or source memorization without target improvement."
  - name: "subject-adversarial GRL"
    decision: "pause"
    reason: "Did not reduce subject information while preserving emotion performance."
  - name: "long-vector RJSD or Fisher-Rao alone"
    decision: "not sufficient"
    reason: "Vector preservation did not solve cross-subject transfer."
  - name: "naive DE plus RJSD concatenation"
    decision: "reject"
    reason: "Worse than DE+DE capacity control and the simple DE branch."
  - name: "complex subject-reference estimator"
    decision: "do not build yet"
    reason: "Unlabeled oracle improvement is only about 2.1 Macro-F1 points."
```

## 7. 中期因果图（诊断性，不是统计因果证明）

```text
short/noisy spectral support
        +
scalar RJSD removes magnitude
        +
unsigned distance removes direction
        |
        v
weak emotion representation and class collapse
        |
        +---- longer support + magnitude + sign partially repair it
        |
        v
strong remaining subject/stimulus domain mismatch
        |
        +---- complex hierarchy: source memorization or optimization collapse
        +---- GRL: no useful Pareto improvement
        +---- subject reference: small oracle ceiling
        +---- naive fusion: capacity increase without generalization
```

## 8. 下一阶段最小实验方案

### 8.1 目标

直接回答：“当前深模型为什么低于同任务 SVM，以及是特征、样本组织还是模型归纳偏置造成的？”

### 8.2 固定比较矩阵

```yaml
dataset: "FACED"
development_scope: "3 folds x 3 seeds"
full_scope_if_promoted: "10 folds"
features:
  - "DE"
  - "full histogram or log-PSD"
  - "E-type 4s signed shape+magnitude RJSD"
models:
  - "Linear SVM"
  - "RBF-SVM"
  - "Logistic Regression"
  - "2-layer MLP"
  - "small axial no-CLS model"
target_monitoring: "allowed for exploration, but final comparisons must label it explicitly"
required_controls:
  - "same folds"
  - "same windows and labels"
  - "same source normalization"
  - "DE+DE capacity control for any fusion"
  - "conventional and strict subject+stimulus results kept separate"
health_checks:
  - "overfit 18-32 samples"
  - "source-dev Macro-F1 does not collapse"
  - "prediction covers all 9 classes"
```

### 8.3 建议停止线

```yaml
stop_rules:
  rjsd_model_optimization:
    condition: "E-type RJSD trails DE by >= 0.10 Macro-F1 in the matched 3x3 matrix"
    action: "Stop model-level optimization and switch to representation analysis."
  subject_reference_network:
    condition: "Unlabeled oracle does not stably exceed 0.30 Macro-F1"
    action: "Do not build an amortized reference network."
  complex_hierarchy:
    condition: "Does not beat small axial/CNN under matched budget"
    action: "Do not increase depth or width."
  adversarial_learning:
    condition: "Subject probe does not clearly decrease without emotion Macro-F1 loss"
    action: "Remove GRL branch."
  strict_faced_generalization:
    condition: "Strict subject+stimulus results remain near chance"
    action: "Do not claim unseen-stimulus generalization from current processed FACED data."
```

## 9. 局限和未完成项

- 多数 FACED 机制实验为 fold-1、每 10 epoch 查看目标域的探索，不能用于独立泛化估计。
- DEAP sqrt-JSD 训练被中断；DEAP Fisher-Rao 正式对照未完成。
- 不同实验的窗口长度、模型参数量和 epoch 不完全一致时，只能进行机制层面的定性比较。
- 现有 FACED processed data 没有真正的刺激前 baseline，无法严格证明主体相对基线机制。
- FACED 常规主体留出可能共享相同刺激视频，因此不能自动解释为新刺激泛化。
- signed sqrt 的部分早期实验同时改变方向与数值缩放，不能单独识别方向效应。

## 10. 可供另一 AI 继续执行的任务

```yaml
recommended_next_tasks:
  - priority: 1
    task: "Implement and run the matched FACED SVM/Logistic/MLP/small-axial matrix."
    expected_output: "Per-fold and per-seed Macro-F1, balanced accuracy, confusion matrices, class coverage, and runtime."
  - priority: 2
    task: "Add full histogram/log-PSD as an energy-preserving representation."
    expected_output: "Matched comparison against DE and E-type RJSD."
  - priority: 3
    task: "Audit exact sample/window counts and label distributions used by SVM and neural models."
    expected_output: "A single reconciliation table proving identical inputs and splits."
  - priority: 4
    task: "If fusion is retried, use residual or late fusion with a DE+DE capacity control."
    expected_output: "Matched parameter counts and paired fold deltas."
  - priority: 5
    task: "Quantify performance separately for conventional and strict subject+stimulus holdout."
    expected_output: "Protocol-specific results without cross-protocol ranking."
```

## 11. Final Project State

```yaml
project_state: "midterm_blocked_by_representation_and_domain_mismatch"
training_pipeline_state: "operational for DE"
compact_rjsd_state: "not competitive on FACED; dataset-dependent on SEED/SEED-IV"
fisher_rao_state: "not competitive on FACED and incomplete on DEAP"
complex_model_state: "not justified"
subject_reference_state: "oracle ceiling too low"
primary_recommended_baseline: "minimal DE model"
primary_next_question: "Why does the exact matched SVM/linear family outperform or differ from the neural models?"
claim_not_supported: "RJSD is generally superior to DE for cross-subject EEG emotion recognition."
claim_supported: "RJSD effectiveness is dataset-dependent, and aggressive spectral-distance compression can remove transferable emotion information."
```
