# 模型推理速度优化（10s_nogro 严格开集协议）

> 结论先行：**LGBM 专家确认是单样本推理的阻塞点**（579k 棵树 = 51.0 ms，占总延迟 61%）。
> 将 LGBM 训练轮数从 1500 轮封顶到 **50 轮**（19,300 棵树，预测 2.0 ms，25× 提速），
> 组合开集性能统计等价（配对簇 bootstrap 三档 CI 全含 0，LOVO 复核 test @0.5% .796 vs 参考 .797）。
> 配合 DF 成员 CUDA Graph 回放（12–15×），端到端单样本延迟 **83.8 ms → ~11.5 ms（7.3×）**。

---

## 1. 背景与目标

- 数据集：`600classes_30samples_10s_no_gro`（GRO 重切分，MTU 级包），严格开集协议
  （644 视频 6:2:2，386 已知 / 129+129 未知视频，s42）。
- 此前最优开集组合（L8k 基线）：geo4 w=(.15,.55,0,.3)，
  test top1 .944 / TPR@0.5% **.797** / @1% .836 / @5% .893 / AUROC .948。
- 多专家结构中 LGBM 成员的作用（保留前提）：
  1. 树模型不易过拟合，概率平缓，在 geo 软 AND 中抑制深度成员的过度自信（磁铁否决）；
  2. 与 DF CNN 表示差异大 → 错误模式互补（val 冒充者 argmax 一致率仅 ~10%，DF 成员间 ~52–94%）。
- 目标：优化推理速度；允许替换/改造 LGBM，但**开集性能损失 ≤ 1.5pp**（@0.5% 档），
  有提升更好。

## 2. 推理基准（单样本，batch=1，RTX 3060 Ti）

测量脚本 `scripts/run_infer_bench.py` → `runs/infer_bench/report.json`。
部署视图 = 一次 load_events + DF 序列编码 + segv2 特征 + 各成员前向 + geo 合并（合并开销可忽略）。

| 组件 | 延迟 | 备注 |
|---|---|---|
| load_events（npz 冷读 + 逐事件通道映射） | ~7.1 ms | OS 缓存热读后 ~1 ms；中位数 2,777 事件/会话 |
| DF 序列编码（5 通道切分+logsize 编码） | 0.16 ms | DFData init 摊销 |
| segment_features_v2（LGBM 特征，120 维） | ~0.5 ms | 摊销后增量（不含 load_events） |
| DF 前向 e100_L（GPU, branch, video_cdn:8192） | 12.3 ms | **kernel 启动开销主导**：batch=64 时仅 0.18 ms/样本 |
| DF 前向 aux_L（GPU, 4 通道） | 13.3 ms | 同上（b64 0.16 ms/样本）；CPU b1 参考 62.8 ms |
| **LGBM 预测（单行，579,000 棵树）** | **51.0 ms** | **占 3 成员部署总延迟的 61%** |
| **合计（lgbm + e100_L + aux_L）** | **83.8 ms** | dt 权重 0，部署不含 |

LGBM 瓶颈的机理：`run_lgbm_v2` 的 `best_iteration=1500`（早停 patience=100 从未触发），
多分类每轮生成 386 棵树 → 1500×386 = **579,000 棵**；LightGBM 单行预测必须遍历所有树。
树数-延迟严格线性（~33 µs/千树）：

| num_iteration | 树数 | 单行预测 |
|---:|---:|---:|
| 25 | 9,650 | 0.50 ms |
| 50 | 19,300 | 1.47 ms |
| 100 | 38,600 | 3.14 ms |
| 200 | 77,200 | 6.37 ms |
| 400 | 154,400 | 14.59 ms |
| 800 | 308,800 | 26.68 ms |
| 1500 | 579,000 | 51.0 ms |

（副本复训与缓存概率逐位一致 max Δprob = 0，确定性成立；批量预测 256 行为 10.7 ms/行，
单样本部署口径取单行为准。）

## 3. 优化路线一（采纳）：LGBM 轮数上限

### 3.1 单成员饱和点

`scripts/run_lgbm_fast.py`（同参数、无早停、精确树数上限）→
`runs/10s_nogro_lgbm_fast/report.json`：

| 轮数 | 单行预测 | val top1 / @0.5% | test top1 / @0.5% |
|---:|---:|---|---|
| 50 | 2.03 ms | .653 / .139 | .650 / .155 |
| 100 | 3.86 ms | .659 / .144 | .661 / .169 |
| 200 | 7.29 ms | .662 / .166 | .669 / .164 |
| 400 | 14.64 ms | .663 / .164 | .667 / .172 |
| 1500（参考） | 51.0 ms | — | .667 / .159 |

**开集相关行为（argmax + max-prob）在 ~200 轮饱和**；后 1300 轮只锐化尾部概率
（multi_logloss 仍在降），不改变 geo 软 AND 所需的行为。

### 3.2 组合级验证（三道工序）

1. **直评（plain geo 家族，val 选权 step .05，与参考同法）**：

| LGBM | val 最优权 | val@.5% | test top1 / @0.5% / @1% / @5% |
|---|---|---:|---|
| 1500 轮（参考） | (.15,.55,0,.30) | .794 | .944 / .797 / .836 / .893 |
| fast50 | (.30,.45,0,.25) | .799 | .944 / **.796** / .837 / .890 |
| fast100 | (.25,.40,0,.35) | .798 | .942 / .791 / .832 / .883 |
| fast200 | (.15,.65,0,.20) | .797 | .942 / **.798** / .850 / .894 |
| fast400 | (.15,.65,0,.20) | .796 | .942 / .798 / .853 / .894 |

   注意 fast50 的最优 lgbm 权重升到 .30：更少轮数概率更平，软 AND 中的话语权相应提高。

2. **配对视频簇 bootstrap**（2000 次，`runs/ng_boot_fast{50,200}_vs_ref/`）：
   - fast50 w=(.30,.45,0,.25) vs 参考：dTPR CI @0.5% [-0.049,+0.023]、@1% [-0.037,+0.019]、
     @5% [-0.002,+0.011]，全含 0；**部署口径 @0.5% 反而更好（TPR .779 vs .766，
     realized FPR 0.39% vs 0.41%）**。
   - fast200 w=(.15,.65,0,.20) vs 参考：三档 CI 全含 0（@1% 点值 +1.4pp）。

3. **LOVO combiner 正式复核**（与参考完全相同的选法，
   `runs/combo_10s_nogro_L8k_fast50/report.json`）：
   - LOVO 聚合榜首即 plain geo w=(.30,.45,0,.25)（与直评一致）→
     **test top1 .944 / @0.5% .796 / @1% .837 / @5% .890 / AUROC .946**；
   - LOVO top-3 配置 test @0.5% 落在 .796–.813（其中 w=(.25,.50,.05,.20) 含 dt=.05
     达 **.813**，反超参考 +1.6pp，多配置点值仅供记录，非预注册选择）。

**1.5pp 门以巨大余量通过（实际 -0.1pp，在抽样噪声内）。**

## 4. 优化路线二（废弃）：MLP 蒸馏学生

`scripts/run_lgbm_distill.py`：教师 = 579k 树 LGBM，学生 = 120→512→512→386 MLP
（KL T=2 + 0.3 CE，200 轮）。GPU 单行前向 0.23 ms，但 **val top1 仅 .150（教师 .667）**：
7,549 行 × 120 维的表格数据上，小 MLP 学不出 386 类结构——树的逐特征切分归纳偏置
不可迁移，属表示能力/样本量问题而非蒸馏超参问题。该路线关闭；
若未来需要 <1 ms 的表格专家，应从"更快的小树集成"（即路线一）而非深度学生入手。

## 5. 额外收获：DF 成员 CUDA Graph（12–15×）

DF 成员 batch=1 的 12–15 ms 几乎全是 kernel 启动开销（b64 摊销仅 0.18 ms/样本）。
对固定 batch=1 形状手动捕获 CUDA Graph（`torch.cuda.CUDAGraph`，预热 side-stream 后捕获，
推理时 `copy_` 输入 + `replay()`）：

| 成员 | eager b1 | graph b1 | 数值 |
|---|---:|---:|---|
| e100_L | 15.37 ms | **1.25 ms** | max Δprob = 0 |
| aux_L | 14.92 ms | **0.96 ms** | max Δprob = 0 |

（Windows 无 triton，`torch.compile` 不可用；手动捕获等效。）
**注意：此为基准台架验证，尚未集成进推理服务**；静态形状约束要求固定
chan_len（L8k 恰好满足：video_cdn 8192 / api 768 等）。

## 6. 新最优基线（本文档冻结）

- **成员**：LGBM-fast50（`--rounds 50`，19,300 树）+ DF e100_L + DF aux_L（dt 权 0，不部署）
- **组合**：plain geo，w = **(.30 lgbm, .45 e100_L, 0 dt, .25 aux_L)**（LOVO 榜首）
- **test（oracle）**：top1 **.944** / TPR@0.5% **.796** / @1% **.837** / @5% **.890** / AUROC **.946**
- **test（部署，val 校准阈值）**：@0.5% TPR **.779**（realized FPR 0.39%）/ @1% .815 / @5% .883
- **单样本延迟预算**：

| 项 | 参考 | 新基线 |
|---|---:|---:|
| 特征提取 | 7.3 ms | 7.3 ms（热读 ~1.5 ms） |
| DF 前向 ×2 | 25.6 ms | 2.2 ms（CUDA Graph，待集成） |
| LGBM | 51.0 ms | **2.0 ms** |
| **合计** | **83.8 ms** | **~11.5 ms（7.3×）**；不含图优化也有 34.8 ms（2.4×） |

- 保守备选：fast200（77k 树，7.3 ms），test @0.5% .798 / @1% .850；直评+配对 bootstrap 均等价，
  LOVO 复核报告落地于 `runs/combo_10s_nogro_L8k_fast200/`。

## 7. 复现命令

```bash
# 1) 推理基准（含 579k 树 LGBM 确定性复训，~10 min）
python scripts/run_infer_bench.py --dataset 10s_nogro

# 2) 快速 LGBM 变体（训练 + probs 缓存）
python scripts/run_lgbm_fast.py --dataset 10s_nogro --rounds 50,100,200,400

# 3) 组合器 LOVO（新基线）
python scripts/run_combiner.py --dataset 10s_nogro --protocol strict \
    --df-tags ngL_dfs_branch_all_e100,ngL_dfs_branch_all_dt,ngL_dfs_branch_aux \
    --lovo --lgbm-suffix _fast50 --out combo_10s_nogro_L8k_fast50

# 4) 配对簇 bootstrap（fast50 vs 参考）
python scripts/run_paired_boot.py --dataset 10s_nogro \
    --a-tags ngL_dfs_branch_all_e100,ngL_dfs_branch_all_dt,ngL_dfs_branch_aux \
    --a-w 0.15,0.55,0.0,0.3 \
    --b-tags ngL_dfs_branch_all_e100,ngL_dfs_branch_all_dt,ngL_dfs_branch_aux \
    --b-w 0.30,0.45,0.0,0.25 --b-lgbm-suffix _fast50 --out ng_boot_fast50_vs_ref
```

## 8. 注意事项与边界

1. **单 seed 结论**（s42 划分 + 模型 seed 42）；沿袭 T2 的做法，如需固化为交付线应补
   seed 43/44/45 复验（历史经验：成员轴结论 4/4 可复现）。
2. CUDA Graph 数字来自台架（固定 batch=1、静态 chan_len 形状、数值与 eager 逐位一致），
   服务化集成（输入池、图缓存生命周期）是后续工程项。
3. batch 批量口径下 DF 本身摊销极薄（0.18 ms/样本），LGBM 批量 10.7 ms/行仍偏慢——
   批处理场景同样建议 fast50（批量预测也随之减少 30×树遍历）。
4. 特征提取的 7.3 ms 中 ~7 ms 是 npz 冷读 + 逐事件 Python 通道映射；热读后 ~1.5 ms。
   如需进一步压测，可向量化通道映射或缓存会话级特征，属工程优化，未动。
5. fast50 与参考的组合差异（-0.1pp @0.5%）在配对簇 bootstrap 噪声内；
   LOVO top-3 的 .813 点值是多重配置报告，不应单独引用为"提升"。
