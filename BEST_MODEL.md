# 最优配置与训练方法（里程碑固化）

日期：2026-11-01　协议：`splits/splits_*_s42.json`（515 known / 129 unknown，
known 会话 60/20/20，gallery_k=3，详见 findings.md §五）

## 最终系统

**类维 ECDF 组合（gcfisher）**——成员不变（LGBM + DF + DF+dt 三路），组合器升级：
argmax/top1 取加权几何平均（闭集最优），开集分 = 逐类 ECDF Fisher
（每个"成员给类 c 的概率"对照"val 冒充者给类 c 的概率分布"转 p 值，
S_c = −Σ wᵢ·ln pᵢ,ᶜ，得分 = max_c S_c；超参照天花板按 logit 指数尾延伸；
val 冒充者自身用 LOO p 值）。权重在验证集上按 @0.5%FPR 网格搜索选出。
机制与两个伪影坑见 findings.md §3.5 第三轮。

| 系统（test，10s） | top1 | @0.5% | @1% | @5% | AUROC |
|---|---|---|---|---|---|
| LGBM segv2 120 维 | .698 | .157 | .200 | .412 | .749 |
| DF branch 全5（df_branch_all_e100） | .915 | .155 | .249 | .554 | .883 |
| DF branch 全5+dt（df_branch_all_dt） | .934 | .063 | .169 | .573 | .892 |
| mean(prob) 三路集成 | .948 | .640 | .704 | .781 | .936 |
| 几何平均加权（.5/.3/.2） | .962 | .669 | .717 | .821 | .949 |
| **gcfisher 加权（.2/.35/.45，val 选出）** | **.969** | **.817** | **.840** | **.912** | **.983** |

5s 版（LGBM + DF 两路 gcfisher，.25/.75）：top1 .920 / @0.5% .580 / @1% .650 /
@5% .786（几何平均基线 .497/.587/.732）。

## 单模型最优配置（DF branch）

- **输入**：每语义通道一条 packet-index 序列（通道内包序号），左对齐右侧补零；
  值 = `sign(size)·log1p(|size|)/log(65536)`（带符号大小，GRO 合并原始值）；
  通道长度 L：video_cdn 4096 / api 512 / thumbnail 256 / channel_info 128 / other 64。
- **dt 变体**（df_branch_all_dt）：每事件加第 2 特征 `log1p(dt_ms)/log1p(1000)`，
  dt 为该通道内相邻事件间隔。
- **架构**（`src/dfnet.py: DFNet(fusion='branch')`，改自 DF/CCS'18）：
  每通道独立 trunk = 4 块 × [Conv1d(k=8, pad=4) → BN → 激活]×2（块1 ELU，
  其余 ReLU）→ MaxPool1d(4,4,ceil) → Dropout(0.1)，宽度 32/64/128/256；
  flatten 拼接（video 17 slots + api 3 + thumb 2 + ci 1 + other 1）×256 = 6400 维
  → Dense512-BN-ReLU-Drop0.7 → Dense512-BN-ReLU-Drop0.5 → Dense(515) softmax。
  参数量 ~8.3M。
- **训练**：CE + label_smoothing=0.05；AdamW lr 1e-3 wd 1e-4；OneCycle
  (pct_start 0.08)；batch 64；最多 100 epochs，val_probe top1 选最优，
  patience 12 / min_epochs 15；AMP fp16。单卡 3060 Ti 约 7–8 分钟。
- **增强**（train only，`DFData._augment`，即第 1–2 步诊断出的干扰因素）：
  1. 起始时域裁剪：p=.5 取 t0~U(0,500ms)，丢弃 t<t0 的事件（跨通道一致）；
  2. 尾部截断：p=.25，保留 t ≤ t_last·U(0.65,1.0)；
  3. 质量全局缩放：video 下行值 += N(0,0.4)/ln(65536)（≈ 字节乘 exp(N(0,0.4))），
     其他通道 N(0,0.15)；
  4. 通道丢弃：非 video 通道 p=.15 整通道置空；
  5. 值噪声 σ=0.02。
- **推理/打分**：softmax 概率向量；闭集 argmax；开集得分 = max prob，
  θ 由 unknown impostor 池分位数标定（`src/evaluate.py`）。

## 复现命令

```bash
# 0) 数据在 E:\evb\datasets（项目外），首次运行自动建 runs/cache/df_series_*.npz
# 1) 划分（已有则跳过）
python scripts/make_splits.py
# 2) LGBM 成员
python scripts/run_lgbm_v2.py 10s
# 3) DF 两个成员（训练并保存权重 runs/<tag>/model.pt）
python scripts/run_df.py --dataset 10s --fusion branch \
    --channels video_cdn,api,channel_info,thumbnail,other --epochs 100 \
    --tag df_branch_all_e100
python scripts/run_df.py --dataset 10s --fusion branch \
    --channels video_cdn,api,channel_info,thumbnail,other --epochs 100 --dt \
    --tag df_branch_all_dt
# 4) 组合器搜索（规则×权重×温度网格 + 类维 ECDF 规则，验证集选模，test 仅报告）
#    当前最优 gcfisher 即由该脚本选出（runs/combo_10s/report.json）
python scripts/run_combiner.py --dataset 10s --df-tags df_branch_all_e100,df_branch_all_dt
# 算术平均基线（可选，run_ensemble 为均值集成入口）
python scripts/run_ensemble.py --dataset 10s --df-tag df_branch_all_e100,df_branch_all_dt
# 5s 版
python scripts/run_df.py --dataset 5s --fusion branch \
    --channels video_cdn,api,channel_info,thumbnail,other --epochs 100 \
    --tag df5_branch_all_e100
python scripts/run_combiner.py --dataset 5s --df-tags df5_branch_all_e100
# 纯推理评估（不训练，读 runs/<tag>/model.pt）
python scripts/run_df.py --dataset 10s --fusion branch \
    --channels video_cdn,api,channel_info,thumbnail,other --dt \
    --tag df_branch_all_dt --load
```

## 环境

- Windows 10 / RTX 3060 Ti 8GB / Python 3.11（Anaconda）
- `requirements.txt`：torch 2.5.1+cu121、lightgbm 4.5.0、scikit-learn 1.8.0、numpy 1.26.4

## 权重与报告索引

- `runs/df_branch_all_e100/model.pt`、`runs/df_branch_all_dt/model.pt`、
  `runs/df5_branch_all_e100/model.pt`：最优权重（含 args/hist 元数据）
- `runs/{df_*,ens_*}/report.json`：全部配置的完整指标；
  `runs/combo_{10s,5s}/report.json`：组合器全网格结果（当前最优 gcfisher）
- `runs/{tag}/probs_{val,test}.npz`：概率矩阵（供新集成实验，免重训）

## 已知边界与后续方向

- 全部结果基于单划分 seed=42；多 seed 均值±方差与协议敏感性未做（下一步首选）；
- 100ep 时 val 仍缓升，容量/时长还有余量（video 8192、更宽 trunk）；
- 开集组合器三轮演进：均值 .640 → 几何平均 .669 → 类维 ECDF .817（@0.5%，10s）；
  温度校准为负结果；后续方向见 findings.md §3.6（A2 打分族 / A1 TTA / A3 gallery
  余弦成员 / C 贪心成员选择 / ② 扩员）；
- 数据为单设备单网络环境；跨设备/跨网络泛化未验证。
