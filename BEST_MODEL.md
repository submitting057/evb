# 最优配置与训练方法（里程碑固化）

日期：2026-11-01　协议：`splits/splits_*_s42.json`（515 known / 129 unknown，
known 会话 60/20/20，gallery_k=3，详见 findings.md §五）

## 最终系统

**gcfisher_a = 类维 ECDF Fisher + 标量打分族证据**——成员不变（LGBM + DF +
DF+dt 三路），纯后处理组合器：

1. argmax/top1 取加权几何平均概率（闭集最优）；
2. 开集分·类维部分：每个"成员给类 c 的概率"对照"val 冒充者给类 c 的概率分布"
   转 p 值，S_c = −Σ wᵢ·ln pᵢ,ᶜ（超参照天花板按 logit 指数尾延伸；val 冒充者
   自身用 LOO p 值）；
3. 开集分·标量部分（A2）：每成员 val 选出的置信度统计量（lgbm→margin、
   e100→负熵、dt→energy）同样转 ECDF p 值；
4. 总分 = max_c S_c + α·Σ wᵢ·(−ln pᵢ,scalar)，α=.3，权重 .3/.3/.4（验证集
   @0.5%FPR 网格选出）。机制与四轮演进见 findings.md §3.5。

| 系统（test，10s） | top1 | @0.5% | @1% | @5% | AUROC |
|---|---|---|---|---|---|
| LGBM segv2 120 维 | .698 | .157 | .200 | .412 | .749 |
| DF branch 全5（df_branch_all_e100） | .915 | .155 | .249 | .554 | .883 |
| DF branch 全5+dt（df_branch_all_dt） | .934 | .063 | .169 | .573 | .892 |
| mean(prob) 三路集成 | .948 | .640 | .704 | .781 | .936 |
| 几何平均加权（.5/.3/.2） | .962 | .669 | .717 | .821 | .949 |
| gcfisher 加权（.2/.35/.45） | .969 | .817 | .840 | .912 | .983 |
| **gcfisher_a（α=.3，w=.3/.3/.4）** | **.974** | **.861** | **.884** | **.933** | **.986** |

5s 版（LGBM + DF 两路，gcfisher .25/.75；A2 附加证据无增益）：top1 .920 /
@0.5% .580 / @1% .650 / @5% .786（几何平均基线 .497/.587/.732）。

## 严格协议（视频不交叠开集，`splits_10s_strict_s42.json`）推荐配置

部署假设"未知视频完全未见过"（644 → 386 known / 129 val 冒充 / 129 test 冒充，
视频级 6:2:2）下的推荐系统 = **geo 几何平均加权**，成员为 386 类严格重训版
（LGBM + DF branch e100 + DF branch dt，w≈.3/.3/.4，`runs/combo_10s_strict*`）：

| 严格协议 test（10s） | top1 | @0.5% | @1% | @5% | AUROC |
|---|---|---|---|---|---|
| **geo 加权（推荐）** | .973 | **.551–.561** | .625 | .819 | **.946** |
| arith 等权（保守替代） | .954 | .577 | .651 | .753 | .929 |
| gcfisher_a（旧协议最优，此协议下崩塌） | .973 | .218 | .384 | .722 | .920 |

注（T0 测量地基，`runs/strict_diag/`）：视频簇 bootstrap 下 geo @0.5% CI
[.432,.697]；配对 dTPR(geo−arith) @0.5%/@1% **含 0（两规则统计不可分）**，
仅 @5% [+.040,+.088] geo 显著占优——推荐 geo 的依据是 @5% 与 AUROC，0.5% 档
两规则打平。test top-19 冒充者事件仅来自 4 只视频（剔掉最大磁铁 +10pp），
0.5% 档对未知视频组成高度敏感；部署口径（val 定阈）realized FPR：geo 1.30%、
arith 0.67%。

上表 gcfisher_a 的类维 ECDF 定价被证明绑定标定池视频身份；P1 收缩修复
（κ 谱 + 池化 LOVO，`run_combiner.py --lovo`）单调有效但封顶于总体定价
（LOVO .656 < geo .696），负结果关闭——实验链见 findings.md §3.5 第六/七轮。
**此后开集配置选择一律用池化 LOVO**（留未知视频 5 折，其排名与 test 同向）。

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
# 4) 导出 logits（A2 打分族需要；同时产出 TTA 概率，供消融）
python scripts/run_tta_dump.py --dataset 10s --tags df_branch_all_e100,df_branch_all_dt
# 5) 组合器搜索（规则×权重×温度×α 网格 + 类维 ECDF + 标量证据，验证集选模）
#    当前最优 gcfisher_a 即由该脚本选出（runs/combo_10s/report.json）
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
  `runs/combo_{10s,5s}/report.json`：组合器全网格结果（当前最优 gcfisher_a）
- `runs/{tag}/probs_{val,test}.npz`：概率矩阵；`runs/{tag}/logits_*.npz`：logits
  （A2 打分族用）；`runs/{tag}_tta/`：TTA 概率（已证伪，留作对照）——
  均为可重建产物，不入库

## 已知边界与后续方向

- 全部结果基于单划分 seed=42；多 seed 均值±方差与协议敏感性未做（下一步首选）；
- **协议边界（重要）**：上表 .861 基于"验证/测试冒充者共享 129 个未知视频"
  的会话级划分（评测惯例口径）。视频不交叠严格协议（29/100 未知视频）下
  gcfisher_a 不迁移（@0.5% .386），稳健水平为 arith .643 / geo .617——
  见 findings §3.5 第六轮；修复方向为跨身份伪冒充者标尺；
- 100ep 时 val 仍缓升，容量/时长还有余量（video 8192、更宽 trunk）；
- 开集组合器五轮演进：均值 .640 → 几何平均 .669 → 类维 ECDF .817 → +标量证据
  .861（@0.5%，10s）→ gallery 余弦成员中性（成员级更强但与同源 softmax 误差
  相关）；温度、标量 p 空间、TTA 均为负结果；后续方向见 findings.md §3.6
  （C 贪心成员选择 / ② 扩员 / 多 seed 稳健性——5s 的 val→test 选择方差已大于
  加成员边际收益，多 seed 上调为第一优先）；
- 数据为单设备单网络环境；跨设备/跨网络泛化未验证。
