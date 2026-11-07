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

## 严格协议推荐配置（当前主线 = 10s_nogro 数据线 + 通道拆分成员）

### 数据线迁移（2026-11-05）：10s_nogro（GRO 关闭重提取）

数据重提取为 packet 级事件（max size 1250=MTU，~3.5× 事件数，video_cdn>4096
的会话 0.81%→16.6%），同 644 视频、strict s42 视频级划分不变（386 known /
129 val 冒充 / 129 test 冒充），`splits/splits_10s_nogro_strict_s42.json`。
全通道成员线（lgbm v2 + e100/dt 全5通道 video:8192/api:768 + aux4）的
LOVO 最优 geo **.15/.55/0/.30**：test top1 .944 / @0.5% .797 / @1% .836 /
@5% .893（`runs/combo_10s_nogro_L8k_strict/`）——比旧 GRO 线 geo4（.654）
高 +14pp，是本推荐配置的对照基线。

### 推荐系统：通道拆分 geo4（2026-11-07）

**成员按通道完全拆分**（通道轴去同质化，T2-A2 损失头负结果的后续方向）：
**LGBM + DF vid（纯 video_cdn,8192）+ DF vid_dt（纯视频+dt）+ DF aux3
（api+thumbnail+channel_info，去掉 other）**，geo 权重 **.15/.15/.30/.40**
（池化 LOVO 选出），`runs/combo_10s_nogro_L8k_chsplit4/`：

| 严格协议 test（10s_nogro） | top1 | @0.5% | @1% | @5% | AUROC |
|---|---|---|---|---|---|
| **通道拆分 geo4（推荐）** | .943 | **.823** | .850 | .900 | .947 |
| 全通道基线 geo（.15/.55/0/.30） | .944 | .797 | .836 | .893 | .947 |
| 拆分3成员（无dt，.20/.40/.40） | .939 | .801 | .837 | .893 | .943 |
| （旧 10s GRO 线）geo4 T2-A1 | .975 | .654 | .729 | .880 | .964 |

单模型（test）：vid .881/.446@0.5%、vid_dt .893/.340、aux3 .639/.094
（辅助通道对 DF 单模型贡献 @0.5% ~30pp——拆分是拿单模型强度换成员独立性）。

**确认口径（如实记录，2026-11-07）**：三档 oracle 点估计全面更优
（+2.6/+1.4/+0.7pp）但配对视频簇 bootstrap CI 含 0（@0.5%
[−.020,+.099]）；10000 次自助 **P(dTPR>0) = .84/.79/.79**。经所有者裁决
确认提升并采纳为推荐系统——该裁决**推翻了"CI 排 0 才可确认"的既有门**
（findings §3.6c 留档）；一致性提醒：同量级的 L8k(8192) vs 4096（+2.3pp、
CI 含 0）按同一裁决标准亦应记为提升。部署口径：@0.5% realized FPR 0.53%、
TPR .823（对照基线 0.41%/.766；realized FPR 略松）。

**机制**：aux3 与两个 DF 的 val 冒充者 argmax 一致率 **0.3–0.4%**（旧 aux4
7.4–7.6%，随机 0.26%）——通道拆分把成员去同质化做到随机水平；正确 probe 上
aux3-vs-vid 仍 71% 同票（真身份不伤投票）。**dt 门通过**：dt(纯视频) 在
LOVO top-15 全部配置权重 ≥0.1，对 3 成员系统的边际 @1% CI **[+.002,+.027]
排 0**。vid 与 vid_dt 互相仍 64.6% 一致（同通道必然），但乘积门仍吃到 dt 的
@1% 增量。叠加方向（6 成员两段式 0.25→0.05 初筛/精筛，规则见
`runs/combo_10s_nogro_L8k_chadd6/SCREEN_RULE.md`）判负关闭：初筛自动把旧
dt/aux4/vid 压零收敛到 [lgbm,e100,aux3]，精筛 @0.5% .772 < 基线。

### 通道拆分复现命令

```bash
# 三个专家（预算与全通道线一致：100ep/min15/pat12/batch64/seed42）
python scripts/run_df.py --dataset 10s_nogro --protocol strict --fusion branch \
    --channels video_cdn --chan-len video_cdn:8192 --epochs 100 \
    --tag ngL_dfs_branch_vid
python scripts/run_df.py --dataset 10s_nogro --protocol strict --fusion branch \
    --channels video_cdn --chan-len video_cdn:8192 --epochs 100 --dt \
    --tag ngL_dfs_branch_vid_dt
python scripts/run_df.py --dataset 10s_nogro --protocol strict --fusion branch \
    --channels api,channel_info,thumbnail \
    --chan-len api:768,thumbnail:384,channel_info:192 --epochs 100 \
    --tag ngL_dfs_branch_aux3
python scripts/run_combiner.py --dataset 10s_nogro --protocol strict \
    --df-tags ngL_dfs_branch_vid,ngL_dfs_branch_vid_dt,ngL_dfs_branch_aux3 \
    --lovo --families plain --out combo_10s_nogro_L8k_chsplit4
# 验收：配对簇 bootstrap vs 全通道基线 + 单侧概率
python scripts/run_paired_boot.py --dataset 10s_nogro \
    --a-tags ngL_dfs_branch_vid,ngL_dfs_branch_vid_dt,ngL_dfs_branch_aux3 \
    --a-w 0.15,0.15,0.3,0.4 --b-tags ngL_dfs_branch_all_e100,ngL_dfs_branch_all_dt,ngL_dfs_branch_aux \
    --b-w 0.15,0.55,0.0,0.3 --out ng_boot_chsplit4_vs_base
python scripts/run_osided_boot.py
```

## 旧 10s（GRO）严格线（存档，2026-11-01）

部署假设"未知视频完全未见过"（644 → 386 known / 129 val 冒充 / 129 test 冒充，
视频级 6:2:2）下的当时推荐 = **geo 几何平均 4 成员**（LGBM + DF branch e100 +
DF branch dt + **DF branch aux**——aux 只看辅助通道，是针对 video_cdn 共线
磁铁的独立反对票），权重 **.35/.10/.20/.35**（池化 LOVO 选出，aux 与 lgbm
并列最大），`runs/combo_10s_strict_t2a1/` + `runs/t2_aux_eval/`：

| 严格协议 test（10s） | top1 | @0.5% | @1% | @5% | AUROC |
|---|---|---|---|---|---|
| **geo4 加权（推荐，T2-A1）** | .975 | **.654** | **.729** | **.880** | **.964** |
| geo3 加权（T2-A1 前基线，.30/.35/.35） | .973 | .551–.561 | .625 | .819 | .946 |
| arith 等权（保守替代） | .954 | .577 | .651 | .753 | .929 |
| gcfisher_a（旧协议最优，此协议下崩塌） | .973 | .218 | .384 | .722 | .920 |

验收（升级判据 = 配对视频簇 bootstrap，dTPR(geo4−geo3)）：@0.5%
**[+.050,+.151]**、@1% **[+.043,+.143]**、@5% **[+.039,+.083]**，三档全部
排 0——**严格协议下 0.5% 档首个统计可判的正提升**。机制：val 磁铁事件上 aux
中位置置信 .500（其余成员 .77–.98），geo 乘积在"辅助通道不认账"的会话上自动
塌陷；两个旧 DF 在冒充者上 argmax 互认 52%（同质化），aux 仅 7.4%/7.6%；
val 冒充者 >.9 的事件 19 条（geo3）→ 3 条（geo4）。部署口径（val 定阈）
realized FPR@0.5%：geo4 **1.12%**（geo3 1.30%），TPR .738（.656）。成员
单独很弱（aux top1 .654）但乘积组合不看单独强弱。

T3 组合器三子项已按预注册门关闭（findings §3.6b）：幂均值 q 族最优 q=−0.25
仅 val 赢、test 配对簇 CI 含 0（geo 已在 AND 严格度最优平台）；LOVO 交叉
拟合 stacking 三档全输纯 geo4 分数（分歧特征无增量信息）；GPD 尾外推不如
经验分位贴名义（偏差源于 129 只标定视频的群体位移，非尾部估计问题）——
**部署阈值用经验分位，双门限表并列报告 realized FPR**。

多 seed 稳健性（T2 复验，`runs/ms_agg/`）：DF 成员以 model seed 43/44/45
在同一 s42 划分上重训（lgbm/划分固定），每 seed 用该 seed 自己的最优
三成员 plain 规则做公平对照——**4/4 seed × 三档共 12/12 配对簇 CI 排 0**，
aux 权重跨 seed 稳定 .35–.40，geo4 @0.5% 四 seed 均值 **.662**（.644–.681）。
ArcFace 头成员被独立门击杀（与 softmax DF 错误一致率 51.9%≈DF 互认 52%，
损失头不是去同质化轴）。

历史注记（T0，`runs/strict_diag/`）：geo3 vs arith 在 0.5%/1% 档统计不可分
（仅 @5% geo 显著）；test top-19 冒充者仅来自 4 只视频。上表 gcfisher_a 的
类维 ECDF 定价被证明绑定标定池视频身份；P1 收缩修复封顶于总体定价
（LOVO .656 < geo3 .696），负结果关闭——实验链见 findings.md §3.5 第六/七轮。
**此后开集配置选择一律用池化 LOVO**（留未知视频 5 折，其排名与 test 同向）。

## 单模型最优配置（DF branch；旧 10s GRO 线参数）

（nogro 主线的差异：video_cdn 长度 8192、api 768、aux3 thumbnail 384 /
channel_info 192；见上文通道拆分小节。）

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

## 复现命令（旧 10s GRO 线全流程；nogro 主线见上文通道拆分命令）

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
    --channels video_cdn,api,channel_info,thumbnail,other --epochs 100 --dt \
    --tag df_branch_all_dt --load
# 6) 严格协议 4 成员（推荐）：aux 专家成员 + 池化 LOVO 权重重搜
python scripts/run_df.py --dataset 10s --protocol strict --fusion branch \
    --channels api,channel_info,thumbnail,other --epochs 100 \
    --tag dfs_branch_aux
python scripts/run_combiner.py --dataset 10s --protocol strict \
    --df-tags dfs_branch_all_e100,dfs_branch_all_dt,dfs_branch_aux \
    --lovo --families plain --out combo_10s_strict_t2a1
# 7) 验收（独立门 + 磁铁否决力 + 配对簇 bootstrap vs 3 成员基线）
python scripts/run_t2_aux_eval.py --w4 0.35,0.10,0.20,0.35
```

## 环境

- Windows 10 / RTX 3060 Ti 8GB / Python 3.11（Anaconda）
- `requirements.txt`：torch 2.5.1+cu121、lightgbm 4.5.0、scikit-learn 1.8.0、numpy 1.26.4

## 权重与报告索引

- **当前推荐（通道拆分）**：`runs/ngL_dfs_branch_{vid,vid_dt,aux3}/model.pt`
  + `runs/combo_10s_nogro_L8k_chsplit4/report.json`；对照全通道线
  `runs/ngL_dfs_branch_{all_e100,all_dt,aux}/model.pt` +
  `runs/combo_10s_nogro_L8k_strict/report.json`；验收
  `runs/ng_boot_{chsplit4_vs_base,chsplit3_vs_base,chsplit4_vs_3,chadd6_vs_base}/`、
  `runs/t2_chsplit_eval/`、两段式规则 `runs/combo_10s_nogro_L8k_chadd6/SCREEN_RULE.md`
- `runs/df_branch_all_e100/model.pt`、`runs/df_branch_all_dt/model.pt`、
  `runs/df5_branch_all_e100/model.pt`：旧 GRO 线最优权重（含 args/hist 元数据）
- `runs/{df_*,ens_*}/report.json`：全部配置的完整指标；
  `runs/combo_{10s,5s}/report.json`：组合器全网格结果（旧协议最优 gcfisher_a）
- `runs/{tag}/probs_{val,test}.npz`：概率矩阵；`runs/{tag}/logits_*.npz`：logits
  （A2 打分族用）；`runs/{tag}_tta/`：TTA 概率（已证伪，留作对照）——
  均为可重建产物，不入库

## 已知边界与后续方向

- **主动反相关联合训练已判负收格（2026-11-07，findings §3.6e）**：对冻结
  锚 vid_dt 显式训练"B 错得不一样"——直接惩罚项被类别抑制捷径吸收
  （61–64% 不动），乘积项训到 6–8% 一致率但成员塌成纯否决器，修复膝点
  (.70,.29) LOVO 压零、换 vid 槽 test @0.5% −4pp。谱系收格：主动反相关
  也买不过信息差异，推荐系统不变；
- **成员替换四轴已系统排除（2026-11-07，findings §3.6d）**：架构替换
  （ResNet-1D/TCN/ViT 全面弱于 DF——粗池化漏斗是与干扰结构匹配的归纳偏置，
  且架构轴不去同质化：resnet↔DF 冒充者 argmax 一致 52%）、LGBM 替换
  （ExtraTrees/RandomForest 单模型大幅更强 +4~6pp top1/+11~22pp @0.5% 但
  独立性差 28–31%，组合被 LOVO 压零——强度换独立性）、预算加倍（200ep
  双双更差）、时间域表示（top1 .327，正交信息不可恢复）。**结论：通道拆分
  geo4 是当前数据的局部最优平台，提升方向在新信息源（通道语义/观察窗）或
  数据侧，不在成员替换**；
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
