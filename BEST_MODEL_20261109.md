# 最优配置变更说明与复现指引（2026-11-09）

本文回答两个问题：**(1) 相对上一次提交 GitHub 的版本（origin/main = `cb59ec9`，
2026-11-02 plan_a 定稿时点），当前最优配置改了什么、为什么有效（本质原理）；
(2) 从那个版本出发，逐步复现当前版本的完整操作链**。

版本坐标系（`git log origin/main..HEAD`，本地领先 7 个提交，均未推送）：

| 提交 | 内容 | 对推荐系统的影响 |
|---|---|---|
| `1a40e52` | nogro 数据线迁移 + L8k 严格基线 | 对照基线立表（.797） |
| `e6f84d1` | **通道拆分专家按所有者裁决确认为推荐** | **推荐系统更新为通道拆分 geo4** |
| `dac1f11`/`936de25`/`dae25ff`/`ccd57e9` | §3.6c–3.6e 负结果收格 | 只加边界注记，推荐不变 |
| `55e7646` | 补漏报告入库 | 无 |

即：**上一版推荐 = 旧 10s（GRO）严格线 geo4 T2-A1**（`c1dc56c` 定型、
`e5b2b70` 多 seed 复验，BEST_MODEL.md 当时所载）；**当前推荐 = 10s_nogro
通道拆分 geo4**（`e6f84d1` 定型，BEST_MODEL.md 现行内容）。

---

## 一、变更总览（旧 → 新）

| 维度 | 上一版（cb59ec9，GRO 线 geo4 T2-A1） | 当前版（e6f84d1，通道拆分 geo4） |
|---|---|---|
| 数据线 | `10s`（采集 GRO 开启，事件被 NIC 合并粗化） | **`10s_nogro`（GRO 关闭重提取，packet 级，MTU 1250 切分，~3.5× 事件）** |
| 划分文件 | `splits_10s_strict_s42.json` | `splits_10s_nogro_strict_s42.json`（同 644 视频、同 s42，视频级划分逐只不变） |
| 通道长度 L | video 4096 / api 512 / thumb 256 / ci 128 | **video 8192 / api 768 / thumb 384 / ci 192** |
| 成员 2 | DF 全 5 通道 e100（video+api+ci+thumb+other） | **DF vid：纯 video_cdn(8192)** |
| 成员 3 | DF 全 5 通道 dt | **DF vid_dt：纯 video_cdn + dt（包间隔第 2 特征）** |
| 成员 4 | DF aux4（api+ci+thumb+**other**） | **DF aux3（api+thumb+ci，去 other）** |
| geo 权重 | .35 / .10 / .20 / .35（lgbm/e100/dt/aux） | **.15 / .15 / .30 / .40（lgbm/vid/vid_dt/aux3）** |
| test @0.5%/1%/5% | .654 / .729 / .880 | **.823 / .850 / .900**（+16.9 / +12.1 / +2.0pp） |
| test top1 | .975 | .943（数据线迁移 .975→.944，拆分再 −0.1pp；闭集换开集，见 §二注） |
| 部署 realized FPR@0.5% | 1.12%（TPR .738） | **0.53%（TPR .823）**——更贴名义 0.5% |

变更可拆成两步，证据链（10s_nogro strict s42，test oracle）：

```
GRO 线 geo4          .654 ──Step A 数据线迁移(nogro+L8k, 成员形态不变)──▶ 全通道 geo .797
全通道 geo（对照）    .797 ──Step B 成员通道拆分(vid/vid_dt/aux3)────────▶ 通道拆分 geo4 .823
```

- **Step A（`1a40e52`）**：@0.5% **+14.3pp**，成员仍是全通道四成员——提升全部来自
  数据侧，与模型无关。LOVO 副产物：全通道线下 dt 权重归零（.15/.55/**0**/.30），
  因为 dt 与 e100 同通道、错误同质。
- **Step B（`e6f84d1`）**：@0.5% **+2.6pp 点估**（.797→.823），配对视频簇
  bootstrap CI 含 0（[−.020,+.099]），10000 次自助单侧 **P(dTPR>0)=.84/.79/.79**，
  2026-11-07 所有者裁决确认提升并采纳（该裁决显式推翻"CI 排 0"门，findings
  §3.6c 留档；同量级的 L8k-vs-4096（+2.3pp、CI 含 0）按同一标准亦应改记提升）。
  dt 复活：拆分后 vid_dt 拿 .30 权重，对 3 成员系统的边际 @1% CI
  **[+.002,+.027] 排 0**（dt 门通过）。

> 注：闭集 top1 随数据线迁移下降 ~3pp（.975→.944，未单独归因），拆分再 −0.1pp
> （容忍内）。项目主指标是开集严档（@0.5%FPR），本变更以微小闭集代价换大幅
> 开集尾部提升，且部署 realized FPR 从 1.12% 收紧到 0.53%。

---

## 二、为什么有效（本质原理）

### 2.0 前提：开集尾部指标的"物理学"

TPR@0.5%FPR 的判定线由 ~9 个得分最高的冒充者（"磁铁"）决定（findings §3.4）。
多成员组合能赢过单模型的**唯一机理是否决**：成员 A 被某条未知流量骗出高置信、
成员 B 没被骗，几何平均（geo，即概率域的 AND 门）就把这个错误摁下去。
**推论：若成员犯的是同一批错（同质化），组合退化为单成员，多算的全是白算。**
错误同质化的度量 = val 冒充者上的 argmax 一致率（随机基线 ≈0.26%）。

### 2.1 Step A 为什么有效：找回了被观测层毁掉的信息

身份信号的物理基础（findings §二）：载荷大小/突发序列是编码内容的确定性函数
叠加上传输调度噪声——同视频跨播放的字节结构稳定（η²、质量守恒）。

**GRO（Generic Receive Offload）在网卡/驱动层把多个包合并成大 blob 再交给
抓包端**，等于在观测层面对 packet 大小序列做了一次与内容无关的粗化平滑：
总字节守恒（质量守恒），但细粒度的大小结构——身份信号所在——被合并掉。
关闭 GRO 重提取（packet 级、MTU 1250 切分）后：

- 事件数 ~3.5×，video_cdn>4096 包的会话 0.81%→16.6%（序列变长是信息回来的
  直接证据）；
- 通道长度必须随之升级到 8192，否则截断丢信息。截断诊断
  （`scripts/run_ng_trunc_diag.py`）：磁铁事件 0% 超 4096、全体冒充者 17.9% 超
  → **L8k 救的是普通尾部的 recall，不是磁铁**（L8k vs 4096 点估 +2.3pp、
  CI 含 0，按新裁决标准应记提升）。

本质：**Step A 的 +14pp 不是模型变强，是被采集管线毁掉的信息被找回来了**
——同一成员架构、同一训练配方，只换观测粒度。这也是跨平台迁移文档
（platform_transferability.md）把"粒度审计"列为 pilot 必做项的原因。

### 2.2 Step B 为什么有效：错误独立性只能从信息源轴买

本项目用三条独立证据钉死了同一结论——**成员错误的同质化由信息源决定，
不由架构/损失头/训练方式决定**（openset_error_complementarity.md）：

1. 同通道换架构：ResNet-1D 与 DF-vid 的冒充者 argmax 一致率 **52.2%** ≈
   两个 DF 互认的 52%（随机 0.26%）——换"脑子"不换"教材"，错误照旧相关；
2. 同通道换损失头：ArcFace 成员与 softmax DF 一致率 51.9%，双门击杀（T2-A2）；
3. 主动反相关联合训练：显式训练"B 错得不一样"全灭（§3.6e）——错误结构买不来。

通道拆分把成员的去同质化做到**随机水平**（`runs/t2_chsplit_eval/report.json`
实测）：

| 冒充者 argmax 一致率 | 数值 | 含义 |
|---|---|---|
| aux3 vs vid / vid_dt / lgbm | **0.24–0.38%** | ≈ 随机 0.26%——aux3 被骗时几乎从不骗向同一个已知视频 |
| （旧 aux4 vs DF） | 7.4–7.6% | other 混流噪声通道引入错误相关 → **去 other 是拆分的必要部分** |
| vid ↔ vid_dt | 64.6% | 同通道必然同质，但 dt 的边际信息仍过门（@1% CI 排 0） |
| aux3 vs vid **正确 probe** 同票 | 71.4% | 真身份不伤投票——AND 门只杀假事件 |

于是 geo 乘积获得理想特性：**假事件必被至少一个独立成员低置信压塌，真事件
不被误伤**。两个配套事实：

- **以单模型强度换独立性**：辅助通道对 DF 单模型贡献 @0.5% ~30pp（全通道
  e100 .797 vs 拆分后 vid .446 / aux3 .094），但乘积组合不看单独强弱、只看
  否决力——LOVO 给最弱的 aux3 定了最大权重 **.40**（旧全通道线下 lgbm 只拿
  .15，e100 拿 .55：组合器在为"强度"定价；拆分后转而为"独立性"定价）；
- **dt 复活**：全通道线 dt 与 e100 同质被压零；拆分后 vid_dt 的包间隔结构
  信息不再被淹没，边际 @1% CI [+.002,+.027] 排 0。

### 2.3 统计确认口径（如实记录）

Step B 三档点估全面更优（+2.6/+1.4/+0.7pp）但配对簇 CI 含 0——不掩盖：
10000 次自助 P(dTPR>0)=.84/.79/.79，所有者裁决"提升很大，确认提升"采纳。
此为先例：后续候选若"点估 +2~3pp / CI 含 0"按本先例执行；L8k 同量级结论
应同步改记。

---

## 三、Step-by-step 复现（从 cb59ec9 版本 → 当前版本）

### 3.0 前置条件

- **代码**：checkout 本仓库当前 main（含 `1a40e52`/`e6f84d1` 新增能力：
  `run_df.py --chan-len`、`run_combiner.py --lgbm-suffix/--lovo`、
  `run_osided_boot.py`）。cb59ec9 的代码缺这些入口，直接跑会失败。
- **环境**：同 BEST_MODEL.md（Windows / RTX 3060 Ti 8GB / Python 3.11；
  torch 2.5.1+cu121、lightgbm 4.5.0、scikit-learn 1.8.0、numpy 1.26.4）。
- **数据**：需要 `E:\evb\datasets\600classes_30samples_10s_no_gro\dataset`
  （**GRO 关闭重提取的数据集，提取发生在采集侧管线，不在本仓库**；packet 级
  事件、max size 1250=MTU；视频清单与 10s 线相同的 644 视频）。若只有旧
  `10s` 数据集，先用原采集框架重放采集（GRO off），无法用代码转换得到。
- **可复用 / 不可复用**：旧 GRO 线全部模型、概率缓存、组合报告**不可复用**
  （数据线变了，全部重训）；划分逻辑可复用（同 seed 同视频清单 → 视频级
  6:2:2 划分逐只相同）；旧 runs 保留作历史对照。
- **预算**：1×LGBM（分钟级）+ 6×DF 训练（旧 4096 线单成员 7–8 分钟，
  8192 更长；3060 Ti 合计约 1 小时量级）+ 组合器与验收（CPU 分钟级）。

### 3.1 Step 0：划分

```bash
# 数据存在性检查（应能看到各视频的会话目录）
ls "E:\evb\datasets\600classes_30samples_10s_no_gro\dataset" | head
# 生成（或跳过）nogro 严格划分
python scripts/make_splits.py --protocol strict --dataset 10s_nogro
```

**验收**：输出 `[wrote] splits/splits_10s_nogro_strict_s42.json`（已存在则
`[skip]`）；split_summary 显示 386 known / 129 val 冒充 / 129 test 冒充
（视频级，与 10s strict 线同视频集合）。

### 3.2 Step 1：LGBM 成员（segv2 段指纹，120 维）

```bash
python scripts/run_lgbm_v2.py --dataset 10s_nogro --protocol strict
```

**产出**：`runs/10s_nogro_lgbmv2_strict/` +
`runs/cache/{segv2_10s_nogro.npy, lgbm_probs_10s_nogro_strict.npz}`
（组合器按 dataset+protocol 自动取该缓存，无需 `--lgbm-suffix`）。

### 3.3 Step 2：对照基线三成员（全通道；Step 6 配对验收需要它们）

chan-len 与 `runs/ngL_dfs_branch_*/report.json` 的 args 逐字段一致
（**全通道线只显式指定 video/api 两个长度**，其余用默认）：

```bash
python scripts/run_df.py --dataset 10s_nogro --protocol strict --fusion branch \
    --channels video_cdn,api,channel_info,thumbnail,other \
    --chan-len video_cdn:8192,api:768 --epochs 100 --tag ngL_dfs_branch_all_e100
python scripts/run_df.py --dataset 10s_nogro --protocol strict --fusion branch \
    --channels video_cdn,api,channel_info,thumbnail,other \
    --chan-len video_cdn:8192,api:768 --epochs 100 --dt --tag ngL_dfs_branch_all_dt
python scripts/run_df.py --dataset 10s_nogro --protocol strict --fusion branch \
    --channels api,channel_info,thumbnail,other \
    --chan-len api:768,thumbnail:384,channel_info:192 --epochs 100 \
    --tag ngL_dfs_branch_aux
```

**单模型验收**（test top1，±1pp 容差）：all_e100 ≈**.931**、all_dt ≈**.931**、
aux ≈**.670**（对照旧 GRO 线 .915/.934/.654；注意组合基线的 .944 是四成员
geo 的成绩，勿与单模型混淆）。

### 3.4 Step 3：推荐成员（通道拆分三成员）

```bash
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
```

**单模型验收**（test，±1pp 容差；这是"以强度换独立性"的直接证据）：

| 成员 | top1 | @0.5% |
|---|---|---|
| vid | .881 | .446 |
| vid_dt | .893 | .340 |
| aux3 | .639 | .094 |

（其余训练超参全部用默认：batch 64 / lr 1e-3 / patience 12 / min_epochs 15 /
seed 42 / AMP，增强五项内置，勿改。）

### 3.5 Step 4：组合器（两条 LOVO，池化留未知视频 5 折选权）

```bash
# 对照基线
python scripts/run_combiner.py --dataset 10s_nogro --protocol strict \
    --df-tags ngL_dfs_branch_all_e100,ngL_dfs_branch_all_dt,ngL_dfs_branch_aux \
    --lovo --families plain --out combo_10s_nogro_L8k_strict
# 推荐系统
python scripts/run_combiner.py --dataset 10s_nogro --protocol strict \
    --df-tags ngL_dfs_branch_vid,ngL_dfs_branch_vid_dt,ngL_dfs_branch_aux3 \
    --lovo --families plain --out combo_10s_nogro_L8k_chsplit4
```

**验收（`report.json` 的 `finals[0]`）**：

| 项 | 基线 `combo_..._strict` | 推荐 `combo_..._chsplit4` |
|---|---|---|
| LOVO 最优 config | `geo\|T=False\|w=(0.15,0.55,0.0,0.3)` | `geo\|T=False\|w=(0.15,0.15,0.3,0.4)` |
| val top1/@0.5%/@1%/@5% | .9524/.7937/.8302/.8999 | .9503/.8406/.8585/.9130 |
| test top1/@0.5%/@1%/@5% | .9436/.7972/.8365/.8929 | .9428/**.8226**/.8496/.9002 |
| AUROC | .9475 | .9468 |

训练含 AMP/cuDNN 非确定性，逐位一致不保证；**关键判定 = LOVO 收敛到同一
权重形态（dt 在基线压零、aux3 在拆分线拿最大 .40）+ 两线 @0.5% 差 ≈+2.5pp**。

### 3.6 Step 5：统计验收

```bash
# 配对视频簇 bootstrap（CI 如实记录；期望三档含 0，@0.5% ≈[-.020,+.099]）
python scripts/run_paired_boot.py --dataset 10s_nogro \
    --a-tags ngL_dfs_branch_vid,ngL_dfs_branch_vid_dt,ngL_dfs_branch_aux3 \
    --a-w 0.15,0.15,0.3,0.4 \
    --b-tags ngL_dfs_branch_all_e100,ngL_dfs_branch_all_dt,ngL_dfs_branch_aux \
    --b-w 0.15,0.55,0.0,0.3 --out ng_boot_chsplit4_vs_base
# 单侧改善概率（裁决依据；期望 P(dTPR>0)=.84/.79/.79，10000 次）
python scripts/run_osided_boot.py
```

（`run_osided_boot.py` 为硬编码的 A=chsplit4 / B=全通道基线对照，权重与
tags 内置，直接运行即可。）

### 3.7 Step 6：机制验证（可选但强烈推荐——确认"独立性"真的买到）

```bash
python scripts/run_t2_aux_eval.py --dataset 10s_nogro \
    --w4 0.15,0.15,0.3,0.4 --w3 0.2,0.4,0.0,0.4 \
    --tags ngL_dfs_branch_vid,ngL_dfs_branch_vid_dt,ngL_dfs_branch_aux3 \
    --out t2_chsplit_eval
```

**验收**：`gate_independence_pass: true`；冒充者 argmax 一致率
aux_vs_e100 ≈.0034 / aux_vs_dt ≈.0038 / aux_vs_lgbm ≈.0024（杀线 >10%）、
e100_vs_dt（=vid↔vid_dt）≈.646、正确 probe 同票 ≈.71。

### 3.8 Step 7：部署口径

阈值用**经验分位**（GPD 尾外推已判负），交付**双门限表**并列 realized FPR：
@0.5% 名义线 val 定阈后 test realized FPR **0.53%** / TPR .823（基线
0.41%/.766）。推理提速线可选：lgbm fast50（统计非劣、2.0ms、25×）+ DF
CUDA Graph（12–15×），端到端 83.8ms→~11.5ms（model_speed_optim.md）。

---

## 四、未改变项与已判负方向（防走回头路）

**未变**：组合规则仍是 geo（AND 严格度最优平台，gcfisher/类维 ECDF 在严格
协议不迁移）；温度=1；阈值=经验分位；五项增强与全部训练超参；seed42。

**已在预注册门下判负（勿重试）**：成员替换四轴——架构（ResNet/TCN/ViT 全面
弱于 DF 且不去同质化 52.2%）、树算法（ET/RF 单模型强但独立性 28–31%，LOVO
压零）、预算加倍（200ep 双输）、时间域表示（.327）；主动反相关联合训练两设计
（§3.6e 全灭）；扩员（8 成员 mega8、6 成员两段式、timebins 5 成员，全部被
LOVO 压零）。**成员准入真实门槛 = 独立性 ∧ 置信方向判别力**；通道拆分 geo4
是当前数据的局部最优平台，提升方向在新信息源/数据侧。

**边界**：全部结果基于单划分 seed=42（nogro 线多 seed 复验未做，旧 GRO 线
aux 结论曾 4/4 seed 复现是间接稳健性证据）；单设备单网络环境；跨平台迁移
理论分析见 platform_transferability.md。

## 五、产物索引（当前版本）

- 推荐权重：`runs/ngL_dfs_branch_{vid,vid_dt,aux3}/model.pt` +
  `runs/combo_10s_nogro_L8k_chsplit4/report.json`
- 对照基线：`runs/ngL_dfs_branch_{all_e100,all_dt,aux}/model.pt` +
  `runs/combo_10s_nogro_L8k_strict/report.json`
- 验收：`runs/ng_boot_{chsplit4_vs_base,chsplit3_vs_base,chsplit4_vs_3,chadd6_vs_base}/`、
  `runs/t2_chsplit_eval/`
- 上游文档：BEST_MODEL.md（现行推荐全文）、findings.md §3.6c–3.6e（证据链）、
  openset_error_complementarity.md（机制）、model_speed_optim.md（部署提速）
