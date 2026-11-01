# plan_a.md —— 性能优化执行方案（2026-11-02 修订版 r2）

本版根据 [审查意见](E:/evb/df/plan_a.review.md) 修订，替代 2026-11-01 版的执行规则。
保留 A1–A3、B1–B6、C1–C2 编号，新增必须先完成的 P0 工程与数据准备。

**当前状态：方案已修订，新增接口尚未实现，尚未执行本轮训练。** 本文所有带新参数或新脚本的命令均为接口规格；完成 P0、通过检查后才可运行。不得把这些示例当成当前代码已经支持的命令。

本轮目标是：在严格协议下提高低误报工作点的识别率，并评估短前缀级联的观察预算收益。当前数据和历史测试池已经被多轮探索使用，本轮结果称为“历史基准确认”；需要新的独立测试池，才能补充新的盲测证据。扩大未知身份与环境覆盖另行采集，不阻塞本轮开发，也不以当前结果替代跨环境验证。

## 〇、全局规则与数据使用边界

1. **已知和未知的 test 会话均不用于训练、早停、权重／规则／阈值选择或决定后续实验分支。** 训练只读 known/train，checkpoint 选择只读 known/val；开发集组合选择读 known/val probe 与 unknown/val。
2. **本轮采用“验证集 ROC 选择”，不称为完整 LOVO 或嵌套交叉验证。** 当前组合器 plain 分支在全部验证样本上选配置；本版承认其开发选择性质，限制候选数量，并将最终确认移至方案冻结后。精确定义见附录 A-1。
3. **正提升要求 `dTPR=新−旧` 的区间下界 `CI_low>0`。** `CI_high<0` 为退步；CI 含 0 为证据不足。禁止将负区间或“任一档显著”解释成主指标改善。
4. **主终点固定为 TPR@0.5%FPR。** @1%、@5%、闭集 top1、部署实际 FPR 为次要指标及退化检查；不得事后改主终点。所有百分点均指绝对差值。
5. **开发阶段可筛选、关闭或组合候选；最终阶段冻结一个主精度候选后只做一次确认。** 未通过时保留原系统，记录结果，本轮不再从同一测试集挑第二名或触发 B6。下一轮若继续使用该测试池，只能作为探索性延续报告。
6. **开发命令不得加载、导出或打印新候选的 test 概率。** 现有训练／组合脚本默认会评估 test，必须先加评估范围控制。冻结旧基线的回归重放是唯一例外：只校验历史数字，不产生新候选测试反馈。
7. **正确保留旧行为，同时显式修复 bug。** 新结构／增强默认关闭；已有 `side=none` checkpoint 的推理应保持一致。side 特征错配属于修复，不要求继续复现错误输入的旧结果。回归比较模型输出和指标，不比较包含运行参数的整个 JSON 字节内容。
8. **新成员和替换成员都按最终组合增量判断。** 单模型 top1、argmax 一致率、联合高置信率用于诊断，不设“top1 不升／一致率超过 10% 就永久淘汰”的硬门。
9. **候选、基线和产物均使用独立名称。** 本轮产物进入 `runs/plan_a_r2/` 或 `runs/pa2_*/`；不得覆盖现有模型、原始 split、概率缓存和历史报告。缓存必须校验样本及类别顺序、源数据与协议指纹。
10. 本轮修订预先设定：精度主候选的闭集 top1 点估计最多下降 0.5pp，@1%／@5% 的配对区间下界须高于 −1pp；级联召回非劣界为 1pp。这些是本轮工程容忍度，不是既有实验发现；如需变更，必须在候选确认前记录新版本。

## 一、基线与比较对象

历史 10s strict 使用 `splits/splits_10s_strict_s42.json`：386 个已知、129 个验证未知、129 个测试未知视频。known/test probe 实际覆盖 385 个 VID；其他预算／匹配子集的簇数必须从数据计算，不能写死。

| 系统与口径 | test top1 | TPR@0.5% | @1% | @5% |
|---|---:|---:|---:|---:|
| 冻结 geo4，权重 .35/.10/.20/.35 | .975403 | .653944 | .729432 | .880407 |
| 冻结 geo3，权重 .30/.35/.35/0 | .972858 | .551315 | .625106 | .819338 |

以上为 oracle ROC 工作点，不能当成验证阈值迁移后的部署成绩。geo4 验证侧按 0.5% 定阈，历史测试实际 FPR 为 1.1151%，TPR 为 73.7913%。

geo4 成员顺序固定为：LGBM、e100、dt、aux；产物分别来自 `runs/cache/lgbm_probs_10s_strict.npz`、`runs/dfs_branch_all_e100/`、`runs/dfs_branch_all_dt/`、`runs/dfs_branch_aux/`。
原文所谓“池化 LOVO .773/.696”在 plain 路径实际对应验证池 ROC 值，精确值为 geo4 .773399、geo3 .696223；只作旧流程回归锚点，不是折外泛化成绩。

本轮精度候选始终与冻结 geo4 对比。5s 的 aux 机制问题则比较“5s 自己选出的四成员系统”与“5s 自己选出的最佳三成员系统”；5s/10s 预算比较必须使用相同 capture 子集，并在该子集重算基线。不得要求子集指标等于上表全量指标。

## 二、执行顺序与停止条件

| 批次 | 内容 | 前置条件 | 完成标志 |
|---|---|---|---|
| 0 | P0：修索引／维数、缓存身份、开发评估隔离、统一预算划分 | 无 | 检查全部通过，旧 geo4 回归正常 |
| 1 | side 修复后复测；A1：5s strict；C1：固定风险档位 | P0 | 仅开发结果与冻结配置格式可用 |
| 2 | B1 容量对照、B2 aux、A2 长度、A3 增强 | P0；不依赖 A1 完成 | 各候选完成组合层开发比较 |
| 3 | B3 通道、B4 表示、B5 embedding 分类器对照 | P0；可接批次 2 | 诊断与开发增量表完整 |
| 4 | B6 条件性新骨干；B1/B2 联合替换 | 仅由开发结果触发 | 候选清单关闭，不再追加网格 |
| 5 | 多训练 seed 复验；C2 开发阶段级联 | 精度候选初选；C2 另需 A1 | 输出最终 freeze 清单 |
| 6 | 冻结后一次性历史基准确认 | 清单、阈值、比较数 K、检验已固定 | 所有预定结果完整报告，结束本轮 |

不再承诺“20 行改完”或固定半天完成。先用一个模型记录峰值显存、训练耗时和评估开销，再估计批次预算。有限候选失败只说明本轮未找到升级，不能证明某个表示或骨干方向已经耗尽。

## 三、P0：必须先完成的工程与数据准备

### P0-1. 段特征对齐与缓存身份

- 修复 [训练与推理入口](E:/evb/df/scripts/run_df.py) 中 `seg_std[idx]` 的池内／全局索引错配：先用 `seg_rows[pool]` 构造该池的段特征表，再按池内 idx 读取；或者统一返回全局 idx，两种方式只选一种并贯穿训练／推理。
- 只使用 known/train 计算段特征均值和标准差，随 checkpoint 保存并在加载时复用。逐池断言段特征 session_id 与原始序列 session_id 一致。
- 原始序列、段特征、embedding、概率缓存保存 `session_ids`、`class_ids`、源 manifest 指纹、split 指纹、表示版本和模型指纹。仅行数相同不允许复用。
- 旧概率文件缺少 ID 时，只能在固定历史 split 下显式导入，记录来源及标签检查；不能为其伪造已验证的来源。新 split／新特征必须建立新缓存键。
- 图像式 padding 不是实际事件。新增强需要的有效长度／mask 必须由真实事件数生成，不能以编码值是否为 0 反推。

**验收**：对 train、val、test 各抽样，逐 ID 对照独立重算的段特征；人为打乱 ID 应被拒绝。该检查检验映射完整性，不用 test 指标调参。

### P0-2. 宽度、长度与 checkpoint 配置

- `DFNet` flatten 维数使用实际末层宽度 `widths[-1]`，去掉固定 256 的假设。长度推导必须与真实卷积／池化输出相符。
- `--chan-len` 同时作用于 `DFData` 和 `DFNet`，以同一份规范化配置驱动；禁止只改 padding 或只改分类头。
- checkpoint 保存规范化通道顺序、宽度、各通道长度、特征数、表示、类别顺序和 side 标准化参数。`--load` 从 checkpoint 恢复；命令行显式冲突应报错，不能静默采用另一结构。
- 保留默认 width/length、`side=none` 的兼容行为。aux 现有 dropout 按位置保护第一个通道（api）；本轮先保持该基线策略并记录，若改成按语义／缺失质量的策略，另立候选，不能夹带到所有训练中。

**验收**：默认五通道、默认 aux、48/96/192/384 五通道、64/128/256/512 aux、加长 aux 均完成前向和保存／恢复检查；确认原先 9600 对 6400、4096 对 2048 的维数错误消失。

### P0-3. 训练、选择、确认三个阶段分离

以下是**待实现接口**，不是当前脚本已有功能：

| 入口 | 新增或明确的接口 | 行为 |
|---|---|---|
| `run_df.py` | `--splits PATH --eval-splits val` | 开发训练仅构建 train/val 池，保存 checkpoint 和 val 概率 |
| `run_df.py` | `--widths`、`--chan-len`、`--mixup same-class-masked` | 仅显式启用时生效，规则见各条目 |
| `run_df.py` | `--load --eval-out PATH` | 重放只写新目录，不覆盖模型来源目录 |
| `run_lgbm_v2.py` | `--splits PATH --tag TAG --eval-splits val` | 独立缓存键，保存 booster、特征配置和 class_ids，开发阶段不评 test |
| `make_splits.py` | `--dataset 5s --inherit PATH --unmatched exclude --out PATH` | 按主 capture 划分继承；不独立随机分配共享会话 |
| 新 `run_plan_eval.py` | `dev --manifest PATH --scope NAME --out DIR` | 只读开发概率，搜索候选及组合，输出完整 selected.json |
| 新 `run_plan_eval.py` | `freeze --manifest PATH --selections PATHS --out PATH` | 固化最终候选、模型／数据哈希、阈值、比较数和验收规则 |
| 新 `run_plan_eval.py` | `final --freeze PATH --out DIR` | 仅按冻结配置生成／读取 test 预测并确认，不搜索参数 |

`run_plan_eval.py` 可复用现有组合与评分函数，不复制多个彼此不一致的验收实现。开发模式检测到 test 概率文件访问即失败；最终模式检测到未冻结的模型／参数或指纹变化即失败。

组合的 selected.json 至少包含：dataset、split 指纹、有序 member tag、各模型指纹、class_ids、rule、weights、temperature（本轮固定关闭并显式保存）、score 定义、标定池 IDs／指纹、各阈值、开发选择指标和候选清单版本。验收从此文件加载，禁止手抄 `--w4` 而遗漏 `--w3`／rule。

旧 `run_t2_aux_eval.py` 保留为历史 geo 对照工具；本轮通用确认不直接照抄其固定目录名、SHORT 映射和固定 geo 实现。现有 `run_combiner.py --lovo --families plain` 仍可用于历史复现，但不得被当成新流程已完成嵌套评估的证据。

### P0-4. 5s/10s 按同一 capture 划分

主划分冻结为当前 10s strict。5s 共享 capture 继承其 `video/role/split/gallery`，匹配时核验 session_id、capture_id、pcap_sha256；身份冲突必须报错。

- 仅存在于 5s 的样本本轮排除，不随机填进验证或测试。
- 仅存在于 10s 的样本保留在原 10s 训练／单预算报告中，从配对预算比较中排除。
- 缺失 gallery 样本不从其他 split 补齐；保持 class_ids 顺序，报告每类可用 train/gallery/probe 数。若某已知类无可用训练样本，停止并修订协议，不能静默改类别集合。
- 全量核验配对事件前缀及定位一致性；异常样本进入质量报告，级联分析按预先定义的数据完整性规则排除。排除不得看预测是否正确。
- 训练前检查同一源 capture 不跨任意预算的 train/val/test；5s 验证会话也不能进入级联测试。

输出为新文件 `splits/splits_5s_strict_aligned_s42.json`，不覆盖历史 split。两种预算的身份列表预期保持 386/129/129，但会话数及 gallery 完整度按实际报告。已有审计确认 19,362 对来自同一原始抓包；该事实不替代本轮全量映射校验。

### P0-5. 回归与首个正确 side 对照

使用历史缓存回放冻结 geo4 的类别预测、分数、oracle 工作点和部署结果；完整数值从历史 JSON 读取，显示到三位小数应匹配基线表。确定性缓存后处理应数值一致。

重新推理旧 checkpoint 时，先比较概率（预设 `rtol=1e-5, atol=1e-7`）和类别预测，再比较派生指标；阈值附近的差异必须定位，不能事后放宽容差。参数元数据、输出路径等非数值字段可以变化。重放输出写新目录。

修复后的 side 模型与同预算 `side=none` 比较，然后作为 e100 位替代候选参与开发组合。历史 `df_branch_all_seg` 不作为可靠的无效证据，也不因修复就预设会成功。

## 四、统一开发选择与训练约定

### D-1. 有限候选清单

在每批运行前写入 `runs/plan_a_r2/dev_candidates.json`：候选 ID、模型参数、成员顺序、替换／追加关系、输入与 split 指纹、训练预算、seed、输出路径。允许根据开发结果按本文条件开启 B6；不得根据 test 反馈追加候选。

本轮组合搜索固定为 `rule∈{arith,geo}`、`temperature=false`、权重步长 0.05 的非负单纯形，另加等权配置；每个系统最多 5 个成员。筛选时同时放入冻结 geo4，避免搜索范围不包含旧系统。零权重成员不计入实际推理成本。

每个候选逐一搜索其完整组合，不先用单模型 top1 选出“最强者”。按验证 TPR@0.5%、@1%、@5%、top1 依次降序排序；完全并列时依次选活跃成员更少、权重向量及规则名按字典序更前者。保存所有候选，不仅保存头名。

阈值来自同一验证未知池，因此这些是开发分数，允许用于排序，但不声称独立泛化、统计显著或部署 FPR 达标。只在验证层检查候选的已知损失、危险未知会话、每视频分布及误报来源。

### D-2. 候选晋级、停止与联合替换

- 概率非有限、标签错位、shape 错误、模型未完成约定预算且无合法早停等视为实现／运行失败；修复后重跑，不写成方法负结果。
- 正常收敛候选均有一次组合层开发评估机会。argmax 一致率和联合 maxprob>.5 的 10%／30% 历史门槛仅保留作参考列。
- 对基线误接收的开发未知会话，比较候选对基线预测类别的支持与最终接受情况；同时记录原可接受已知会话新增的误拒。均匀概率成员应作为零信息检查，不能仅因降低联合高置信率就被判有价值。
- 若候选在开发主指标上优于冻结 geo4，且 top1 点估计下降不超过 0.5pp、@1%/@5% 点估计下降各不超过 1pp，可进入多 seed 阶段。只是开发晋级，不称为确认提升。
- B1 与 B2 若分别晋级，额外登记一个“各自开发头名同时替换”的配置，重新搜组合。两项单独增益不作相加，联合配置也须独立开发评估。
- 主精度方向只保留开发排名前两名进入 seeds 43/44/45 复验。每 seed 重搜该候选组合；与同 seed 的原四成员系统比较增量。按四 seed 平均开发主指标选定一种结构，平局看最差 seed，再看成本；不选择测试最好的 seed。
- 最终报告预先固定的四 seed 结果，不再增加跨 seed 集成。主精度统计量为同一测试样本上的四 seed 平均配对增量，另列 seed42 对冻结 geo4 的差及每 seed 差。只做单 seed 时必须降级结论，不能称训练随机性已验证。

多 seed 复验保持 split seed=42、LGBM 及其随机性固定，只改变 DF 的 model seed；新成员与被替换角色采用对应 seed，embedding 对照也使用对应 seed 的特征提取器。复验命令应替换公共参数中的 `--model-seed` 并采用新的输出 tag，禁止覆盖 seed42 产物。同 seed 原四成员基线直接冻结已有 [多 seed 报告](E:/evb/df/runs/ms_agg/report.json) 的 geo 配置，不在 test 上重选：42 为 `.35/.10/.20/.35`，43 为 `.35/.15/.15/.35`，44 为 `.40/.10/.15/.35`，45 为 `.35/.10/.15/.40`，顺序均为 LGBM/e100/dt/aux。

### D-3. 训练预算与命令环境

以下命令采用 PowerShell，工作目录为 `E:\evb\df`。**全部依赖 P0 新接口完成**。基线保留历史结果；机制对照使用相同的最大 epochs、早停、batch 和 seed。

```powershell
Set-Location 'E:\evb\df'
$pa10Args = @('--dataset','10s','--protocol','strict','--splits','splits/splits_10s_strict_s42.json','--fusion','branch','--batch','64','--seed','42','--model-seed','42','--eval-splits','val')
$pa5Args = @('--dataset','5s','--protocol','strict','--splits','splits/splits_5s_strict_aligned_s42.json','--fusion','branch','--batch','64','--seed','42','--model-seed','42','--eval-splits','val')
```

每条训练命令显式写 epochs、min_epochs 和 patience。候选发生 OOM 时，登记新的 batch=32 运行配置，并为相应机制比较补同 batch 对照；改变 batch 会改变 BN 和优化步数，不能称“其余训练完全相同”。表中预算是最大值，合法早停可提前结束，实际轮数必须记录。

## 五、方案明细

### A1. 5s 严格协议与四／三成员公平比较

**目标**：比较一个／两个下载 batch 的识别能力，并检验 aux 在短前缀的组合价值。“5s/10s”是数据集标签，不是固定墙钟时长；本轮同时报告 manifest 实际区间时长。

**前置**：P0-3/P0-4 完成。不得独立调用原随机生成器再事后剔除训练会话。

```powershell
# 待实现的继承接口；输出新 split
python scripts/make_splits.py --protocol strict --dataset 5s --seed 42 --inherit splits/splits_10s_strict_s42.json --unmatched exclude --out splits/splits_5s_strict_aligned_s42.json

python scripts/run_lgbm_v2.py --dataset 5s --protocol strict --splits splits/splits_5s_strict_aligned_s42.json --tag pa2_5s_lgbm --eval-splits val
python scripts/run_df.py @pa5Args --channels video_cdn,api,channel_info,thumbnail,other --epochs 100 --min-epochs 15 --patience 12 --tag pa2_5s_e100
python scripts/run_df.py @pa5Args --channels video_cdn,api,channel_info,thumbnail,other --epochs 100 --min-epochs 15 --patience 12 --dt --tag pa2_5s_dt
python scripts/run_df.py @pa5Args --channels api,channel_info,thumbnail,other --epochs 100 --min-epochs 15 --patience 12 --tag pa2_5s_aux

# 新脚本：manifest 的 scope 分别包含三／四成员列表
python scripts/run_plan_eval.py dev --manifest runs/plan_a_r2/dev_candidates.json --scope 5s-3m --out runs/plan_a_r2/dev_5s_3m
python scripts/run_plan_eval.py dev --manifest runs/plan_a_r2/dev_candidates.json --scope 5s-4m --out runs/plan_a_r2/dev_5s_4m
```

两边各自产出 selected.json，保留 rule、weights、temperature 及成员顺序，最终比较直接加载两个文件。不能把四成员搜索的头名权重强制套进 geo，也不能沿用 10s 的三成员默认权重。

**开发决定**：短前缀系统按 D-1/D-2 选择三成员或四成员；该决定在 test 之前固定。如果将“5s aux 提升”作为正式结论，冻结两个系统、主终点和检验，将该比较计入最终比较数 K。未达到正区间只报告证据不足，不改成选其他 FPR 档证明有效。

**产出**：aligned split、两份完整选择配置、开发报告、冻结后的 5s 三／四成员结果，以及相同 capture 子集上的 5s/10s 表。全量单预算表与配对预算表分开。

### A2. 辅助通道长度微调

仅在 B2 的 b2c 中将 api 512→640、channel_info 128→192；默认其他通道不变。与相同宽度和训练预算的 b2b 比较长度效应，再与冻结 geo4 比较系统效应。

video 8192 从本轮默认队列移除：目前超过 4096 的会话约 0.81%，优先级低。该比例不证明收益“不可能”或“不可测”；若后续开发错误集中在被截断子群，再另立预先登记的实验。

### A3. 同类掩码特征混合

**目标**：验证同类特征混合是否能作正则化；不再声称模拟真实清晰度连续切换。本轮仅用于 B1 的 logsize、无 dt 变体。

**精确定义（待实现 `--mixup same-class-masked`）**：

1. 每个训练样本以 p=0.3 触发；伙伴从同 VID 的其他 known/train 会话抽取，优先不同日期。无可用伙伴则跳过；验证、gallery 和 test 不可作伙伴。
2. 对两条原始序列分别完成既有裁头、截尾、大小缩放、通道丢弃和噪声，再各自编码／截断／padding，生成有效事件 mask。保留增强前事件方向作为方向标记。
3. 抽一个 λ~U(0.3,0.7)，在所有通道共用。只对“两侧均为有效事件且方向相同”的位置混合编码大小：`x=λ*x_a+(1−λ)*x_b`。其余位置保留锚样本 a；输出 mask 也保持 a。伙伴 padding 不得削弱锚样本，反向事件不得抵消成 padding。
4. 混合发生在时域增强之后，不对混合结果重新做时间裁剪；标签保持锚样本 VID。不为结果虚构时间轴。本轮若同时指定 `--dt` 或 `--value dir`，参数检查直接报错，留待另行定义。
5. 相同序号不代表同一媒体片段，因此它仍是有风险的特征级正则化假设。记录实际混合位置占比，与相同配置的未混合模型比较；失败只关闭这一具体增强。

### B1. 容量与训练时长分离对照

已有 e100 最佳 epoch=96；dt 最佳 epoch=80、实际训练 93 轮；aux 最佳 epoch=72、实际训练 85 轮。不能据此宣称所有模型均未触发早停或都需要更长训练。

增加相同 150 轮预算的默认宽度对照，用于区分宽度和训练时长；e100 与 dt 角色分别比较。

```powershell
python scripts/run_df.py @pa10Args --channels video_cdn,api,channel_info,thumbnail,other --epochs 150 --min-epochs 15 --patience 20 --tag pa2_b1_ref150
python scripts/run_df.py @pa10Args --channels video_cdn,api,channel_info,thumbnail,other --epochs 150 --min-epochs 15 --patience 20 --dt --tag pa2_b1_ref150_dt
python scripts/run_df.py @pa10Args --channels video_cdn,api,channel_info,thumbnail,other --widths 48,96,192,384 --epochs 150 --min-epochs 15 --patience 20 --tag pa2_b1a_w48
python scripts/run_df.py @pa10Args --channels video_cdn,api,channel_info,thumbnail,other --widths 48,96,192,384 --epochs 150 --min-epochs 15 --patience 20 --dt --tag pa2_b1b_w48_dt
python scripts/run_df.py @pa10Args --channels video_cdn,api,channel_info,thumbnail,other --widths 48,96,192,384 --epochs 150 --min-epochs 15 --patience 20 --mixup same-class-masked --tag pa2_b1c_w48_mx
```

每个候选分别替换对应角色，其余成员固定，按 D-1 搜索组合。先保存全部组合开发结果，再选候选；不得先以 val top1≥.901/.929 淘汰。报告单模型 top1 只是体检列。

P0 side 对照使用以下完整预算：

```powershell
python scripts/run_df.py @pa10Args --channels video_cdn,api,channel_info,thumbnail,other --epochs 100 --min-epochs 15 --patience 12 --side seg --tag pa2_side_fixed
```

与历史 e100 的相同预算配置比较；若基础训练实现改变了初始化／增强随机序列，补一个新的 `side=none` 对照并记录变化，不归因给 side 单因素。

### B2. aux 专家强化

aux 的历史 .654351 为验证 top1，测试 top1 为 .625954。提高单模型 top1 未必提高组合否决效果，本项以替换 aux 后的组合表现晋级。

```powershell
python scripts/run_df.py @pa10Args --channels api,channel_info,thumbnail,other --epochs 100 --min-epochs 15 --patience 12 --dt --tag pa2_b2a_aux_dt
python scripts/run_df.py @pa10Args --channels api,channel_info,thumbnail,other --widths 64,128,256,512 --epochs 100 --min-epochs 15 --patience 12 --tag pa2_b2b_aux_w64
python scripts/run_df.py @pa10Args --channels api,channel_info,thumbnail,other --widths 64,128,256,512 --chan-len api=640,channel_info=192 --epochs 100 --min-epochs 15 --patience 12 --tag pa2_b2c_aux_w64_len
```

分别替换 aux 角色，LGBM/e100/dt 固定并重搜组合。若另想保留旧 aux、把新 aux 当第五成员，必须登记为不同系统，并计算成本；其资格同样由组合开发增量决定，不通过任意 10%／30% 的硬门判定生死。

### B3. 通道子集专家

```powershell
python scripts/run_df.py @pa10Args --channels api --epochs 100 --min-epochs 15 --patience 12 --tag pa2_b3_api
python scripts/run_df.py @pa10Args --channels api,thumbnail --epochs 100 --min-epochs 15 --patience 12 --tag pa2_b3_api_thumb
python scripts/run_df.py @pa10Args --channels thumbnail,channel_info --epochs 100 --min-epochs 15 --patience 12 --tag pa2_b3_thumb_ci
```

各自作为第五成员加入冻结 geo4，独立做一次开发搜索；不在本轮继续任意组合两个第五成员。除一般体检外，记录基线预测类别的置信变化、危险未知视频覆盖及已知误拒损失。成员失败只记录具体配置与开发结果。

### B4. 表示域专家

b4a 使用纯方向表示：

```powershell
python scripts/run_df.py @pa10Args --channels video_cdn,api,channel_info,thumbnail,other --value dir --epochs 100 --min-epochs 15 --patience 12 --tag pa2_b4a_dir
```

b4b 为独立的小型粗时间／字节量表示实验，不以 b4a 的结果为触发条件。实现前把输入窗口、bin 尺度、通道／方向、归一化、mask、池化尺寸和参数量写入 manifest；本轮最多预登记两个尺度配置，预算均为 100 轮、min_epochs=15、patience=12、seed42。若未完成该定义，状态为“待实现”，不能从其他模型的失败推断它无效。

两者均作为第五成员开发比较，优先保留真正改变共同误接收结构的表示。原 10ms 固定对齐模型失败不等于所有粗时间表示都无效。

### B5. LGBM-on-DF-embedding 分类器对照

本项只检验“相同 DF 表示、更换分类器”的增量，不用于宣称骨干轴耗尽。

待实现 `run_lgbm_on_emb.py`：冻结 `dfs_branch_all_e100/model.pt`，使用 `model.eval()` 提取最终分类线性层之前的 512 维向量，记录具体模块位置，关闭 dropout；开发阶段仅提取 train/val。LGBM 使用 known/train 特征和标签训练，超参数预算沿用段特征 LGBM，known/val 用于早停。保存 booster、embedding 模型指纹、类顺序和逐行 IDs。test 提取延迟至最终冻结后。

作为第五成员进行一次开发搜索。DF 表示原来已使用 known/val 选择 checkpoint，故该对照仍属于开发评估，不能把验证向量称为整个表征学习流程的完全折外特征。

### B6. 条件性新骨干

触发条件：批次 2/3 已完成或明确记录资源原因未执行，且没有候选达到 D-2 的开发晋级条件；或者预登记的误差分析显示尺度缺口而已有候选未覆盖。**不得参考 test CI 触发。**

本轮最多两个候选：video trunk 第一层大核 stem（k=32），或多尺度并联卷积。分别在 manifest 固定 padding、输出长度、总参数量、100 轮／min_epochs=15／patience=12 的训练预算，先完成维数前向检查。保留粗化、正则化与既有干扰增强作为控制变量；该选择是本轮受控对照，不是认定其他架构都必然失败。

替换 e100 角色进行开发组合选择，不设 val top1≥.901 的硬门。失败记录为“本轮两个配置未找到增量”。

### C1. 固定保守风险档位

本项输出操作点，不用测试结果反向搜索新的保守系数。对最终冻结系统保存四个验证侧目标：0.5%、0.35%、1%、5%；0.35% 单独标为“保守档”，不能标成已达到真实 0.5%。

实现调用 `_threshold_for_fpr(val_impostor_scores, f)`。分数越高越接受，0.5% 对应上尾约 Q99.5，0.35% 对应约 Q99.65；不能写成取低端 0.35% 分位。阈值离散顺序、相等分数处理和浮点精度应保持一致，测试阈值提高时接受集合只能缩小。

历史 geo4 的固定方案复算如下；是已有对照，不是未来保证：

| 验证目标 | 阈值 | 历史 test 实际 FPR | TPR |
|---|---:|---:|---:|
| 0.5% | .814702 | 1.1151% | 73.7913% |
| 0.35% | .840263 | .8558% | 71.5861% |

输出 oracle、部署标准档、部署保守档及实际 FPR 的视频簇区间。若要宣称满足部署 0.5% 风险预算，须另做绝对风险验证；本表没有证明该预算已满足。对同一数据升阈后 FPR 反而升高，应先查实现。

### C2. 统一 capture 的观察预算级联

前置：A1 完成，5s/10s 的 capture、split、gallery 与类别顺序已统一，系统配置在开发阶段选定。stage1 高分提前接受、低分提前拒绝、中间升级 stage2；所有决策只使用该时刻已经可观测的信息。

改造 `exp4_budget_cascade.py` 时必须读取两端 selected.json，而非固定概率均值或固定目录。早退高／低阈值及二阶段阈值只在 matched val 池选择，网格与排序提前写入 manifest。排序优先满足开发误报预算，其次最大化 TPR、再最小化平均决策时间。最终冻结整套决策函数，不在 test 重搜三个阈值。

**配对比较对象**：完全相同的 matched capture 上，冻结级联 vs 冻结全长系统。两者的实际 FPR 都须报告；不同 FPR 下的 TPR 不称为严格同工作点比较。

**本轮预算非劣门**（均在冻结确认时检查）：

- 配对视频簇 `dTPR=级联−全长` 的区间下界 >−0.01；
- `dFPR=级联−全长` 的区间上界 ≤0.001（最多增加 0.1pp 的本轮工程容忍度）；
- `dTime=级联−全长` 的区间上界 <0。

三个条件同时成立，才称“在报告的实际风险水平下，观察预算下降且召回非劣”。该相对门不等于满足绝对 FPR=0.5%；若要作后者结论，还需两系统实际 FPR 的相应上界均不超过 0.005。CI 重叠不作为持平或等效依据。

时间统计分两层：现有 manifest 只能提供观测区间长度的离线代理；若没有在线时间戳，只能声称该代理减少。端到端决策时间须计入起点定位、确认 batch 结束的静默等待、数据处理及推理，且定位算法不能读取当前决策时刻之后的流量。分别报告已知／未知会话的均值、p50/p95、升级率；总体均值声明已知／未知混合比例，不能无条件外推实际业务流量。

## 六、冻结与最终确认

### F-1. 冻结清单

开发结束后，最多确认三个预先声明的问题：

1. 一个主精度候选相对原四成员系统的四 seed 平均提升；
2. 若预先决定作正式结论，5s 四成员相对其最佳三成员的 aux 增量；
3. 若完成 C2，级联预算非劣与节省。

K 为本轮实际正式比较数，必须在读取新候选 test 输出前固定；未登记的消融结果只能描述，不能另挑显著项。其余候选只保留开发结果，不导出 test 排名榜。

freeze 文件同时固定：代码 commit／工作区补丁指纹、输入与 split 哈希、模型和候选 ID、四个 model seed、组合完整配置、已标定阈值、matched IDs、主次终点、非劣容忍度、K、bootstrap 设置和报告模板。先验证所有产物一致，再允许最终脚本读取 test。

以下为 P0 完成后的主精度流程示例。`--selections` 接收一个或多个空格分隔的选择文件；单个主精度文件包含全部四 seed 的已冻结配置。若同时确认 A1/C2，须在最终运行之前追加其配置文件，并在 manifest 固定 K，不能在看过本次结果后追加。

```powershell
python scripts/run_plan_eval.py dev --manifest runs/plan_a_r2/dev_candidates.json --scope 10s-main --out runs/plan_a_r2/dev_10s_main
python scripts/run_plan_eval.py freeze --manifest runs/plan_a_r2/dev_candidates.json --selections runs/plan_a_r2/dev_10s_main/selected.json --out runs/plan_a_r2/final_freeze.json
python scripts/run_plan_eval.py final --freeze runs/plan_a_r2/final_freeze.json --out runs/plan_a_r2/final_report
```

### F-2. 统计与采纳

所有区间使用双侧视频簇配对 bootstrap，定义见附录 A-2。正式 K 项结论按 `alpha=0.05/K` 使用相应区间，避免“多个比较任一个通过”导致的错误发现膨胀。K=1 时 2000 次；K>1 时 10000 次；rng seed 固定为 0。同一问题内要求的多个约束必须同时通过，不能以次终点替代主终点。

主精度确认要求：平均 `dTPR@0.5%` 的调整后区间下界 >0；@1%/@5% 下界均 >−0.01；平均 top1 点估计下降不超过 0.005。完整报告每 seed 结果，尤其是 seed42 与原冻结 geo4 的对比。通过可称“在历史严格基准和给定训练 seeds 上确认提升”，不能称跨数据划分／环境已验证。

部署报告使用冻结验证阈值，另报实际 FPR 及其区间。oracle 主指标改善不自动获得“部署误报达标”结论。新独立未知池与新时间会话的验证可在配置冻结后补做，未经该补充不宣称新盲测成功。

若最终主候选失败，不测试开发第二名、不改变主终点或风险档位重试；记录“未确认提升”并保留原配置。最终确认不是下一轮调参反馈入口。

## 七、附录 A：可实现的评价定义

### A-1. 验证集 ROC 选择

对固定候选配置 k，在 known/val probe 计算类别、正确性 `correct_k` 和接受分数，在 unknown/val 计算接受分数。每个目标 f 用 `_threshold_for_fpr(imp_val_k,f)` 定阈，计算 `mean(correct_k & score_k≥threshold_k)`。D-1 按这些开发指标排序。

本流程没有逐折学习组合权重，也没有对训练／checkpoint／候选选择做完整外层隔离；所以称“验证集 ROC 选择”。若以后实现分组嵌套评估，应另建版本：外层按未知 VID 与已知 probe VID 分组，内层独立选择所有待调参数，外层只应用，分别汇总 TPR 和实际 FPR；仅把同一组分数拆成五份再拼回不是该评估。对整个训练流程的无偏声明还需处理 checkpoint 对外层已知验证的依赖。

### A-2. 视频簇配对 bootstrap

从实际评价池获取已知 probe VID 列表与未知 VID 列表；分别整簇有放回抽取与原池相同数量的 VID，带入被抽 VID 的所有会话。新旧系统、全部 seeds 使用完全相同的抽样索引。保持会话加权指标，并另外报告每 VID 宏平均；不得混用两种估计目标。

- **oracle ROC 差**：每次重采样内，两套系统及各 seed 分别根据抽中的未知分数重定同一 f 的阈值，再计算 TPR；主精度统计量先算各 seed 的新旧差，再对四 seed 求平均。该区间条件于所用训练 seeds，不是对所有训练随机性或新域的保证。
- **部署差／实际 FPR**：验证阶段冻结的阈值保持不动；只对评价视频簇重采样，计算 TPR、FPR 及差值。不得在测试 bootstrap 内重定阈后还称部署结果。该区间条件于当前标定池；要研究标定不确定性，应额外重采样验证视频簇并单列结果。
- **级联差**：冻结整套早退决策函数，在相同 matched 簇上计算 dTPR、dFPR、dTime。正／负会话分别统计时间；总体时间按冻结的混合比例加权，不能因每次抽样比例变化改变目标。

区间分位点为 `[alpha/2,1−alpha/2]`；正提升必须下界大于零，非劣必须下界大于事先规定的负容忍度。小样本、零误报或少量视频支配尾部时，明确报告有效视频数和区间局限；不能把重采样区间当成对任意未来分布的风险保证。

### A-3. 专家诊断表

保存：成员 top1、未知分数分位数、与各成员的 argmax 一致率、全体 maxprob>.5 比例、基线危险未知 VID 数、对基线预测类别的支持变化、已知新增误拒、加入后的组合增量与推理成本。

历史 10%／30% 门槛作为参考线，不是准入必要条件。全体 maxprob>.5 不要求成员指向同一类，也受概率尺度影响；应同时观察同类支持。通过诊断不等于组合有效，未通过诊断也不直接关闭表示方向。

### A-4. 回归锚点与报告字段

回归只针对固定历史系统和固定全量池；从 [aux 验收报告](E:/evb/df/runs/t2_aux_eval/report.json) 与 [多 seed 报告](E:/evb/df/runs/ms_agg/report.json) 读取精确值，不能把三位小数直接作为精确浮点常量。新 split、matched 子集、规则或温度变化后必须使用其自己的参考配置。

所有结果输出：dataset／表示版本、角色与 capture 数、VID 数、gallery/probe 数、成员与类别顺序、训练 seed／数据 seed、候选选择范围、标定池与阈值、oracle 指标、部署实际 FPR、CI 类型、比较数 K、训练预算、成本、代码与数据指纹。

## 八、既有负结果的处理边界

| 历史实验 | 本轮处理 | 允许的结论边界 |
|---|---|---|
| 合成未知素材、单成员 OE | 不重复同一配置 | 已试设置无增量，不推广到所有未知数据或联合训练方式 |
| ArcFace 头 | 暂缓原头替换 | 同表示／骨干下未提供足够互补性，不证明所有损失都无效 |
| q 族、普通分歧 stacking、GPD、TTA | 不重复原网格 | 当前成员与数据上的负结果，不证明全部组合数学到达上限 |
| 类维 ECDF | 不作为严格协议默认 | 旧标定身份依赖明显，新输入或新校准协议需另证 |
| 温度缩放 | 本轮搜索显式关闭 | 减少搜索空间，不宣称其在所有模型上必然有害 |
| 候选掩码、IP 剪枝 | 本轮不做 | 现实现先算完 CNN 再掩码，不节省主干前向；分类头成本仍与类别数有关 |
| side-MLP | 修复索引后重测 | 旧错配实验不能证明人工摘要无效 |
| merge/stack | 维持 branch 基线 | 既有结构较弱，不统一归因为 Transformer 式记忆化 |

B4a 方向表示失败不关闭 B4b 时间／字节量表示；B5 更换分类器不能给骨干方向下终局结论。未执行或资源不足应标“未执行”，不能记为负结果。

## 九、修订追踪与交付清单

| 审查项 | 本版落实位置 |
|---|---|
| R01 CI 方向 | 全局规则 3、F-2、A-2：正下界与证据不足定义 |
| R02 LOVO 定义 | 全局规则 2、D-1、A-1：明确采用开发验证 ROC |
| R03 test 参与选择 | P0-3、D-2、F-1/F-2：开发隔离、冻结、一次确认与 K |
| R04 宽度维数 | P0-2：真实宽度／长度推导及恢复检查 |
| R05 基线规则遗漏 | P0-3、A1：从 selected.json 加载完整组合 |
| R06 跨预算错划分 | P0-4、A1、C2：训练前按 capture 继承 |
| R07 side 错配 | P0-1/P0-5、B1：修复并恢复正确对照 |
| R08 mixup 语义 | A3：先增强、有效掩码、同方向混合，无 dt |
| R09 不同训练预算 | D-3、B1–B4：显式预算、额外时长及 OOM 对照 |
| R10 保守阈值 | C1：上尾定义、固定风险档与实测局限 |
| R11 CI 重叠 | C2、A-2：配对非劣／风险／时间三个约束 |
| R12 top1 先筛 | D-1/D-2、B1/B2：逐候选组合比较与联合替换 |
| R13 经验硬门 | D-2、A-3：诊断代替不可逆淘汰 |

- [ ] P0 代码改动、缓存／划分审计与基线回归报告。
- [ ] 每批候选清单与所有开发结果，包含未执行、运行失败及证据不足原因。
- [ ] 5s aligned split、完整三／四成员配置及预算匹配质量表。
- [ ] 最终冻结文件、K、统计量、容忍度与全部产物指纹。
- [ ] 冻结后一次确认报告，oracle／部署分开，所有预登记比较完整呈现。
- [ ] findings.md 与 BEST_MODEL.md 只根据已完成证据更新；区分历史协议、当前严格基准和新的独立验证。

本版所列前置项是待办，不表示相应 bug 已修复或新模型已经有效。执行时以完成的检查和冻结产物为依据。
