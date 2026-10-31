# 加密视频 VID 识别（未解密流量）

通过观测加密视频流量识别 YouTube 视频 VID：闭集（top-1/5）+ 开集
（**recall@0.5%/1%/5% FPR**，重点）。

| 文档 | 内容 |
|---|---|
| [EXPLORATION.md](EXPLORATION.md) | 探索过程全链路（思维链：诊断→发现→失败→复活） |
| [findings.md](findings.md) | 关键发现与全部结果表（结论视图） |
| [BEST_MODEL.md](BEST_MODEL.md) | **最优配置、训练方法、复现命令、权重索引** |

里程碑结果（10s test，协议见 findings.md §五）：

| 系统 | top1 | @0.5%FPR | @1%FPR | @5%FPR |
|---|---|---|---|---|
| LGBM 段指纹（第一轮） | .698 | .157 | .200 | .412 |
| DF branch-CNN 单模型 | .934 | .063 | .169 | .573 |
| **三路概率集成（最终）** | **.948** | **.640** | **.704** | **.781** |

结构：`src/`（config/data/segment/dfnet/evaluate 等）、`scripts/`（实验入口）、
`splits/`（固定划分）、`runs/`（报告 + 最优权重，缓存与概率矩阵不入库）。
数据位于 `E:\evb\datasets`（仓库外）。环境见 `requirements.txt`。
