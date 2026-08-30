# 公开合成 Portfolio 评测集 v1

`public-synthetic-portfolio-v1` 与既有公开集及冻结的 boundary v2 完全隔离。它包含 108 条可审计的公开合成样例：检索 40、引用/拒答/冲突/时效 fixture 24、Agent 工具与计划 24、确定性批改 20。语料覆盖自我介绍、科研方法、STAR、英文表达、政策时效、联系导师、压力面、群面、跨专业、研究伦理、数据泄漏、引用、来源权威、拒答、混合检索、Agent、计划、批改、冲突与恢复等不同场景，不使用编号主题或替换年份扩充数量。

精确内容指纹忽略 `case_id` 和切分等顺序元数据；另有 `normalized-trigram-template-audit-v1` 将数字、年份、ID、步骤和常见占位词归一化，再按三元 shingle 计算 Jaccard 相似度。阈值固定为 `0.72`，不会以高阈值放行模板；当前审计 104 项文本，克隆对为 0。

运行脚本的公开输入固定为本目录，不提供替换资产参数；不需要私人资料、外部服务或本地模型。聚合报告不保留样例正文、标识或临时工作目录。回答、Agent 与批改 fixture 分组均明确标记 `metrics_kind=fixture_consistency`，指标名使用 `deterministic_fixture_*`；它们不执行应用 Agent 或模型。延迟来自全部 24 次实际回答 fixture 调用，不对分组分位数二次聚合。所有结果仅证明公开合成回归链路，不能表述为真实用户质量或 RAGSDK 运行结果。
