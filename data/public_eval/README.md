# 公开脱敏检索评测集

这是一组完全合成、可公开再分发的最小检索评测集，用于验证评测代码和建立 BM25 baseline。它不包含私人知识库正文、真实院校规则或个人经历，不能代表真实业务质量。

- `corpus.jsonl`：12 个合成知识片段。
- `questions.jsonl`：12 个可回答问题和 3 个不可回答问题。
- 指标：Recall@K、MRR、nDCG@K、不可回答识别准确率和平均检索延迟。

后续 RAGSDK Dense、Hybrid 和 Reranker 必须在相同语料、问题、`top_k` 与生成模型设置下运行，才可与 BM25 结果比较。
