# Resume bullets

- Built an AI customer-support agent (Claude, LangGraph, MCP) that resolves order lookups, address changes and policy questions at 98.1% task completion on a 52-case eval, with every data change gated by human confirmation and tenant isolation enforced in code (0 leaks in 7 adversarial cases, 100% injection detection).
- Raised retrieval MRR@5 from 90.4 to 95.2 and recall@5 from 87.2 to 96.2 through a measured ablation of hybrid search (pgvector + BM25), Contextual Retrieval and a reranker swap, and kept agentic RAG off after showing its gains were within noise at 1.5x the latency.
- Built a $0.02-per-PR eval gate with noise-aware regression rules and deterministic number checks that caught an error the LLM judge missed, cut its CI runtime from 6.5 to 2 minutes, and used Langfuse traces to find that memory extraction was 80% of the $0.0017 cost per turn.

## 中文对照

- 基于 Claude、LangGraph 和 MCP 开发电商客服 AI agent，自动处理查单、改地址和政策咨询，在 52 条评测用例上任务完成率 98.1%；所有改动客户数据的操作都需人工确认，用户隔离由代码强制（7 条对抗用例 0 泄露，注入检测率 100%）。
- 通过逐层消融实验（pgvector 向量 + BM25 混合检索、Contextual Retrieval、更换重排模型），把检索 MRR@5 从 90.4 提升到 95.2、recall@5 从 87.2 提升到 96.2；并用数据证明 Agentic RAG 的提升在噪声范围内且延迟增加 1.5 倍，因此默认关闭。
- 搭建每次 PR 约 $0.02 的评测门禁：抗噪声的回归判定规则，加上确定性数字校验（抓住了 LLM 评委漏判的错误）；把 CI 评测时长从 6.5 分钟降到 2 分钟；借助 Langfuse trace 发现记忆抽取占每轮 $0.0017 成本的 80%。
