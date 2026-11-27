# Resume bullets

**SupportPilot — AI Customer Support Agent** | Python · LangGraph · FastAPI · Claude API · MCP · PostgreSQL (pgvector) · Redis · Langfuse · pytest · Docker · GitHub Actions

- Built a LangGraph + MCP support agent that resolves 110/117 eval tickets across 7 types, with human-confirmed writes, idempotent tools and checkpointed resume; chose one agent over a router after an A/B test (mixed intent 13/14 vs. 6/14).
- Enforced customer isolation in code (identity injected via MCP `_meta`) plus a prompt-injection guard, retries, timeouts, model fallback and human handoff: 0 of 21 protected orders leaked, 13/16 injections flagged with no false positives.
- Raised retrieval MRR@5 from 90.4 to 95.2 through an ablation of hybrid search (pgvector + BM25), Contextual Retrieval and a reranker swap; kept agentic RAG off after its gain stayed within noise at up to 1.7x the latency.
- Built a $0.03-per-PR eval gate with noise-aware regression rules, and from Langfuse traces cut cost per turn 61% ($0.00183 to $0.00072) by caching mem0's extraction prompt, with recall unchanged and p50/p95 latency of 2.4/6.2 s.

All numbers: offline evaluation with Claude Haiku 5.5 as agent and grader (117 cases, 33 labelled
retrieval queries for P5; the MRR figures are from the 26-query P2/P3 set), detailed in
[eval-report.md](eval-report.md). There is no production traffic behind them.

## 中文对照

**SupportPilot — AI 客服 Agent** | Python · LangGraph · FastAPI · Claude API · MCP · PostgreSQL (pgvector) · Redis · Langfuse · pytest · Docker · GitHub Actions

- 基于 LangGraph + MCP 开发客服 agent，在覆盖 7 类工单的评测中端到端解决 110/117 条；改动数据须经人工确认，工具调用幂等，支持断点恢复；通过 A/B 对照选用单 agent 而非“路由 + 专职 agent”（混合意图 13/14 对 6/14）。
- 在代码层强制客户隔离（身份经 MCP `_meta` 注入），并加入提示词注入检测、重试、超时、模型降级与转人工：受保护订单 21 例 0 泄露，注入检出 13/16 且无误报。
- 通过逐层消融（pgvector + BM25 混合检索、Contextual Retrieval、更换重排模型）将检索 MRR@5 从 90.4 提升到 95.2；Agentic RAG 的提升在噪声范围内且延迟最高增加 1.7 倍，因此默认关闭。
- 搭建每次 PR 约 $0.03 的评测门禁（抗噪声回归判定）；依据 Langfuse trace 为 mem0 抽取 prompt 加缓存，每轮成本降低 61%（$0.00183 → $0.00072），记忆召回不变，每轮延迟 p50/p95 为 2.4/6.2 秒。

所有数字均来自离线评测（Claude Haiku 5.5 同时作为 agent 和评分模型；P5 为 117 条用例、33 条带标注的检索查询；MRR 数字来自 P2/P3 的 26 条查询集），详见 [eval-report.md](eval-report.md)，没有真实线上流量支撑。
