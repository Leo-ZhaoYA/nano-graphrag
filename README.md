# nano-graphrag 项目代码总结

## 1. 项目定位

本项目是一个轻量、异步、可替换存储后端的 GraphRAG 实现。它把原始文档转换为“文本块 + 实体关系图 + 社区报告 + 向量索引”，再根据问题从图谱、社区和原文中组织上下文，交给 LLM 生成带引用信息的答案。项目同时保留了不建图的 Naive RAG 路径。

核心公开接口位于 `nano_graphrag/__init__.py`：`GraphRAG` 负责全流程编排，`QueryParam` 描述查询模式和 token/top-k 等约束。

## 2. 分层架构

```text
服务/脚本层
  server.py (FastAPI、文件管理、WebSocket) / client.py / evaluation.py
        ↓
流程编排层
  GraphRAG.ainsert() / GraphRAG.aquery()
        ↓
处理层
  _chunker.py → _extracter.py → _community_generator.py
  _answer_generator.py ← _llm.py、prompt.py、_utils.py
        ↓
抽象存储层
  BaseKVStorage / BaseVectorStorage / BaseGraphStorage
        ↓
具体实现
  JsonKVStorage、NanoVectorDBStorage/HNSWVectorStorage、
  NetworkXStorage/Neo4jStorage
```

`base.py` 定义三类存储接口，使业务流程不直接依赖某个数据库。默认配置使用 JSON KV、NanoVectorDB 和 NetworkX；也可切换到 HNSW 或 Neo4j。`config/nanoG_config.yaml` 主要控制图存储、LLM/Embedding 来源和 Neo4j 连接。

## 3. 建图流程

`GraphRAG.ainsert()` 首先通过 MD5 内容 ID 去重，并调用 `_chunker.py` 的 `get_tabular_chunks()` 将文本或表格拆成带 token 数、来源文档、顺序和类型的 `TextChunkSchema`。随后：

1. `full_docs`、`text_chunks` 持久化原文和分块；可选地写入 `chunks_vdb`。
2. `_extracter.py` 调用配置的 LLM，从每个 chunk 提取实体、关系和描述，并合并同名节点/边后写入图存储。
3. `upsert_entities_vdb()` 将实体名称与描述嵌入，建立实体向量索引。
4. 图存储执行 Leiden 聚类，为节点写入社区信息。
5. `_community_generator.py` 按社区层级打包节点和边，调用 LLM 生成 JSON 报告，保存到 `community_reports`。
6. 各存储的 `index_done_callback()` 将 JSON、GraphML、向量库等中间结果落盘，下一次用同一 `working_dir` 可恢复。

插入支持增量更新和失败恢复，但当前新增内容会清空并重新生成社区报告；并非完整的增量社区计算。

## 4. 查询流程

`GraphRAG.aquery()` 按 `QueryParam.mode` 分派：

- **local**：查询实体向量库，取得相关实体；再沿图取得一跳关系、来源 chunk 和社区报告，按 token 限制组装上下文，由 LLM 生成局部答案。
- **global**：按社区层级、出现次数和报告评分筛选社区；先并行 map 出支持点，再 reduce 成全局答案。
- **naive**：直接查询文本块向量库，将相关 chunk 交给 LLM，不使用知识图谱。

`only_need_context=True` 时返回检索上下文而不调用最终回答；`fix_citation_in_response()` 根据内部 hash 映射修正来源引用。

## 5. 服务入口与外部依赖

`server.py` 提供文件上传/下载、文件列表、chunk 搜索、实体/关系手工 upsert、WebSocket 进度通知等接口。`process_query()` 读取文件、按整体哈希选择工作目录、创建 `GraphRAG`，完成建图后执行 global/local 查询。`client.py` 是对应的异步调用客户端。

LLM 和 Embedding 可使用 OpenAI/Azure OpenAI 或本地 HTTP 服务；代码通过 `_llm.py` 统一封装并用异步并发限制和 KV 缓存控制请求。Neo4j、OpenAI API Key、URL 等敏感配置不应提交到仓库；当前 YAML 示例含有硬编码凭据，部署时应立即替换或移出版本控制。

## 6. 项目特点与限制

优点是核心流程集中、存储接口清晰、支持异步、缓存、表格 chunk、NetworkX 到 Neo4j 的转换，以及可单独获得上下文用于二次集成。主要限制包括：社区报告没有真正的增量更新；默认流程高度依赖 LLM 输出格式和 embedding 服务；Neo4j 与本地文件状态需要保持一致；当前 checkout 未发现可见的 Python 测试文件，维护核心代码时应补充 `tests/test_*.py` 回归测试。

## 7. 典型调用

```python
from nano_graphrag import GraphRAG, QueryParam

rag = GraphRAG(working_dir="./graphrag_cache")
rag.insert("待分析的文档内容")
answer = rag.query("问题", QueryParam(mode="local"))
```

异步应用应使用 `await rag.ainsert(...)` 和 `await rag.aquery(...)`。开发检查命令为 `flake8 . --count --select=E9,F63,F7,F82`，测试命令为 `python -m pytest -v ./`。
