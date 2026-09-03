# bazi-api

命盘显微镜后端：确定性排盘、结构化知识、混合检索和带证据的 AI 解读。

## 架构

后端是按业务领域组织的模块化单体，不是微服务集合。

```text
src/bazi_api/
├── main.py                  # 应用工厂与路由注册
├── core/                    # 配置、依赖容器
├── db/                      # SQLite 基础设施；后续替换 PostgreSQL
├── modules/
│   ├── charts/              # 确定性排盘
│   ├── knowledge/           # 原典、注释、知识卡和关系图谱
│   ├── retrieval/           # BM25、向量、RRF、重排
│   ├── conversations/       # 聊天用例编排与会话仓储
│   ├── feedback/            # 反馈写入
│   ├── observability/       # 检索轨迹
│   └── system/              # 健康检查
├── integrations/            # LLM 与 Embedding provider
└── cli/                     # 评测和 LightRAG 导出
```

每个模块只创建实际需要的层：HTTP 放在 `router.py`，接口数据放在 `schemas.py`，数据读写放在 `repository.py`，业务流程放在 `service.py`。排盘和检索不是 CRUD，因此没有人为添加空的 repository。

## 启动

```bash
uv sync --extra dev
cp .env.example .env
uv run uvicorn bazi_api.main:app --app-dir src --host 127.0.0.1 --port 8000
```

API 文档：[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## 模型配置

支持 OpenAI-compatible 与 Anthropic Messages 两种接口。OpenAI-compatible 配置：

```dotenv
OPENAI_API_KEY=your-key
OPENAI_BASE_URL=https://your-provider.example/v1
OPENAI_CHAT_MODEL=your-chat-model
```

Anthropic 配置：

```dotenv
LLM_PROVIDER=anthropic
ANTHROPIC_AUTH_TOKEN=your-key
ANTHROPIC_BASE_URL=https://api.anthropic.com
ANTHROPIC_CHAT_MODEL=claude-sonnet-4-6
```

Key 只由后端读取，并保存在 Git 忽略的 `.env` 中。当前 provider 没有 Key 时返回可追溯的摘录演示回答。

检索轨迹包含用户问题与完整证据，默认不开放。需要查看时设置
`OBSERVABILITY_API_KEY`，并以 `Authorization: Bearer <key>` 访问
`/api/v1/observability/*`。会话 ID 只由服务端签发；当前版本没有用户账号体系，
因此若部署到公网，仍应在反向代理或网关层为整个应用配置访问控制。

默认使用本地 `BGE-M3 + bge-reranker-v2-m3 + bm25s`，不调用远程 Embedding，
不需要 API Key 或订阅。首次安装与预热会下载数 GB 模型文件：

```bash
uv sync --extra dev --extra retrieval --extra local-ml
PYTHONPATH=src uv run python scripts/prewarm_retrieval.py
```

向量按“模型版本 + 文档内容哈希”保存在 `data/embedding-cache.sqlite3`；应用重启直接加载，
只有新增或变化内容会重新编码。模型不可用时自动降级到本地 hash 向量、BM25 和词法重排。

当前约 625 条预览检索文档继续使用 YAML + 内存向量索引，不需要 Qdrant。
达到 10,000 条、需要多进程共享索引或索引内存超过 1GB 时，再安装 Qdrant 客户端，
并部署独立 Qdrant Server，通过 `QDRANT_URL` 连接：

```bash
uv sync --extra qdrant
```

`QDRANT_PATH` 的嵌入式模式只适合单进程本地开发；其目录带独占锁，不能由多个
API worker 共享。嵌入式模式会清理旧的项目集合；生产和多进程部署必须使用独立
服务，并由部署流程管理旧版本集合的保留与回收。

LightRAG 默认关闭；图谱超过 500 个节点并出现大量跨书、人物、流派关系问题后，
再作为补充检索通道评估，不替代当前证据链。

## 知识与评测

- `knowledge/originals/`：带版本信息、按篇章条文切分的原典全文。
- `knowledge/annotations/`：逐条绑定原文的现代译注与研究资料。
- `knowledge/cards/`：带规则、条件、分歧、禁用范围和出处的知识卡。
- `knowledge/graph/`：人物、概念、规则、流派与出处的节点和关系。
- `knowledge/sources/`：兼容旧资料，加载时归入现代研究资料层。
- `knowledge/lightrag-ontology.yml`：LightRAG 实体约束。
- `evals/questions.json`：固定评测问题和期望证据。
- `evals/ziping-zhenquan.json`：150 道《子平真诠》专项题。

完整字段、录入示例和审核流程见 [`knowledge/README.md`](knowledge/README.md)。
默认 `reviewed_only` 只使用人工审核证据；个人预览可显式开启 `personal_preview`，
额外纳入 `machine_verified`，但永不纳入 `draft` 或 `retired`。知识图谱只做查询扩展和关联召回。

《子平真诠》当前有 231 张 v2 规则卡和 12 张原典命例卡。规则卡覆盖全部 47 章，
并逐段覆盖第 8 至 20 章及第 31 至 47 章。v2 规则缺少前提、结论、例外、破格、救应、
优先级、反例或原典引用时，应用启动即拒绝加载。排盘结果还会在模型调用前确定性计算月令本气、
透干、同干/同五行根、天干合、地支合冲刑害、三合三会候选与格局候选；候选不等于成格或合化。

只有 `status: reviewed` 的知识卡进入正式检索。机器校勘内容必须由请求显式开启，
并在每条证据上返回风险标签、置信度和未解决异文。图谱抽取只能辅助召回，回答仍需回到卡片、注释或原文。

```bash
uv run python -m pytest -q
PYTHONPATH=src uv run python -m bazi_api.cli.evaluate --mode hybrid_rerank --limit 10
PYTHONPATH=src uv run python -m bazi_api.cli.evaluate --mode hybrid_rerank --limit 10 \
  --dataset ziping-zhenquan.json --scope personal_preview
PYTHONPATH=src uv run python -m bazi_api.cli.export_lightrag
```

检索参数可重复使用 `--param name=v1,v2` 做矩阵扫描，并用 `--output` 保存报告。
CI 使用纯本地 `hash + lexical` 跑完整离线答案评测，再通过
`--baseline evals/baselines/questions.hash.hybrid.json` 和 `--max-regression` 阻止质量回退；
`--require-acceptance` 可让未达到题库验收阈值的运行直接返回失败。

当前 243 张 v2 卡加入后的真实 BGE 专项评测结果：《子平真诠》150 题
`Recall@5 = 0.9733`、`MRR@10 = 0.9314`，引用文本一致率和引用 ID 可解析率均为 `1.0`，
边界题检索命中率为 `0.90`。625 条文档全部缓存命中时，索引加载约 `0.34 s`，
热身后检索 `p95` 约 `1.56 s`。第一次进程内加载 BGE-M3 与重排模型约需 11 秒，
属于冷启动成本；本地模型缺失或加载失败时仍自动降级到 hash 向量与词法重排。

评测把“返回文档本身命中”与“规则卡/命例卡的可解析 `trace_refs` 命中”都计作召回，
因为四层架构要求规则通过引用链命中原典；原文一致率和引用 ID 可解析率仍单独要求 100%，
不能用不存在或内容不一致的引用抬高召回。

## API v1

- `GET /api/v1/health`
- `POST /api/v1/chart`
- `POST /api/v1/chat`
- `GET /api/v1/knowledge/cards`
- `GET /api/v1/knowledge/overview`
- `GET /api/v1/knowledge/originals`
- `GET /api/v1/knowledge/annotations`
- `GET /api/v1/knowledge/graph`
- `POST /api/v1/feedback`
- `GET /api/v1/observability/recent`

兼容用的 `/api/*` 路由会返回 `Deprecation` 和 `Sunset` 响应头；客户端迁移完成后可设置
`LEGACY_API_ENABLED=false` 关闭。跨域请求仅允许 `GET`、`POST` 以及应用实际使用的请求头。

原有 `/api/*` 地址暂时作为隐藏的兼容入口保留，新代码统一使用 `/api/v1/*`。

首版不做大运流年、自动旺衰判定、健康诊断或确定性人生预测。
当前历法适配只支持 `Asia/Shanghai` 民用时间；扩展其他时区前必须完成节气瞬时转换校验。

## 下一阶段

应用数据迁移到 PostgreSQL + SQLAlchemy 2 + Alembic 后，再接 SQLAdmin。知识卡审核、发布、版本和重新索引必须经过业务 Service，不能直接由通用 Admin 绕过。
