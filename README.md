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
# Set APP_ACCESS_KEY in .env before starting (see below).
uv run uvicorn bazi_api.main:app --app-dir src --host 127.0.0.1 --port 8000
```

API 文档：[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## 访问控制

当前应用是单用户工作台。前后端必须配置相同的 `APP_ACCESS_KEY`，至少 32 个字符。
使用 `python -c "import secrets; print(secrets.token_urlsafe(32))"` 生成随机口令，分别填入
后端 `.env` 和前端 `.env.local`；不要使用 `NEXT_PUBLIC_` 前缀或将口令提交到版本库。
浏览器首次访问时以用户名 `bazi` 和该口令登录。更换口令后需要重启前后端。

所有 HTTP 接口都要求认证，包括任务、会话、事件流、健康检查、API 文档和旧版入口。
浏览器使用 HTTP Basic；程序调用也可通过 `X-Bazi-Access-Key` 请求头传入口令。
缺少配置时返回 `503`，缺少或错误凭证返回 `401`，不会自动开放本机或代理来源。
公网部署必须使用 HTTPS。共享口令意味着共享同一工作台，不提供多人账号之间的数据隔离。

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

模型 Key 只由后端读取，并保存在 Git 忽略的 `.env` 中。当前 provider 没有 Key 时返回
带引用编号的离线资料目录，不生成个体预测。

检索轨迹包含用户问题与完整证据，默认不开放。需要查看时设置
`OBSERVABILITY_API_KEY`，并以 `Authorization: Bearer <key>` 访问
`/api/v1/observability/*`。观测接口还需要应用访问口令，可同时发送 `X-Bazi-Access-Key`
和观测专用的 `Authorization: Bearer <key>`。会话 ID 只由服务端签发。

同一 SQLite 数据库只允许一个后端服务进程；该进程内可并发处理多个生成任务。
应用在恢复任务前取得操作系统文件锁，第二个进程会明确拒绝启动。关闭或进程退出后释放锁，
再次启动才能恢复中断任务；不要手动删除数据库旁的任务锁文件。
当前事件订阅与取消机制也只面向单进程。部署时使用一个 worker，升级时先停止旧进程再启动新进程。

默认使用本地 `BGE-M3 + bge-reranker-v2-m3 + bm25s`，不调用远程 Embedding，
不需要 API Key 或订阅。首次安装与预热会下载数 GB 模型文件：

```bash
uv sync --extra dev --extra retrieval --extra local-ml
PYTHONPATH=src uv run python scripts/prewarm_retrieval.py
```

向量按“模型版本 + 文档内容哈希”保存在 `data/embedding-cache.sqlite3`。
本地 BGE 默认启动只读缓存，不加载或计算模型；BM25 始终包含全部可用资料。
缓存不完整时，未编码资料仍通过 BM25 参与混合检索。运行上面的预热命令可增量补全向量，
每 8 篇写入一次，已完成的批次在中断后仍可复用；完成后重启后端加载新增缓存。
健康接口的 `vector_documents`、`vector_coverage` 和 `embedding_build_required` 显示实际覆盖情况。

16GB 设备默认使用 CPU、每批 2 个窗口、每窗口最多 512 tokens，长文按重叠窗口覆盖全文后聚合。
窗口参数和算法进入模型版本，旧版向量不会与新版混用，因此升级后需要重新预热。
首次问答才按需加载查询向量模型和重排模型；模型权重仍会占用数 GB，CPU 推理也会慢于 GPU。
建议停止后端后执行预热，避免预热进程与问答进程同时持有模型。
预热默认只构建向量；加 `--benchmark` 才会额外加载重排模型并测试查询延迟。
确需启动时自动补全，可显式设置 `BUILD_EMBEDDINGS_ON_STARTUP=true`。
模型不可用时检索采用既有降级策略；预热命令会明确报错，避免把降级结果误认为 BGE 缓存已完成。

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
PYTHONPATH=src uv run python -m bazi_api.cli.export_feedback_candidates \
  --database data/app.db --output evals/candidates/feedback.json
```

检索参数可重复使用 `--param name=v1,v2` 做矩阵扫描，并用 `--output` 保存报告。
CI 使用纯本地 `hash + lexical` 跑完整离线答案评测，再通过
`--baseline evals/baselines/questions.hash.hybrid.json` 和 `--max-regression` 阻止质量回退；
`--require-acceptance` 可让未达到题库验收阈值的运行直接返回失败。
仓库根目录的 `make eval` 固定使用 `hash + lexical`，同时运行 61 条主评测、10 条多轮追问
和 52 条策略对抗评测。
负反馈导出只生成 `pending_human_review` 候选，不会自动加入固定题库或在线修改模型。

历史版本（625 条索引、原始 BGE 编码）的专项评测结果：《子平真诠》150 题
`Recall@5 = 0.9733`、`MRR@10 = 0.9314`，引用文本一致率和引用 ID 可解析率均为 `1.0`，
边界题检索命中率为 `0.90`。625 条文档全部缓存命中时，索引加载约 `0.34 s`，
热身后检索 `p95` 约 `1.56 s`。第一次进程内加载 BGE-M3 与重排模型约需 11 秒，
属于该历史版本的冷启动成本。这些数字不能代表当前扩展知识库与分窗编码版本，
升级后应重新评测；当前正常启动不会预先加载两个 BGE 模型。

2026-09-09 使用新版分窗 BGE、完整 3,439 条缓存、CPU 和 `hybrid`（不启用重排）复测：
主库 61 题 `Recall@5 = 0.9672`、`MRR@10 = 0.9372`；子平 150 题
`Recall@5 = 0.9333`、`MRR@10 = 0.8329`，全部缓存命中且无模型降级。
这些是证据检索指标；离线答案只是资料目录，不能代表真实 LLM 回答准确率。
本机 Apple M5 / 16GB 的完整缓存启动约 `7.33 s`，macOS physical footprint 峰值约
`0.97 GiB`，启动过程未导入 PyTorch 或 sentence-transformers。首次问答按需加载模型后
仍会占用数 GB，不能将启动峰值当作整个问答过程的内存上限。

评测把“返回文档本身命中”与“规则卡/命例卡的可解析 `trace_refs` 命中”都计作召回，
因为四层架构要求规则通过引用链命中原典；原文一致率和引用 ID 可解析率仍单独要求 100%，
不能用不存在或内容不一致的引用抬高召回。

## API v1

- `GET /api/v1/health`
- `POST /api/v1/chart`
- `POST /api/v1/chat`
- `POST /api/v1/chat/stream`
- `GET /api/v1/conversations/{session_id}`
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

当前版本允许回答各类命理预测问题；回答仍区分程序事实、知识依据与命理推演。
会话绑定命盘指纹、流派和证据范围；上下文不一致时返回
`409 session_context_mismatch`。模型回答先检查引用格式和明确禁止的医疗建议，再通过独立模型请求
核验结论与证据的支持关系、输出边界，以及多方面问题是否逐项回答或明确证据不足。
生成和核验使用同一份完整规则上下文，关键前提、例外、禁忌与出处不截断。
核验失败、格式异常或核验服务不可用时不放行该回答，
最多修复一次，再次失败返回固定的安全说明，不复述失败草稿或原文断语。
独立核验会增加模型调用和延迟，也不能替代人工对事实和专业内容的判断。
当前历法适配只支持 `Asia/Shanghai` 民用时间；扩展其他时区前必须完成节气瞬时转换校验。

## 下一阶段

应用数据迁移到 PostgreSQL + SQLAlchemy 2 + Alembic 后，再接 SQLAdmin。知识卡审核、发布、版本和重新索引必须经过业务 Service，不能直接由通用 Admin 绕过。
