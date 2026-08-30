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
│   ├── knowledge/           # 知识卡和来源
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

在 `.env` 中填写 OpenAI-compatible 接口：

```dotenv
OPENAI_API_KEY=your-key
OPENAI_BASE_URL=https://your-provider.example/v1
OPENAI_CHAT_MODEL=your-chat-model
```

Key 只由后端读取。没有 Key 时返回可追溯的摘录演示回答。

默认 `hash + memory` 模式无需下载模型。可选检索依赖：

```bash
uv pip install --python .venv/bin/python -r requirements-retrieval.txt
```

本地 BGE-M3 和 reranker：

```bash
uv pip install --python .venv/bin/python -r requirements-local-ml.txt
```

## 知识与评测

- `knowledge/cards/`：带审核状态、条件和出处的知识卡。
- `knowledge/sources/`：按章节切分的 Markdown 来源。
- `knowledge/lightrag-ontology.yml`：LightRAG 实体约束。
- `evals/questions.json`：固定评测问题和期望证据。

只有 `status: reviewed` 的知识卡进入正式检索。图谱抽取只能辅助召回，回答仍需回到审核卡片或原文。

```bash
uv run pytest -q
PYTHONPATH=src uv run python -m bazi_api.cli.evaluate --mode hybrid --limit 5
PYTHONPATH=src uv run python -m bazi_api.cli.export_lightrag
```

## API v1

- `GET /api/v1/health`
- `POST /api/v1/chart`
- `POST /api/v1/chat`
- `GET /api/v1/knowledge/cards`
- `POST /api/v1/feedback`
- `GET /api/v1/observability/recent`

原有 `/api/*` 地址暂时作为隐藏的兼容入口保留，新代码统一使用 `/api/v1/*`。

首版不做大运流年、自动旺衰判定、健康诊断或确定性人生预测。

## 下一阶段

应用数据迁移到 PostgreSQL + SQLAlchemy 2 + Alembic 后，再接 SQLAdmin。知识卡审核、发布、版本和重新索引必须经过业务 Service，不能直接由通用 Admin 绕过。
