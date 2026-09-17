# bazi-api

命盘显微镜后端：确定性排盘、结构化知识、按需查典核验和基于排盘与话题事实的直接解读。

## 架构

后端是按业务领域组织的模块化单体，不是微服务集合。

```text
src/bazi_api/
├── main.py                  # 应用工厂与路由注册
├── core/                    # 配置、依赖容器
├── db/                      # SQLite 基础设施；后续替换 PostgreSQL
├── modules/
│   ├── charts/              # 确定性排盘 + 话题事实（财官印配星、配偶宫、五行分布、神煞、流年）
│   ├── knowledge/           # 原典、注释、知识卡和关系图谱；核验语料
│   ├── retrieval/           # bm25 单模式检索，仅在按需查典时调用
│   ├── conversations/       # direct 解读编排与按需查典核验端点
│   ├── experts/             # 专家流派配置
│   ├── feedback/            # 反馈写入
│   ├── observability/       # 检索轨迹
│   ├── tasks/               # 生成任务与事件流
│   └── system/              # 健康检查
├── integrations/            # LLM provider
└── cli/                     # 评测与导出
```

每个模块只创建实际需要的层：HTTP 放在 `router.py`，接口数据放在 `schemas.py`，数据读写放在 `repository.py`，业务流程放在 `service.py`。排盘和检索不是 CRUD，因此没有人为添加空的 repository。

## 知识库角色与直接解读

主回答不经过检索：`charts/` 先做确定性排盘并计算话题事实，`conversations/` 把这些结构化事实连同用户问题一起交给 LLM，由模型直接生成解读。知识库因此退为**按需查典核验通道**——只有用户明确要求核对某个术语、条文或原文出处时，才用 bm25 在 `knowledge/` 语料中检索并返回证据，供人核对，不参与主回答的生成路径。

这条设计取舍有代价：主回答的事实边界依赖模型自身，不再由检索证据逐条约束。因此核验端点返回的每条证据仍然带可解析的引用 ID 和原文文本，便于人工回溯；知识卡的 `reviewed` 状态、规则卡的前提取舍和出处要求也仍然生效。

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

直接解读必须配置模型 Key，没有 Key 时生成接口返回 `503`，不会降级为资料目录。
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

模型 Key 只由后端读取，并保存在 Git 忽略的 `.env` 中。

检索轨迹包含用户问题与完整证据，默认不开放。需要查看时设置
`OBSERVABILITY_API_KEY`，并以 `Authorization: Bearer <key>` 访问
`/api/v1/observability/*`。观测接口还需要应用访问口令，可同时发送 `X-Bazi-Access-Key`
和观测专用的 `Authorization: Bearer <key>`。会话 ID 只由服务端签发。

同一 SQLite 数据库只允许一个后端服务进程；该进程内可并发处理多个生成任务。
应用在恢复任务前取得操作系统文件锁，第二个进程会明确拒绝启动。关闭或进程退出后释放锁，
再次启动才能恢复中断任务；不要手动删除数据库旁的任务锁文件。
当前事件订阅与取消机制也只面向单进程。部署时使用一个 worker，升级时先停止旧进程再启动新进程。

后端依赖只有 `uv sync --extra dev` 一组可选依赖；启动不需要下载模型权重，也不需要预热任何索引。
bm25 索引直接基于 `knowledge/` 语料在进程内构建。

## 知识与评测

- `knowledge/originals/`：带版本信息、按篇章条文切分的原典全文。
- `knowledge/annotations/`：逐条绑定原文的现代译注与研究资料。
- `knowledge/cards/`：带规则、条件、分歧、禁用范围和出处的知识卡。
- `knowledge/graph/`：人物、概念、规则、流派与出处的节点和关系。
- `knowledge/sources/`：兼容旧资料，加载时归入现代研究资料层。
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
PYTHONPATH=src uv run python -m bazi_api.cli.evaluate --mode bm25 --limit 10
PYTHONPATH=src uv run python -m bazi_api.cli.evaluate --mode bm25 --limit 10 \
  --dataset ziping-zhenquan.json --scope personal_preview
PYTHONPATH=src uv run python -m bazi_api.cli.evaluate_policy \
  --dataset policy-adversarial.json --require-perfect
PYTHONPATH=src uv run python -m bazi_api.cli.export_feedback_candidates \
  --database data/app.db --output evals/candidates/feedback.json
```

检索参数可重复使用 `--param name=v1,v2` 做矩阵扫描，并用 `--output` 保存报告。

评测把“返回文档本身命中”与“规则卡/命例卡的可解析 `trace_refs` 命中”都计作召回，
因为四层架构要求规则通过引用链命中原典；原文一致率和引用 ID 可解析率仍单独要求 100%，
不能用不存在或内容不一致的引用抬高召回。

### 实测指标与门限来源

2026-09-17 用 `--mode bm25` 实测：

| 数据集 | 题量 | Recall@5 | MRR@10 |
| --- | --- | --- | --- |
| `questions.json` | 61 | 0.7869 | 0.5998 |
| `multi-turn.json` | 10 | 0.9000 | 0.6950 |

主库的引用文本一致率、引用 ID 可解析率和引用链命中率均为 `1.0`。13 条未命中：
`q004`、`q006`、`q007`、`q014`、`q018`、`q028`、`q029`、`q032`、`q035`、`q036`、
`q046`、`q056`、`q058`。其中六条是单点定义题（`q006`、`q007`、`q014`、`q018`、
`q028`、`q036`，如“壬水的阴阳属性是什么”）：前 5 名被注释段落和同类近邻节点占满，
期望的单条 `stem-*` / `branch-*` / `god-*` 节点没能进入前 5；
其余七条是“能否直接断定”的推演边界题。期望证据 ID 和评分规则没有为通过这些用例而调整。

绝对门限按上面的实测值重新设定，并在实测值下留出余量：

| 门限 | 旧值（hybrid） | 新值（bm25） |
| --- | --- | --- |
| `recall_at_5` | 0.90 | 0.75 |
| `mrr_at_10` | 0.80 | 0.55 |
| `multi_turn_recall_at_5` | 0.90 | 0.85 |

门限下调不是质量回退，而是量纲变了：本地向量栈按设计删除，检索从主流程降级为按需查典通道，
旧 `hybrid` 门限（0.90 / 0.80，对应 Recall@5 = 0.9344 / MRR@10 = 0.8864 的向量实现）在 bm25
单通道下不可达。在此之前的历史评测数字都由已删除的向量栈产生，保留在 Git 历史里，当前代码无法复现。
基线文件 `evals/baselines/questions.hash.hybrid.json` 的身份字段（`mode: hybrid`、
`model_version: hash-*`）已随该栈失效，已删除并由 `evals/baselines/questions.bm25.json` 取代；
详见 [`evals/baselines/README.md`](evals/baselines/README.md)。

CI 和 `make eval` 用 `--baseline evals/baselines/questions.bm25.json` 与 `--max-regression`
（0.02）阻止质量回退，并用 `--require-acceptance` 让未达到上述绝对门限的运行直接返回失败。
基线绑定 `dataset_sha256`、`cases`、`limit`、`answer_mode`、`model_version`、`index_version`
和检索参数，任一项不一致即拒绝比较，所以语料或配置变更必须先重新生成候选报告并人工确认。

### 策略评测的口径

`evaluate_policy` 的“策略分类”在改造前后都只是一个常量（改造前返回 `"evidence_answer"`，
现在返回 `"direct_answer"`），没有真正的分类器。因此这个门**只校验数据集形状**
（必填字段、`expected_policy` 的合法取值、数据集非空），其 `policy_accuracy_rate` 恒为 `1.0`，
**不衡量任何策略分类准确率**。它保留下来是为了防止数据集被写成不可用的形状，不能当作模型能力的证据。

负反馈导出只生成 `pending_human_review` 候选，不会自动加入固定题库或在线修改模型。

## API v1

- `GET /api/v1/health`
- `POST /api/v1/chart`
- `POST /api/v1/chat`
- `POST /api/v1/chat/stream`
- `POST /api/v1/conversations/{session_id}/messages/{message_id}/verification`
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
`409 session_context_mismatch`。

direct-v1 回答的边界由两层约束构成，**没有**对答案的独立模型核验：

1. **提示词层**：要求未来只给条件式趋势、时间窗口和不确定性，不得承诺具体年份必然
   发财、升职、结婚、离婚或发生灾祸；健康内容只讲传统文献观点与生活习惯提醒，不诊断
   疾病、不预测寿命或死亡时间、不安排手术或建议用药停药；古籍中的身份、性别、婚姻和
   疾病断语必须标明历史语境。**“古籍断语套用现实个人”这类语义边界无法用正则可靠判定，
   只能留在提示词软约束层。**
2. **确定性正则安全闸**（`integrations/llm.py` 的 `_explicit_safety_violation`）：
   在整段规范化文本与逐子句两级扫描，覆盖年份/时间 + 断言结果的承诺（含结果词在断言词
   之前、跨逗号断句、疑问年份词三种形态）、死期与寿元表述、医疗越界（诊断/停药/剂量/
   手术）以及**编造篇章或条文编号**（direct 通道无知识库可依据，出现 `第X章/节/条/篇/卷`
   即判违规）。命中即不采用该次输出，最多修复一次，再次失败返回固定的安全说明，
   不复述失败草稿或原文断语。

安全闸是少量正则，不是语义审查：它拦得住高置信度的越界措辞，拦不住同义改写，误判与
漏判都需要按回归数据持续调整。人工核对具体论断时走“查典籍”通道——用 bm25 在已审核
语料中检索原文与出处，供人比对，这条通道不参与主回答的生成。
当前历法适配只支持 `Asia/Shanghai` 民用时间；扩展其他时区前必须完成节气瞬时转换校验。

## 下一阶段

应用数据迁移到 PostgreSQL + SQLAlchemy 2 + Alembic 后，再接 SQLAdmin。知识卡审核、发布、版本和重新索引必须经过业务 Service，不能直接由通用 Admin 绕过。
