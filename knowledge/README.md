# 四层知识库格式

知识库不把整本书无差别切片后直接投入向量库。资料按证据职责分成四层，任何回答都必须能从知识卡追溯到注释或原典。

```text
Layer 1 原典全文（originals/）
  -> Layer 2 现代注释（annotations/ 与兼容的 sources/）
  -> Layer 3 结构化知识卡（cards/）
  -> Layer 4 流派关系图谱（graph/，只辅助召回）
  -> 检索回答（引用只能来自 Layer 1-3）
```

## Layer 1：原典全文

每部书一个 YAML 文件。`work` 固定版本元数据，`passages` 按篇、章、条文切分。
状态固定为 `draft -> machine_verified -> reviewed -> retired`。自动检查通过但尚未人工逐字校勘时，
必须使用 `machine_verified`，不得写成 `reviewed`。

```yaml
work:
  id: work-example
  title: 书名
  edition: 影印底本与出版信息
  status: reviewed
passages:
  - id: example-chapter-001
    chapter_path: [卷一, 第一章]
    sequence: 1
    text: 原文
    content_sha256: 64位哈希
    source_pages: [来源页 URL 或影印页码]
    collation_method: 校勘方法
    source_count: 2
    verification_level: multi_source_alignment
    confidence: 0.9
    unresolved_variants: []
    locator: 卷一·第一章·第1条
    concepts: [概念甲]
    graph_refs: [work:example, concept:example]
    status: reviewed
```

切分单位优先使用原书结构，不按固定字数硬切。一个 `id` 对应一个稳定条文；版本变更时保留旧记录并升级版本资料，不静默覆盖。

### 原始来源存档

网络录入本先保存到 `raw/<work-id>/`，再生成原典层。存档至少包含来源网址、下载日期、逐文件 SHA-256、访问限制、清理规则和排除内容；登录受限的下载入口不得绕过。

《子平真诠》的公开网页底本可重复导入：

```bash
.venv/bin/python scripts/import_ziping_zhenquan.py --retrieved-on YYYY-MM-DD
.venv/bin/python scripts/verify_ziping_knowledge.py
```

当前导入保留目录、序文和 47 章原始 HTML，并生成清理后的 TXT 与 310 条正文段落。
徐乐吾凡例、第 48 篇附文，以及第二篇网页夹带的其他典籍材料只在清单中记录，不进入原典层。
自动校验覆盖归档哈希、结构连续性、隐藏水印、夹带材料边界和引用定位。因未找到可合法公开下载的
赵展如原刊影印本，当前作品、段落、2 条注释、248 张候选卡及相关图谱均标记为
`machine_verified + single_source_integrity`，只进入个人预览通道。

## Layer 2：现代注释

译文、注解和研究笔记分别使用 `translation`、`commentary`、`research_note`。每条注释用 `passage_refs` 指回一条或多条原文，并保留作者、出版物、流派和分歧。

```yaml
annotations:
  - id: annotation-example-001
    title: 第一章释义
    kind: commentary
    content: 现代解释
    passage_refs: [example-chapter-001]
    author: 作者
    publication: 出版物
    school: 流派甲
    concepts: [概念甲]
    disagreements: [流派乙对此有不同解释]
    status: reviewed
```

旧的 `sources/*.md` 会作为 `research_note` 继续加载，便于逐步迁移，不会阻塞现有服务。

## Layer 3：结构化知识卡

知识卡不是摘要，而是可执行、可审核的规则单元。`schema_version: 1` 保留旧卡兼容；
`schema_version: 2` 的规则卡必须明确前提、结论、条件、例外、破格、救应、优先级、
反例和原典出处，缺一项仓储就拒绝加载。命例卡必须有四柱、应用步骤、所引规则和原典出处。

```yaml
cards:
  - id: rule-example
    title: 规则名称
    schema_version: 2
    card_type: rule
    content: 对规则的简明解释
    rule: 满足 A 且不满足 B 时，才可讨论 C
    school: 流派甲
    concepts: [概念甲]
    premises: [已按月令提出格局候选]
    conclusion: 满足 A 且不存在 B 时，C 才可进入下一步判断
    conditions: [条件 A]
    exceptions: [例外 B]
    break_conditions: [关键相神被合去]
    rescue_conditions: [另有制忌或合忌]
    priority: 80
    priority_note: 先于行运规则，后于确定性命盘事实
    counterexamples: [只见某十神就直接定格]
    exclusions: [day_master_element=水]
    disagreements:
      - school: 流派乙
        claim: 对条件 A 使用不同权重
        source_refs: [{passage_id: example-chapter-001}]
    prohibited_uses: [不得单独推断健康或具体事件]
    annotation_refs: [annotation-example-001]
    source_refs: [{passage_id: example-chapter-001}]
    graph_refs: [rule:example, school:example]
    status: reviewed
```

当前《子平真诠》v2 内容由 `scripts/build_ziping_rule_cards.py` 从稳定原典段落生成：
231 张规则卡覆盖全部 47 章，其中第 8 至 20 章逐段 71 张、第 31 至 47 章逐段 143 张；
另有 12 张跨正官、财、印、食神、七杀、伤官、阳刃、建禄与外格的原典命例卡。
这些卡保留原文作为内容与结论断点，结构字段负责约束应用顺序，不把生成内容冒充现代权威注释。

```bash
uv run python scripts/build_ziping_rule_cards.py
```

命例卡的定位是“展示规则怎样落到四柱”，不是相似命盘预测器。检索命例后仍须先算当前命盘，
只比较相同的结构事实，并明确不同条件；不得因一柱或一字相同照搬古人的身份、财富或吉凶评价。

`exclusions` 中的 `day_master=甲`、`day_master_element=木`、`day_master_yin_yang=阳` 可由检索器执行过滤；自然语言限制应写入 `conditions`、`exceptions` 或 `prohibited_uses`。

## Layer 4：流派关系图谱

图谱包含 `person`、`concept`、`rule`、`school`、`work`、`source` 等节点，以及带来源的有向边。图谱负责查询扩展和关联召回，不能独立支持回答。

```yaml
nodes:
  - id: concept:example
    type: concept
    name: 概念甲
edges:
  - id: edge-example
    source: school:example
    target: rule:example
    relation: adopts
    source_refs: [{passage_id: example-chapter-001}]
    status: reviewed
```

仓储在启动时校验重复 ID、内容哈希、原文/注释引用、规则与命例互引、图谱端点、
状态信任级别、v2 必填推理字段和显式图谱引用。
正式内容只能引用 `reviewed` 来源；机器校勘内容只能引用 `machine_verified` 或 `reviewed` 来源。
默认只有 `reviewed` 进入检索；个人预览额外纳入 `machine_verified`。

## 审核流程

1. 录入原典并指定版本，状态为 `draft`。
2. 自动检查结构、哈希、夹带材料、引用定位与异文，进入 `machine_verified`。
3. 人工逐字确认来源和异文后，才能升级原典为 `reviewed`。
4. 译注、知识卡和图谱关系按相同状态依赖逐层升级，不得高于所引用来源。
5. 固定评测验证召回、引用链、分歧、禁用范围和通道隔离，再发布索引。
