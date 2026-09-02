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

当前导入保留目录、序文和 47 章原始 HTML，并生成清理后的 TXT 与 310 条段落。
徐乐吾凡例、第 48 篇附文，以及第二篇网页夹带的其他典籍材料只在清单中记录，不进入原典层。
自动校验覆盖归档哈希、结构连续性、隐藏水印、夹带材料边界和引用定位。因未找到可合法公开下载的
赵展如原刊影印本，当前作品、段落、2 条注释、5 张候选卡及相关图谱均标记为
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

知识卡不是摘要，而是可执行、可审核的规则单元。`content` 供人阅读，`rule` 表达规则；成立条件、例外、流派分歧和禁用范围必须拆开。

```yaml
cards:
  - id: rule-example
    title: 规则名称
    card_type: rule
    content: 对规则的简明解释
    rule: 满足 A 且不满足 B 时，才可讨论 C
    school: 流派甲
    concepts: [概念甲]
    conditions: [条件 A]
    exceptions: [例外 B]
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

仓储在启动时校验重复 ID、内容哈希、原文/注释引用、卡片引用、图谱端点、状态信任级别和显式图谱引用。
正式内容只能引用 `reviewed` 来源；机器校勘内容只能引用 `machine_verified` 或 `reviewed` 来源。
默认只有 `reviewed` 进入检索；个人预览额外纳入 `machine_verified`。

## 审核流程

1. 录入原典并指定版本，状态为 `draft`。
2. 自动检查结构、哈希、夹带材料、引用定位与异文，进入 `machine_verified`。
3. 人工逐字确认来源和异文后，才能升级原典为 `reviewed`。
4. 译注、知识卡和图谱关系按相同状态依赖逐层升级，不得高于所引用来源。
5. 固定评测验证召回、引用链、分歧、禁用范围和通道隔离，再发布索引。
