from __future__ import annotations

import argparse
import re
from pathlib import Path

from bazi_api.core.config import get_settings
from bazi_api.modules.knowledge.repository import KnowledgeRepository


def export(output: Path) -> int:
    settings = get_settings()
    repository = KnowledgeRepository(settings.knowledge_path)
    repository.load()
    output.mkdir(parents=True, exist_ok=True)
    documents = repository.documents()

    for document in documents:
        safe_id = re.sub(r"[^A-Za-z0-9._-]+", "-", document.id).strip("-")
        body = (
            f"# {document.title}\n\n"
            f"- 知识层：{document.layer}\n"
            f"- 证据类型：{document.kind}\n"
            f"- 流派命名空间：{document.school}\n"
            f"- 概念：{', '.join(document.concepts)}\n"
            f"- 来源：{document.source}\n"
            f"- 追溯 ID：{', '.join(document.trace_refs)}\n"
            f"- 图谱节点：{', '.join(document.graph_refs)}\n\n"
            f"## 已审核内容\n\n{document.text}\n"
        )
        (output / f"{safe_id}.md").write_text(body, encoding="utf-8")

    relationships = [edge for edge in repository.graph_edges if edge.status == "reviewed"]
    graph_lines = [
        "# 知识图谱关系（仅用于召回）",
        "",
        "这些关系不是独立回答证据，回答必须回到前三层的可引用文档。",
        "",
    ]
    for edge in relationships:
        sources = repository._format_refs(edge.source_refs) or "未标注"
        graph_lines.append(
            f"- `{edge.source}` --{edge.relation}--> `{edge.target}`；来源：{sources}"
        )
    (output / "_graph-relationships.md").write_text("\n".join(graph_lines) + "\n", encoding="utf-8")
    return len(documents)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/lightrag_input"))
    args = parser.parse_args()
    count = export(args.output)
    print(f"Exported {count} reviewed layer 1-3 documents plus graph hints to {args.output}")


if __name__ == "__main__":
    main()
