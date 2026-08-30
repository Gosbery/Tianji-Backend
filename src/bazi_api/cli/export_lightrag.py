from __future__ import annotations

import argparse
from pathlib import Path

from bazi_api.core.config import get_settings
from bazi_api.modules.knowledge.repository import KnowledgeRepository


def export(output: Path) -> int:
    settings = get_settings()
    repository = KnowledgeRepository(settings.knowledge_path)
    repository.load()
    output.mkdir(parents=True, exist_ok=True)
    count = 0
    for card in repository.cards:
        if card.status != "reviewed":
            continue
        sources = (
            ", ".join(
                f"{reference.source_id}/{reference.section}" for reference in card.source_refs
            )
            or "项目种子知识卡"
        )
        body = (
            f"# {card.title}\n\n"
            f"- 实体类型：命理概念\n"
            f"- 流派命名空间：{card.school}\n"
            f"- 概念：{', '.join(card.concepts)}\n"
            f"- 适用条件：{', '.join(card.conditions) or '无额外条件'}\n"
            f"- 排除条件：{', '.join(card.exclusions) or '无'}\n"
            f"- 来源：{sources}\n\n"
            f"## 已审核解释\n\n{card.content}\n"
        )
        (output / f"{card.id}.md").write_text(body, encoding="utf-8")
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/lightrag_input"))
    args = parser.parse_args()
    count = export(args.output)
    print(f"Exported {count} reviewed cards to {args.output}")


if __name__ == "__main__":
    main()
