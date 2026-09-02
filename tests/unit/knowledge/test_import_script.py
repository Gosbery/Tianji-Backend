import runpy
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[3]


def test_importer_derives_terms_and_refs_from_graph() -> None:
    namespace: dict[str, Any] = runpy.run_path(
        str(BACKEND_ROOT / "scripts" / "import_ziping_zhenquan.py")
    )
    catalog = namespace["load_graph_concepts"](BACKEND_ROOT / "knowledge" / "graph")

    concepts = namespace["concepts_for"]("论用神", "以月令取用", catalog)
    refs = namespace["graph_refs_for"](concepts, catalog)

    assert {"月令", "用神"}.issubset(concepts)
    assert "concept:month-command" in refs
    assert "concept:yongshen" in refs
    assert "work:ziping-zhenquan" in refs
