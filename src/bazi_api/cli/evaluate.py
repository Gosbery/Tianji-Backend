from __future__ import annotations

import argparse
import asyncio
import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from bazi_api.core.config import Settings, get_settings
from bazi_api.modules.charts.schemas import BirthInput, ChartFacts
from bazi_api.modules.charts.service import ChartCalculator
from bazi_api.modules.conversations.service import ChatService
from bazi_api.modules.knowledge.repository import KnowledgeRepository
from bazi_api.modules.knowledge.schemas import EvidenceScope
from bazi_api.modules.retrieval.service import (
    RETRIEVAL_TUNING_FIELDS,
    RetrievalService,
    retrieval_tuning,
)

AnswerMode = Literal["off"]
DEFAULT_BIRTH = {
    "date": "1990-01-01",
    "time": "12:00:00",
    "timezone": "Asia/Shanghai",
    "name": "评测样例",
}
REPORT_METRICS = (
    "recall_at_5",
    "mrr_at_10",
    "evidence_text_consistency",
    "evidence_id_resolution",
    "citation_chain_rate",
)
BASELINE_IDENTITY_FIELDS = (
    "mode",
    "dataset",
    "dataset_sha256",
    "cases",
    "limit",
    "evidence_scope",
    "answer_mode",
    "model_version",
    "index_version",
    "tuning",
)


class EvaluationConfigurationError(ValueError):
    pass


def settings_with_tuning(settings: Settings, overrides: dict[str, int | float]) -> Settings:
    unknown = set(overrides).difference(RETRIEVAL_TUNING_FIELDS)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise EvaluationConfigurationError(f"不支持的检索参数: {names}")
    try:
        return Settings.model_validate({**settings.model_dump(), **overrides})
    except ValidationError as exc:
        raise EvaluationConfigurationError(str(exc)) from exc


def parse_parameter_grid(values: list[str], settings: Settings) -> list[dict[str, int | float]]:
    if not values:
        return [{}]
    options: dict[str, list[int | float]] = {}
    for item in values:
        name, separator, raw_values = item.partition("=")
        name = name.strip()
        if not separator or not raw_values.strip():
            raise EvaluationConfigurationError(f"参数必须使用 name=value 格式: {item}")
        if name not in RETRIEVAL_TUNING_FIELDS or not hasattr(settings, name):
            raise EvaluationConfigurationError(f"不支持的检索参数: {name}")
        current = getattr(settings, name)
        parser = int if isinstance(current, int) and not isinstance(current, bool) else float
        try:
            parsed = [parser(value.strip()) for value in raw_values.split(",")]
        except ValueError as exc:
            raise EvaluationConfigurationError(f"{name} 包含非法数值: {raw_values}") from exc
        options[name] = parsed
    names = list(options)
    combinations = [
        dict(zip(names, combination, strict=True))
        for combination in itertools.product(*(options[name] for name in names))
    ]
    for overrides in combinations:
        settings_with_tuning(settings, overrides)
    return combinations


def compare_with_baseline(
    report: dict[str, Any],
    baseline: dict[str, Any],
    max_regression: float = 0.02,
) -> dict[str, Any]:
    if not math.isfinite(max_regression) or not 0 <= max_regression <= 1:
        raise EvaluationConfigurationError("最大回归幅度必须是 0 到 1 之间的有限数值")
    baseline_tolerance = _nested_number(baseline, "max_regression")
    if baseline_tolerance is None:
        raise EvaluationConfigurationError("基线缺少 max_regression 策略")
    if not 0 <= baseline_tolerance <= 1:
        raise EvaluationConfigurationError("基线 max_regression 策略无效")
    if max_regression > baseline_tolerance:
        raise EvaluationConfigurationError(
            f"--max-regression={max_regression} 超过基线允许值 {baseline_tolerance}"
        )
    for key in BASELINE_IDENTITY_FIELDS:
        if key not in baseline:
            raise EvaluationConfigurationError(f"基线缺少身份字段: {key}")
        if key not in report:
            raise EvaluationConfigurationError(f"当前报告缺少身份字段: {key}")
        if report.get(key) != baseline.get(key):
            raise EvaluationConfigurationError(
                f"基线 {key}={baseline.get(key)!r} 与当前 {report.get(key)!r} 不一致"
            )
    comparisons: dict[str, dict[str, float | bool]] = {}
    for path in REPORT_METRICS:
        current = _nested_number(report, path)
        previous = _nested_number(baseline, path)
        if previous is not None and current is None:
            raise EvaluationConfigurationError(f"当前报告缺少基线要求的指标: {path}")
        if current is None or previous is None:
            continue
        delta = current - previous
        comparisons[path] = {
            "baseline": round(previous, 4),
            "current": round(current, 4),
            "delta": round(delta, 4),
            "passed": delta >= -max_regression,
        }
    if not comparisons:
        raise EvaluationConfigurationError("基线中没有可比较的指标")
    return {
        "max_regression": max_regression,
        "passed": all(item["passed"] for item in comparisons.values()),
        "metrics": comparisons,
    }


def _nested_number(payload: dict[str, Any], path: str) -> float | None:
    value: Any = payload
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _load_cases(settings: Settings, dataset: str) -> list[dict[str, Any]]:
    root = settings.evals_path.resolve()
    path = (root / dataset).resolve()
    if not path.is_relative_to(root):
        raise EvaluationConfigurationError("评测数据集必须位于 evals 目录内")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationConfigurationError(f"无法读取评测数据集 {path}: {exc}") from exc
    if not isinstance(payload, list) or not payload:
        raise EvaluationConfigurationError("评测数据集必须是非空 JSON 数组")
    for index, case in enumerate(payload, start=1):
        if not isinstance(case, dict) or not {
            "id",
            "category",
            "question",
            "expected_ids",
            "expected_policy",
        }.issubset(case):
            raise EvaluationConfigurationError(f"第 {index} 条评测数据缺少必填字段")
        if case["expected_policy"] != "direct_answer":
            raise EvaluationConfigurationError(f"第 {index} 条评测数据的 expected_policy 无效")
    return payload


def _chart_for_case(case: dict[str, Any], calculator: ChartCalculator) -> ChartFacts:
    birth = BirthInput.model_validate(case.get("birth", DEFAULT_BIRTH))
    return calculator.calculate(birth)


async def run(
    mode: str,
    limit: int,
    dataset: str = "questions.json",
    evidence_scope: EvidenceScope = "reviewed_only",
    *,
    parameter_overrides: dict[str, int | float] | None = None,
    answer_mode: AnswerMode = "off",
    settings: Settings | None = None,
) -> dict[str, Any]:
    if limit < 10:
        raise EvaluationConfigurationError("Recall@5 / MRR@10 评测的 --limit 必须至少为 10")
    settings = settings_with_tuning(settings or get_settings(), parameter_overrides or {})
    settings.ensure_directories()
    repository = KnowledgeRepository(settings.knowledge_path)
    repository.load()
    cases = _load_cases(settings, dataset)
    dataset_sha256 = hashlib.sha256(
        json.dumps(
            cases,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    calculator = ChartCalculator()

    retrieval = await RetrievalService.create(
        settings,
        repository.documents("personal_preview"),
        repository.graph_nodes,
        repository.graph_edges,
    )
    report = await _evaluate_cases(
        cases,
        retrieval,
        repository,
        calculator,
        mode,
        limit,
        evidence_scope,
    )

    report.update(
        {
            "mode": mode,
            "dataset": dataset,
            "dataset_sha256": dataset_sha256,
            "limit": limit,
            "evidence_scope": evidence_scope,
            "answer_mode": answer_mode,
            "model_version": retrieval.model_version,
            "index_version": retrieval.index_version,
            "tuning": retrieval_tuning(settings),
        }
    )
    return report


async def _evaluate_cases(
    cases: list[dict[str, Any]],
    retrieval: RetrievalService,
    repository: KnowledgeRepository,
    calculator: ChartCalculator,
    mode: str,
    limit: int,
    evidence_scope: EvidenceScope,
) -> dict[str, Any]:
    recall_hits = 0
    reciprocal_rank = 0.0
    per_category: defaultdict[str, list[int]] = defaultdict(list)
    misses: list[dict[str, Any]] = []
    passage_text = {item.id: item.text for item in repository.original_passages}
    known_ids = {
        *(item.id for item in repository.works),
        *(item.id for item in repository.original_passages),
        *(item.id for item in repository.annotations),
        *(item.id for item in repository.cards),
        *(item.id for item in repository.graph_nodes),
        *(item.id for item in repository.graph_edges),
    }
    citation_ids = 0
    resolvable_citation_ids = 0
    canonical_quotes = 0
    matching_canonical_quotes = 0
    chain_total = 0
    chain_hits = 0
    chart_profiles: set[str] = set()

    for case in cases:
        chart = _chart_for_case(case, calculator)
        chart_profiles.add(chart.birth.model_dump_json(exclude={"name"}))
        query = str(case["question"])
        if case.get("previous_question"):
            query = ChatService._retrieval_query(
                query,
                [
                    {"role": "user", "content": str(case["previous_question"]), "payload": {}},
                    {
                        "role": "assistant",
                        "content": "不参与检索的上一轮助手文本",
                        "payload": {
                            "evidence": [
                                {
                                    "title": title,
                                    "concepts": case.get("previous_evidence_concepts", []),
                                }
                                for title in case.get("previous_evidence_titles", [])
                            ]
                        },
                    },
                ],
                repository.topics,
            )
        else:
            query = ChatService._retrieval_query(query, [], repository.topics)
        results = await retrieval.search(
            query,
            chart,
            str(case.get("school", "基础共识")),
            mode,
            limit,
            evidence_scope,
        )
        expected = set(case["expected_ids"])
        ranked_ids = [result.document.id for result in results]
        ranked_targets = [{result.document.id, *result.document.trace_refs} for result in results]
        success = any(targets & expected for targets in ranked_targets[:5])
        recall_hits += int(success)
        per_category[str(case["category"])].append(int(success))
        rank = next(
            (
                index
                for index, targets in enumerate(ranked_targets[:10], start=1)
                if targets & expected
            ),
            None,
        )
        if rank is not None:
            reciprocal_rank += 1 / rank
        for result in results:
            document = result.document
            ids_to_check = [document.id, *document.trace_refs]
            citation_ids += len(ids_to_check)
            resolvable_citation_ids += sum(item in known_ids for item in ids_to_check)
            if document.kind == "canonical_passage":
                canonical_quotes += 1
                matching_canonical_quotes += int(passage_text.get(document.id) == document.text)
        if case["category"] == "citation_chain":
            chain_total += 1
            chain_hits += int(any(item.document.kind == "canonical_passage" for item in results))
        if not success:
            misses.append(
                {
                    "id": case["id"],
                    "question": case["question"],
                    "expected": case["expected_ids"],
                    "retrieved": ranked_ids,
                }
            )

    total = len(cases)
    recall_at_5 = recall_hits / total
    mrr_at_10 = reciprocal_rank / total
    quote_consistency = matching_canonical_quotes / canonical_quotes if canonical_quotes else 1.0
    citation_resolution = resolvable_citation_ids / max(1, citation_ids)
    citation_chain_rate = chain_hits / chain_total if chain_total else 1.0
    acceptance = {
        "recall_at_5": recall_at_5 >= 0.90,
        "mrr_at_10": mrr_at_10 >= 0.80,
        "evidence_text_consistency": quote_consistency == 1.0,
        "evidence_id_resolution": citation_resolution == 1.0,
    }
    result: dict[str, Any] = {
        "cases": total,
        "chart_profiles": len(chart_profiles),
        "recall_at_k": round(recall_at_5, 4),
        "recall_at_5": round(recall_at_5, 4),
        "mrr_at_10": round(mrr_at_10, 4),
        "evidence_text_consistency": round(quote_consistency, 4),
        "evidence_id_resolution": round(citation_resolution, 4),
        "citation_chain_rate": round(citation_chain_rate, 4),
        "retrieval_match_policy": "document_id_or_resolvable_trace_ref",
        "by_category": {
            category: round(sum(values) / len(values), 4)
            for category, values in per_category.items()
        },
        "misses": misses,
    }
    if "multi_turn" in per_category:
        multi_turn_recall = sum(per_category["multi_turn"]) / len(per_category["multi_turn"])
        result["multi_turn_recall_at_5"] = round(multi_turn_recall, 4)
        acceptance["multi_turn_recall_at_5"] = multi_turn_recall >= 0.90
    result["acceptance"] = acceptance
    result["passed"] = all(acceptance.values())
    return result


async def _run_matrix(
    args: argparse.Namespace,
    settings: Settings,
    parameter_sets: list[dict[str, int | float]],
) -> dict[str, Any]:
    reports = [
        await run(
            args.mode,
            args.limit,
            args.dataset,
            args.scope,
            parameter_overrides=parameters,
            answer_mode=args.answer_mode,
            settings=settings,
        )
        for parameters in parameter_sets
    ]
    if len(reports) == 1:
        return reports[0]
    return {
        "runs": reports,
        "comparison": [
            {
                "tuning": report["tuning"],
                "recall_at_5": report["recall_at_5"],
                "mrr_at_10": report["mrr_at_10"],
                "passed": report["passed"],
            }
            for report in reports
        ],
        "passed": all(report["passed"] for report in reports),
    }


def _load_baseline(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationConfigurationError(f"无法读取基线 {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise EvaluationConfigurationError("基线必须是 JSON 对象")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="bm25", choices=["bm25"])
    parser.add_argument("--limit", default=10, type=int)
    parser.add_argument("--dataset", default="questions.json")
    parser.add_argument(
        "--scope", default="reviewed_only", choices=["reviewed_only", "personal_preview"]
    )
    parser.add_argument("--answer-mode", default="off", choices=["off"])
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="NAME=VALUE[,VALUE...]",
        help="覆盖或扫描检索参数；可重复指定",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--max-regression", default=0.02, type=float)
    parser.add_argument("--require-acceptance", action="store_true")
    args = parser.parse_args()
    if args.limit < 10:
        parser.error("--limit 必须至少为 10，以计算 Recall@5 / MRR@10")
    if not math.isfinite(args.max_regression) or not 0 <= args.max_regression <= 1:
        parser.error("--max-regression 必须是 0 到 1 之间的有限数值")

    settings = get_settings()
    try:
        parameter_sets = parse_parameter_grid(args.param, settings)
        if args.baseline and len(parameter_sets) != 1:
            raise EvaluationConfigurationError("参数扫描不能同时使用单一 --baseline")
        report = asyncio.run(_run_matrix(args, settings, parameter_sets))
        if args.baseline:
            comparison = compare_with_baseline(
                report, _load_baseline(args.baseline), args.max_regression
            )
            report["baseline_comparison"] = comparison
            report["passed"] = bool(report.get("passed")) and comparison["passed"]
        report["max_regression"] = args.max_regression
        rendered = json.dumps(report, ensure_ascii=False, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
    except EvaluationConfigurationError as exc:
        parser.exit(2, f"evaluate: {exc}\n")

    if (args.baseline or args.require_acceptance) and not report["passed"]:
        parser.exit(1)


if __name__ == "__main__":
    main()
