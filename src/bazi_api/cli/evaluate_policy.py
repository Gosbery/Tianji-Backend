from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Any

from bazi_api.core.config import get_settings
from bazi_api.integrations.llm import classify_question_policy


def evaluate_policy_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    failures = []
    by_policy: defaultdict[str, list[bool]] = defaultdict(list)
    for index, case in enumerate(cases, start=1):
        if not isinstance(case, dict) or not {"id", "question", "expected_policy"}.issubset(case):
            raise ValueError(f"第 {index} 条策略评测数据缺少必填字段")
        expected = str(case["expected_policy"])
        if expected != "evidence_answer":
            raise ValueError(f"第 {index} 条策略评测数据的 expected_policy 无效")
        actual = classify_question_policy(str(case["question"]))
        passed = actual == expected
        by_policy[expected].append(passed)
        if not passed:
            failures.append({"id": case["id"], "expected": expected, "actual": actual})
    total = len(cases)
    if total == 0:
        raise ValueError("策略评测数据集不能为空")
    accuracy = (total - len(failures)) / total
    return {
        "cases": total,
        "policy_accuracy_rate": round(accuracy, 4),
        "by_policy": {
            policy: round(sum(results) / len(results), 4)
            for policy, results in by_policy.items()
        },
        "failures": failures,
        "passed": accuracy == 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="评测问题策略分类器")
    parser.add_argument("--dataset", default="policy-adversarial.json")
    parser.add_argument("--require-perfect", action="store_true")
    args = parser.parse_args()
    root = get_settings().evals_path.resolve()
    path = (root / args.dataset).resolve()
    try:
        if not path.is_relative_to(root):
            raise ValueError("策略评测数据集必须位于 evals 目录内")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("策略评测数据集必须是 JSON 数组")
        report = evaluate_policy_cases(payload)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.exit(2, f"evaluate-policy: {exc}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.require_perfect and not report["passed"]:
        parser.exit(1)


if __name__ == "__main__":
    main()
