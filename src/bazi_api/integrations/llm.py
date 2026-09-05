from __future__ import annotations

import json
import logging
import re
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

import httpx

from bazi_api.core.config import Settings
from bazi_api.core.errors import InvalidUpstreamResponseError, UpstreamServiceError
from bazi_api.modules.charts.schemas import ChartFacts
from bazi_api.modules.retrieval.schemas import RetrievalHit

from .http import post_with_retries

logger = logging.getLogger(__name__)

PolicyDecision = Literal[
    "allow",
    "refuse_no_evidence",
    "refuse_invalid_citations",
]
QuestionPolicy = Literal["evidence_answer"]


@dataclass
class GenerationResult:
    answer: str
    uncertainties: list[str]
    followups: list[str]
    token_usage: int | None = None
    policy_decision: PolicyDecision = "allow"
    question_policy: QuestionPolicy = "evidence_answer"
    citations_validated: bool = False
    uncertainty_validated: bool = False
    degradation_reason: str = ""
    model_version: str = ""
    prompt_version: str = "open-prediction-v4"


class AnswerGenerator:
    def __init__(self, settings: Settings, http_client: httpx.AsyncClient) -> None:
        self.provider = settings.llm_provider
        if self.provider == "anthropic":
            self.api_key = settings.anthropic_auth_token
            self.base_url = settings.anthropic_base_url.rstrip("/")
            self.model = settings.anthropic_chat_model
        else:
            self.api_key = settings.openai_api_key
            self.base_url = settings.openai_base_url.rstrip("/")
            self.model = settings.openai_chat_model
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens
        self.timeout = settings.llm_timeout_seconds
        self.max_retries = settings.http_request_retries
        self.http_client = http_client

    async def generate(
        self,
        question: str,
        chart: ChartFacts,
        hits: list[RetrievalHit],
        *,
        school: str = "基础共识",
        evidence_scope: str = "reviewed_only",
        history: list[dict[str, str]] | None = None,
        expert_context: str = "",
    ) -> GenerationResult:
        question_policy = classify_question_policy(question)
        if not hits:
            return self._no_evidence(question_policy)
        if not self.api_key:
            logger.info("llm_extractive_demo", extra={"provider": "extractive-demo"})
            return self._extractive_demo(question, chart, hits, question_policy)

        generated = await self._generate_once(
            question,
            chart,
            hits,
            school=school,
            evidence_scope=evidence_scope,
            history=history or [],
            expert_context=expert_context,
        )
        generated.question_policy = question_policy
        generated.model_version = self.model
        self._validate_model_generation(generated, len(hits))
        if self._generation_is_trusted(generated):
            return generated

        logger.warning(
            "llm_generation_repair",
            extra={"provider": self.model, "error_code": self._validation_reason(generated)},
        )
        repaired = await self._generate_once(
            question,
            chart,
            hits,
            school=school,
            evidence_scope=evidence_scope,
            history=history or [],
            expert_context=expert_context,
            repair_reason=self._validation_reason(generated),
        )
        repaired.question_policy = question_policy
        repaired.model_version = self.model
        self._validate_model_generation(repaired, len(hits))
        if self._generation_is_trusted(repaired):
            return repaired
        fallback = self._extractive_demo(question, chart, hits, question_policy)
        fallback.degradation_reason = "model_output_failed_validation"
        fallback.model_version = self.model
        return fallback

    async def _generate_once(
        self,
        question: str,
        chart: ChartFacts,
        hits: list[RetrievalHit],
        *,
        school: str,
        evidence_scope: str,
        history: list[dict[str, str]],
        expert_context: str = "",
        repair_reason: str = "",
    ) -> GenerationResult:

        context = self._evidence_context(hits)
        chart_json = self._chart_context(chart)
        system = self._system_prompt(school, evidence_scope, expert_context)
        history_context = self._history_context(history)
        repair = (
            f"\n\n上一次输出未通过可信校验（{repair_reason}）。请重新完整回答，不能复述上次草稿。"
            if repair_reason
            else ""
        )
        user = (
            f"命盘：\n{chart_json}\n\n近期对话：\n{history_context or '无'}\n\n"
            f"用户问题：{question}\n\n资料：\n{context}{repair}"
        )
        request_payload, headers, endpoint = self._model_request(system, user)
        logger.info("llm_request_started", extra={"provider": self.model})
        try:
            response = await post_with_retries(
                self.http_client,
                endpoint,
                headers=headers,
                payload=request_payload,
                timeout=self.timeout,
                max_retries=self.max_retries,
                operation=f"llm:{self.model}",
            )
            if self.provider == "openai" and response.status_code in {400, 422}:
                logger.warning(
                    "llm_response_format_fallback",
                    extra={"provider": self.model, "status_code": response.status_code},
                )
                request_payload.pop("response_format", None)
                response = await post_with_retries(
                    self.http_client,
                    endpoint,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Idempotency-Key": str(uuid.uuid4()),
                    },
                    payload=request_payload,
                    timeout=self.timeout,
                    max_retries=self.max_retries,
                    operation=f"llm:{self.model}:compatibility-fallback",
                )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning(
                "llm_request_failed",
                extra={"provider": self.model, "status_code": _response_status(exc)},
            )
            raise UpstreamServiceError() from exc

        payload = self._response_payload(response)
        try:
            content = self._response_content(payload, self.provider)
        except InvalidUpstreamResponseError:
            if not self._anthropic_output_exhausted(payload):
                raise
            retry_max_tokens = min(self.max_tokens * 2, 16_384)
            if retry_max_tokens <= self.max_tokens:
                raise
            logger.warning(
                "llm_output_token_retry",
                extra={"provider": self.model, "max_tokens": retry_max_tokens},
            )
            retry_payload = {**request_payload, "max_tokens": retry_max_tokens}
            retry_headers = {**headers, "Idempotency-Key": str(uuid.uuid4())}
            try:
                response = await post_with_retries(
                    self.http_client,
                    endpoint,
                    headers=retry_headers,
                    payload=retry_payload,
                    timeout=self.timeout,
                    max_retries=self.max_retries,
                    operation=f"llm:{self.model}:output-token-retry",
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                logger.warning(
                    "llm_request_failed",
                    extra={"provider": self.model, "status_code": _response_status(exc)},
                )
                raise UpstreamServiceError() from exc
            payload = self._response_payload(response)
            content = self._response_content(payload, self.provider)
        parsed = self._parse_model_json(content)
        usage_payload = payload.get("usage")
        token_usage = self._token_usage(usage_payload)
        generated = GenerationResult(
            answer=str(parsed.get("answer") or "证据不足，暂时无法回答。"),
            uncertainties=self._string_list(parsed.get("uncertainties")),
            followups=self._string_list(parsed.get("followups")),
            token_usage=token_usage,
        )
        return generated

    async def generate_stream(
        self,
        question: str,
        chart: ChartFacts,
        hits: list[RetrievalHit],
        on_answer_chunk: Callable[[str], Awaitable[None]],
        *,
        school: str = "基础共识",
        evidence_scope: str = "reviewed_only",
        history: list[dict[str, str]] | None = None,
        expert_context: str = "",
    ) -> GenerationResult:
        generated = await self.generate(
            question,
            chart,
            hits,
            school=school,
            evidence_scope=evidence_scope,
            history=history,
            expert_context=expert_context,
        )
        await on_answer_chunk(generated.answer)
        return generated

    @staticmethod
    def _evidence_context(hits: list[RetrievalHit]) -> str:
        def bounded_text(hit: RetrievalHit) -> str:
            text = hit.document.text
            return text if len(text) <= 1400 else text[:1397].rstrip() + "……"

        review_labels = {
            "reviewed": "已复核",
            "machine_verified": "机器校勘",
            "draft": "草稿",
            "retired": "已停用",
        }
        verification_labels = {
            "unverified": "未核验",
            "single_source_integrity": "单源完整性",
            "multi_source_alignment": "多源对齐",
            "human_review": "人工复核",
        }
        return "\n\n".join(
            f"[{index}] 第{hit.document.layer}层 · {hit.document.title}\n"
            f"状态：{review_labels.get(hit.document.review_status, '未核验')}；"
            f"校验：{verification_labels.get(hit.document.verification_level, '未核验')}；"
            f"置信度：{hit.document.confidence:.2f}\n"
            f"风险提示：{hit.document.warning or '无'}\n"
            f"来源：{hit.document.source}\n"
            f"追溯：{', '.join(hit.document.trace_refs) or hit.document.id}\n"
            f"内容：{bounded_text(hit)}"
            for index, hit in enumerate(hits, start=1)
        )

    @staticmethod
    def _system_prompt(school: str, evidence_scope: str, expert_context: str = "") -> str:
        evidence_label = "含机器校勘预览" if evidence_scope == "personal_preview" else "仅人工审核"
        prompt = (
            f"你是采用“{school}”范围的研究助手，当前证据范围为“{evidence_label}”。"
            "命盘数据中的四柱、月令、透干、根气、合冲刑害由程序计算。"
            "请在内部按以下顺序完成分析：一、核对与问题有关的命盘信息；二、提出可能解释及其依据；"
            "三、逐项套用资料中的前提和条件；四、检查破格、救应、例外、力量与位置先后；"
            "五、形成结论。以上分析步骤不得展示给用户；最终 answer 只保留用户需要的结论、"
            "具体表现、时间趋势和建议。知识性结论必须紧跟有效引用 [n]，不得引用不存在的编号。"
            "answer 必须使用清晰的 Markdown 排版：每个主要部分用“## 标题”单独成行；"
            "每段最多三句话；并列判断、准备建议和注意事项使用“- ”或“1. ”列表且每项单独成行；"
            "标题、段落和列表之间留一个空行。禁止把“一、二、三”多个部分连续写在同一段。"
            "标题使用“结论”“具体表现”“时间趋势”“建议”等自然名称，不得使用“命盘程序事实”"
            "“候选依据与逐项检验”“条件套用”“检查与推演”或其他分析过程名称。"
            "回答必须使用自然中文，不得输出 JSON 字段名、审核状态、内部枚举值或“候选、未定成格”"
            "等系统措辞。资料审核层级只在证据面板展示，正文不要复述。"
            "方括号只允许写资料编号 [1]、[2]，不得写文字标签引用。"
            "birth 中的 date、time、timezone、gender 均为用户已经提供的输入，不得再次声称缺少"
            "出生日期、出生时间或性别，也不得要求用户重复提供。luck 是程序计算的大运事实；用户询问"
            "当前或下一大运时，须直接使用命盘中已经计算出的当前大运与下一步大运回答，"
            "不得把第一步大运误称为下一步大运。出生地未采集，但除非用户主动要求真太阳时校正，"
            "不得将出生地作为回答前提。"
            "允许在证据范围内讨论财富、婚姻、感情、健康、事业、考试、诉讼和具体事件，不得只因题材"
            "敏感而拒绝回答，但必须遵守以下边界：过往只作回溯验证，先列命局条件与对应岁运，再说明与"
            "实际事件是否吻合，不得根据用户经历反向修改规则；未来只给条件式趋势、时间窗口和不确定性，"
            "不得承诺具体年份必然发财、升职、结婚、离婚或发生灾祸。健康内容只能说明传统文献观点，"
            "不得诊断疾病、预测寿命或死亡时间、安排手术、建议治疗用药或停药；用户提到现实症状时应"
            "建议咨询有资质的医生。古籍中的身份、性别、婚姻和疾病断语要标明历史语境，不得直接复述为"
            "现实结论。命例只用于展示规则应用，不得因一柱、一字或相似十神套用古人结果。"
            "应结合本轮命盘与资料给出直接、完整的判断；资料未明确覆盖的内容必须列为不确定性，"
            "不要反复插入“程序事实”“命理推演”等技术免责声明。"
            "只返回 JSON，字段为 answer、"
            "uncertainties、followups。"
        )
        if expert_context:
            prompt += f"\n\n专家方法约束：\n{expert_context}"
        return prompt

    @staticmethod
    def _history_context(history: list[dict[str, str]]) -> str:
        lines = []
        for item in history[-8:]:
            if item.get("role") not in {"user", "assistant"} or not item.get("content"):
                continue
            speaker = "用户" if item.get("role") == "user" else "助手"
            lines.append(f"{speaker}：{item.get('content', '')[:1200]}")
        return "\n".join(lines)

    @staticmethod
    def _chart_context(chart: ChartFacts) -> str:
        payload = chart.model_dump(mode="json", exclude={"birth": {"name"}})
        payload.pop("pattern_candidates", None)
        relations = payload.get("structural_relations")
        if isinstance(relations, list):
            for relation in relations:
                if not isinstance(relation, dict):
                    continue
                relation.pop("note", None)
                label = relation.get("label")
                if isinstance(label, str):
                    relation["label"] = label.replace("候选", "")
        luck = payload.get("luck")
        if isinstance(luck, dict):
            cycles = luck.get("cycles")
            if isinstance(cycles, list):
                for cycle in cycles:
                    if isinstance(cycle, dict):
                        cycle.pop("status", None)
            for key in ("current_cycle", "next_cycle"):
                cycle = luck.get(key)
                if isinstance(cycle, dict):
                    cycle.pop("status", None)
        return json.dumps(payload, ensure_ascii=False)

    async def _consume_model_stream(
        self,
        endpoint: str,
        headers: dict[str, str],
        payload: dict[str, object],
        on_answer_chunk: Callable[[str], Awaitable[None]],
    ) -> tuple[str, object]:
        content = ""
        emitted_answer = ""
        usage: object = None
        async with self.http_client.stream(
            "POST", endpoint, headers=headers, json=payload, timeout=self.timeout
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    event = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise InvalidUpstreamResponseError() from exc
                if not isinstance(event, dict):
                    continue
                event_usage = event.get("usage")
                if isinstance(event_usage, dict):
                    usage = {**(usage if isinstance(usage, dict) else {}), **event_usage}
                message = event.get("message")
                if isinstance(message, dict) and isinstance(message.get("usage"), dict):
                    usage = {
                        **(usage if isinstance(usage, dict) else {}),
                        **message["usage"],
                    }
                delta = self._stream_text_delta(event)
                if not delta:
                    continue
                content += delta
                answer_prefix = self._partial_answer(content)
                if len(answer_prefix) > len(emitted_answer):
                    await on_answer_chunk(answer_prefix[len(emitted_answer) :])
                    emitted_answer = answer_prefix
        return content, usage

    def _stream_text_delta(self, event: dict[str, object]) -> str:
        if self.provider == "anthropic":
            delta = event.get("delta")
            if isinstance(delta, dict) and delta.get("type") == "text_delta":
                text = delta.get("text")
                return text if isinstance(text, str) else ""
            return ""
        choices = event.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            return ""
        delta = choices[0].get("delta")
        if not isinstance(delta, dict):
            return ""
        text = delta.get("content")
        return text if isinstance(text, str) else ""

    @staticmethod
    def _partial_answer(content: str) -> str:
        match = re.search(r'"answer"\s*:\s*"', content)
        if not match:
            return ""
        result: list[str] = []
        index = match.end()
        escapes = {
            "n": "\n",
            "r": "\r",
            "t": "\t",
            "b": "\b",
            "f": "\f",
            '"': '"',
            "\\": "\\",
            "/": "/",
        }
        while index < len(content):
            char = content[index]
            if char == '"':
                break
            if char != "\\":
                result.append(char)
                index += 1
                continue
            if index + 1 >= len(content):
                break
            escaped = content[index + 1]
            if escaped == "u":
                code = content[index + 2 : index + 6]
                if len(code) < 4 or not all(item in "0123456789abcdefABCDEF" for item in code):
                    break
                result.append(chr(int(code, 16)))
                index += 6
                continue
            if escaped not in escapes:
                break
            result.append(escapes[escaped])
            index += 2
        return "".join(result)

    def _model_request(
        self, system: str, user: str
    ) -> tuple[dict[str, object], dict[str, str], str]:
        idempotency_key = str(uuid.uuid4())
        if self.provider == "anthropic":
            payload: dict[str, object] = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "thinking": {"type": "disabled"},
                "system": system,
                "messages": [{"role": "user", "content": user}],
            }
            return (
                payload,
                {
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "Idempotency-Key": idempotency_key,
                },
                f"{self.base_url}/messages"
                if self.base_url.endswith("/v1")
                else f"{self.base_url}/v1/messages",
            )
        return (
            {
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            {
                "Authorization": f"Bearer {self.api_key}",
                "Idempotency-Key": idempotency_key,
            },
            f"{self.base_url}/chat/completions",
        )

    @staticmethod
    def _response_payload(response: httpx.Response) -> dict[str, object]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise InvalidUpstreamResponseError() from exc
        if not isinstance(payload, dict):
            raise InvalidUpstreamResponseError()
        return payload

    @staticmethod
    def _response_content(payload: dict[str, object], provider: str = "openai") -> str:
        if provider == "anthropic":
            content = payload.get("content")
            if not isinstance(content, list):
                raise InvalidUpstreamResponseError()
            texts = [
                block.get("text")
                for block in content
                if isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            ]
            if not texts:
                raise InvalidUpstreamResponseError()
            return "\n".join(texts)
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise InvalidUpstreamResponseError()
        message = choices[0].get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise InvalidUpstreamResponseError()
        return message["content"]

    def _anthropic_output_exhausted(self, payload: dict[str, object]) -> bool:
        if self.provider != "anthropic" or payload.get("stop_reason") != "max_tokens":
            return False
        content = payload.get("content")
        if not isinstance(content, list):
            return False
        return not any(
            isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
            for block in content
        )

    @staticmethod
    def _token_usage(usage_payload: object) -> int | None:
        if not isinstance(usage_payload, dict):
            return None
        total = usage_payload.get("total_tokens")
        if isinstance(total, int):
            return total
        input_tokens = usage_payload.get("input_tokens")
        output_tokens = usage_payload.get("output_tokens")
        if isinstance(input_tokens, int) and isinstance(output_tokens, int):
            return input_tokens + output_tokens
        return None

    @staticmethod
    def _string_list(value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]

    @staticmethod
    def _parse_model_json(content: str) -> dict[str, object]:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.removeprefix("```json").removeprefix("```")
            cleaned = cleaned.removesuffix("```").strip()
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("llm_unstructured_response", extra={"provider": "chat-completions"})
            return {
                "answer": "",
                "uncertainties": ["当前模型未返回结构化 JSON。"],
                "followups": [],
            }
        return parsed if isinstance(parsed, dict) else {"answer": str(parsed)}

    def _extractive_demo(
        self,
        question: str,
        chart: ChartFacts,
        hits: list[RetrievalHit],
        question_policy: QuestionPolicy = "evidence_answer",
    ) -> GenerationResult:
        if not hits:
            return GenerationResult(
                answer="当前知识库没有找到足够依据，暂时不作命理判断。",
                uncertainties=["没有命中的已审核知识卡。"],
                followups=["可以换成一个更具体的基础概念提问。"],
                policy_decision="refuse_no_evidence",
                citations_validated=True,
                uncertainty_validated=True,
            )
        excerpts = []
        for index, hit in enumerate(hits[:3], start=1):
            excerpts.append(f"[{index}] **{hit.document.title}**：{hit.document.text}")
        answer = (
            f"你的日主是{chart.day_master}（{chart.day_master_yin_yang}{chart.day_master_element}）。"
            f"针对“{question}”，当前知识库支持先从以下几点理解：\n\n"
            + "\n\n".join(excerpts)
            + "\n\n以上内容来自当前检索到的证据。"
        )
        return GenerationResult(
            answer=answer,
            uncertainties=["当前为无 API Key 的摘录演示模式，尚未进行模型综合推理。"],
            followups=[
                f"{chart.day_master}日主与其他天干的十神关系如何理解？",
                "月柱在基础解读中通常提供什么背景？",
            ],
            citations_validated=True,
            uncertainty_validated=True,
            question_policy=question_policy,
            model_version="extractive-demo",
        )

    @staticmethod
    def _no_evidence(question_policy: QuestionPolicy) -> GenerationResult:
        return GenerationResult(
            answer="当前所选流派和审核范围内没有找到足够证据，暂时不作命理判断。",
            uncertainties=["没有可用于支持回答的合格证据。"],
            followups=["可以调整流派或证据范围后重新提问。"],
            policy_decision="refuse_no_evidence",
            question_policy=question_policy,
            citations_validated=True,
            uncertainty_validated=True,
            degradation_reason="no_evidence",
            model_version="policy-engine-v1",
        )

    @staticmethod
    def _generation_is_trusted(generated: GenerationResult) -> bool:
        return (
            generated.citations_validated
            and generated.uncertainty_validated
            and generated.policy_decision == "allow"
        )

    @staticmethod
    def _validation_reason(generated: GenerationResult) -> str:
        reasons = []
        if not generated.citations_validated:
            reasons.append("invalid_citations")
        if not generated.uncertainty_validated:
            reasons.append("missing_uncertainty")
        return ",".join(reasons) or "unknown"

    @staticmethod
    def _validate_model_generation(
        generated: GenerationResult, evidence_count: int
    ) -> GenerationResult:
        generated.answer = _humanize_internal_labels(generated.answer)
        generated.uncertainties = [
            _humanize_internal_labels(item) for item in generated.uncertainties
        ]
        generated.followups = [_humanize_internal_labels(item) for item in generated.followups]
        generated.answer = re.sub(
            r"\[([^\]]+)\]",
            lambda match: match.group(0) if match.group(1).isdigit() else f"（{match.group(1)}）",
            generated.answer,
        )
        citations = [int(item) for item in re.findall(r"\[(\d+)\]", generated.answer)]
        bracket_labels = re.findall(r"\[([^\]]+)\]", generated.answer)
        generated.citations_validated = (
            bool(citations)
            and all(1 <= citation <= evidence_count for citation in citations)
            and all(label.isdigit() for label in bracket_labels)
        )
        generated.uncertainty_validated = any(
            item.strip()
            and not _contains_any(
                re.sub(r"\s+", "", item),
                ("没有不确定性", "毫无不确定性", "不存在不确定性", "已经确定"),
            )
            for item in generated.uncertainties
        )
        return generated


def classify_question_policy(_question: str) -> QuestionPolicy:
    return "evidence_answer"


def _humanize_internal_labels(text: str) -> str:
    text = re.sub(r"\bpattern_candidates\b", "格局参考", text, flags=re.IGNORECASE)
    text = re.sub(
        r"[（(][^（）()\n]*(?:machine_verified|机器校勘|未定成格)"
        r"[^（）()\n]*[）)]",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(^|[。！？])[^。！？\n]*(?:machine_verified|机器校勘)"
        r"[^。！？\n]*[。！？]?",
        r"\1",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(?:[,，;；]\s*)?(?:状态|status)\s*(?:为|是|[:：=])?\s*"
        r"(?:current|future|past)\b",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\bcurrent_cycle\b", "当前大运", text, flags=re.IGNORECASE)
    text = re.sub(r"\bnext_cycle\b", "下一步大运", text, flags=re.IGNORECASE)
    text = re.sub(r"未定成格|(?:结构关系)?候选", "", text)
    text = re.sub(r"[（(]\s*[）)]", "", text)
    return re.sub(r"([，；])\s*[，；]", r"\1", text)


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _response_status(exc: httpx.HTTPError) -> int | None:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code
    return None
