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
from bazi_api.core.errors import (
    InvalidUpstreamResponseError,
    ServiceUnavailableError,
    UpstreamServiceError,
)
from bazi_api.modules.charts.schemas import ChartFacts, TopicFactPack

from .http import post_with_retries

logger = logging.getLogger(__name__)

PolicyDecision = Literal[
    "allow",
    "refuse_no_evidence",
    "refuse_invalid_citations",
]
QuestionPolicy = Literal["evidence_answer", "direct_answer"]


@dataclass
class GenerationResult:
    answer: str
    uncertainties: list[str]
    followups: list[str]
    token_usage: int | None = None
    policy_decision: PolicyDecision = "allow"
    question_policy: QuestionPolicy = "direct_answer"
    citations_validated: bool = False
    citation_format_validated: bool = False
    evidence_validated: bool = False
    safety_validated: bool = False
    uncertainty_validated: bool = False
    verification_failure: str = ""
    degradation_reason: str = ""
    model_version: str = ""
    prompt_version: str = "complete-evidence-v6"


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

    async def generate_direct(
        self,
        question: str,
        chart: ChartFacts,
        topic_pack: TopicFactPack | None,
        *,
        school: str = "基础共识",
        history: list[dict[str, str]] | None = None,
        expert_context: str = "",
    ) -> GenerationResult:
        if not self.api_key:
            raise ServiceUnavailableError(
                "未配置模型 Key，无法直接解读；请在后端 .env 配置 OPENAI_API_KEY"
                " 或 ANTHROPIC_AUTH_TOKEN"
            )
        repair_reason = ""
        token_usage = 0
        for _attempt in range(2):
            generated = await self._generate_direct_once(
                question,
                chart,
                topic_pack,
                school=school,
                history=history or [],
                expert_context=expert_context,
                repair_reason=repair_reason,
            )
            generated.model_version = self.model
            generated.prompt_version = "direct-v1"
            generated.question_policy = "direct_answer"
            token_usage += generated.token_usage or 0
            if self._direct_output_is_trusted(generated):
                generated.token_usage = token_usage or None
                return generated
            repair_reason = self._direct_failure_reason(generated)
            logger.warning(
                "llm_direct_repair", extra={"provider": self.model, "error_code": repair_reason}
            )
        fallback = GenerationResult(
            answer="本次直接解读未通过内容校验，暂不作具体判断，请重试。",
            uncertainties=["答案未通过安全与完整性校验。"],
            followups=[],
            token_usage=token_usage or None,
            degradation_reason="direct_validation_failed",
            model_version=self.model,
            prompt_version="direct-v1",
            question_policy="direct_answer",
        )
        return fallback

    async def _generate_direct_once(
        self,
        question: str,
        chart: ChartFacts,
        topic_pack: TopicFactPack | None,
        *,
        school: str,
        history: list[dict[str, str]],
        expert_context: str,
        repair_reason: str,
    ) -> GenerationResult:
        system = self._direct_system_prompt(school, topic_pack, expert_context)
        topic_context = (
            json.dumps(
                {
                    "话题": topic_pack.label,
                    "程序计算事实": topic_pack.facts,
                },
                ensure_ascii=False,
            )
            if topic_pack
            else "无（自由提问，未选择话题）"
        )
        repair = (
            f"\n\n上一次输出未通过校验（{repair_reason}）。请重新完整回答，不能复述上次草稿。"
            if repair_reason
            else ""
        )
        user = (
            f"命盘：\n{self._chart_context(chart)}\n\n话题事实：\n{topic_context}\n\n"
            f"近期对话：\n{self._history_context(history) or '无'}\n\n"
            f"用户问题：{question}{repair}"
        )
        request_payload, headers, endpoint = self._model_request(system, user)
        logger.info("llm_direct_request_started", extra={"provider": self.model})
        try:
            response = await post_with_retries(
                self.http_client,
                endpoint,
                headers=headers,
                payload=request_payload,
                timeout=self.timeout,
                max_retries=self.max_retries,
                operation=f"llm:{self.model}:direct",
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
                    operation=f"llm:{self.model}:direct:compatibility-fallback",
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
        generated = GenerationResult(
            answer=str(parsed.get("answer") or ""),
            uncertainties=self._string_list(parsed.get("uncertainties")),
            followups=self._string_list(parsed.get("followups")),
            token_usage=self._token_usage(payload.get("usage")),
        )
        generated.answer = _humanize_internal_labels(generated.answer)
        generated.uncertainties = [
            _humanize_internal_labels(item) for item in generated.uncertainties
        ]
        return generated

    @staticmethod
    def _direct_output_is_trusted(generated: GenerationResult) -> bool:
        if not generated.answer.strip():
            return False
        if not any(item.strip() for item in generated.uncertainties):
            return False
        return not _explicit_safety_violation(generated)

    @staticmethod
    def _direct_failure_reason(generated: GenerationResult) -> str:
        if not generated.answer.strip():
            return "empty_answer"
        if not any(item.strip() for item in generated.uncertainties):
            return "missing_uncertainty"
        return "unsafe_output"

    @staticmethod
    def _direct_system_prompt(
        school: str, topic_pack: TopicFactPack | None, expert_context: str = ""
    ) -> str:
        topic_line = (
            f"本次用户选择了「{topic_pack.label}」话题，回答须优先围绕该话题展开。"
            if topic_pack
            else "用户未选择固定话题，按问题本身解读。"
        )
        prompt = (
            f"你是采用“{school}”范围的命理解读助手。"
            "命盘数据中的四柱、月令、透干、根气、大运、流年干支由程序计算，直接采信，不要自行另排。"
            f"{topic_line}"
            "话题事实是程序按话题补算的确定性事实（十神位置、干支计数、神煞查表），照实陈述；"
            "干支计数只是数量对比，不代表身强身弱结论，身强身弱由你结合月令与整体结构判断。"
            "用户一次询问多个方面、比较对象或时间段时，逐项回应；某项信息不足时在答案或不确定性中点名说明。"
            "分析步骤不得展示给用户；answer 只保留用户需要的结论、具体表现、时间趋势和建议。"
            "answer 必须使用清晰的 Markdown 排版：每个主要部分用“## 标题”单独成行；每段最多三句话；"
            "并列判断和建议使用“- ”或“1. ”列表且每项单独成行；标题、段落和列表之间留一个空行。"
            "回答必须使用自然中文，不得输出 JSON 字段名或内部枚举值。"
            "birth 中的 date、time、timezone、gender 均为用户已经提供的输入，不得再次声称缺少"
            "出生日期、出生时间或性别。"
            "luck 是程序计算的大运事实；用户询问当前或下一大运时直接使用，"
            "不得把第一步大运误称为下一步大运。出生地未采集，除非用户主动要求真太阳时校正，"
            "不得将出生地作为回答前提。"
            "允许讨论财富、婚姻、感情、健康、事业、考试、诉讼和具体事件，不得只因题材敏感而拒绝回答，"
            "但必须遵守以下边界：过往只作回溯验证，先列命局条件与对应岁运，再说明与实际事件是否吻合；"
            "未来只给条件式趋势、时间窗口和不确定性，不得承诺具体年份必然发财、升职、结婚、离婚或发生灾祸。"
            "健康内容只能说明传统文献观点与生活习惯层面的提醒，不得诊断疾病、预测寿命或死亡时间、"
            "安排手术、建议治疗用药或停药；用户提到现实症状时应建议咨询有资质的医生。"
            "古籍中的身份、性别、婚姻和疾病断语要标明历史语境，不得直接复述为现实结论。"
            "引用传统典籍观点时可以写书名（如《滴天髓》《子平真诠》），"
            "但不得编造具体篇章、条文编号或原文引文。"
            "流年表为程序计算的干支，用于时间趋势时结合大运与命局条件说明，不得断言某年必然发生某事。"
            "只返回 JSON，字段为 answer、uncertainties、followups。"
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

def _explicit_safety_violation(generated: GenerationResult) -> bool:
    # These obvious cases cannot be overridden by a permissive verifier response.
    patterns = (
        r"(?:你|您|命主|患者|孩子).{0,16}(?:患有|患了|得了|确诊|诊断为)",
        r"(?:建议|应该|应当|必须|立即|请|可以|需要|务必|尽快|^).{0,12}"
        r"(?:停药|停用|停止服用|停止用药|停止治疗|服用|加大剂量|减药|减量|动手术|进行手术)",
        r"(?:每日|每天|每次).{0,8}(?:服用|口服|注射)",
        r"(?:一定|必然|肯定|必定|必会|必将|必有|已经(?:可以)?确定).{0,16}"
        r"(?:发财|中奖|大奖|升职|升官|结婚|离婚|成功|上岸|灾祸|重病|确诊|死亡|去世)",
        r"(?:发财|中奖|大奖|升职|结婚|离婚|成功).{0,12}已经(?:可以)?确定",
        r"(?:你|您|命主).{0,20}(?:会在|将在).{0,20}(?:死|去世|身亡)",
        r"(?:你|您|命主).{0,20}(?:只能活|还能活).{0,12}(?:岁|年)",
    )
    text = "\n".join([generated.answer, *generated.uncertainties, *generated.followups])
    text = re.sub(r"[\s*_`#\u200b-\u200f\ufeff]", "", text)
    for clause in re.split(r"[，,。！？!?；;\n]|但是|而是", text):
        for pattern in patterns:
            for match in re.finditer(pattern, clause):
                prefix = clause[max(0, match.start() - 8) : match.start()]
                matched = match.group(0)
                if re.search(r"(?:不应|不能|不可|不要|不得|禁止|避免|切勿|不建议)", prefix):
                    continue
                if re.search(r"(?:不应|不能|不可|不要|不得|禁止|切勿|不建议|没有|未被)", matched):
                    continue
                return True
    return False


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


def _response_status(exc: httpx.HTTPError) -> int | None:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code
    return None
