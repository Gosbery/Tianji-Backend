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
    "refuse_high_risk",
    "refuse_no_evidence",
    "refuse_invalid_citations",
    "refuse_unsafe_output",
]
QuestionPolicy = Literal["evidence_answer", "explain_boundary", "hard_refusal"]


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
        self, question: str, chart: ChartFacts, hits: list[RetrievalHit]
    ) -> GenerationResult:
        if not self.api_key:
            logger.info("llm_extractive_demo", extra={"provider": "extractive-demo"})
            return self._extractive_demo(question, chart, hits)

        context = "\n\n".join(
            f"[{index}] 第{hit.document.layer}层 · {hit.document.title}\n"
            f"来源：{hit.document.source}\n"
            f"追溯：{', '.join(hit.document.trace_refs) or hit.document.id}\n"
            f"内容：{hit.document.text}"
            for index, hit in enumerate(hits, start=1)
        )
        chart_json = chart.model_dump_json(exclude={"birth": {"name"}})
        system = (
            "你是一名八字研究助手。请结合命盘 JSON、用户问题和资料片段进行完整分析。"
            "只返回 JSON，字段为 answer、uncertainties、followups。"
        )
        user = f"命盘：\n{chart_json}\n\n用户问题：{question}\n\n资料：\n{context}"
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
            answer=str(parsed.get("answer", "证据不足，暂时无法回答。")),
            uncertainties=self._string_list(parsed.get("uncertainties")),
            followups=self._string_list(parsed.get("followups")),
            token_usage=token_usage,
        )
        return self._validate_model_generation(generated)

    async def generate_stream(
        self,
        question: str,
        chart: ChartFacts,
        hits: list[RetrievalHit],
        on_answer_chunk: Callable[[str], Awaitable[None]],
    ) -> GenerationResult:
        if not self.api_key:
            generated = self._extractive_demo(question, chart, hits)
            await on_answer_chunk(generated.answer)
            return generated

        context = "\n\n".join(
            f"[{index}] 第{hit.document.layer}层 · {hit.document.title}\n"
            f"来源：{hit.document.source}\n"
            f"追溯：{', '.join(hit.document.trace_refs) or hit.document.id}\n"
            f"内容：{hit.document.text}"
            for index, hit in enumerate(hits, start=1)
        )
        chart_json = chart.model_dump_json(exclude={"birth": {"name"}})
        system = (
            "你是一名八字研究助手。请结合命盘 JSON、用户问题和资料片段进行完整分析。"
            "只返回 JSON，字段为 answer、uncertainties、followups，并且 answer 必须是第一个字段。"
        )
        user = f"命盘：\n{chart_json}\n\n用户问题：{question}\n\n资料：\n{context}"
        request_payload, headers, endpoint = self._model_request(system, user)
        request_payload["stream"] = True
        if self.provider == "openai":
            request_payload["stream_options"] = {"include_usage": True}

        try:
            content, usage_payload = await self._consume_model_stream(
                endpoint, headers, request_payload, on_answer_chunk
            )
        except httpx.HTTPStatusError as exc:
            if self.provider != "openai" or exc.response.status_code not in {400, 422}:
                raise UpstreamServiceError() from exc
            request_payload.pop("response_format", None)
            content, usage_payload = await self._consume_model_stream(
                endpoint,
                {**headers, "Idempotency-Key": str(uuid.uuid4())},
                request_payload,
                on_answer_chunk,
            )
        except httpx.HTTPError as exc:
            raise UpstreamServiceError() from exc

        parsed = self._parse_model_json(content)
        generated = GenerationResult(
            answer=str(parsed.get("answer", "证据不足，暂时无法回答。")),
            uncertainties=self._string_list(parsed.get("uncertainties")),
            followups=self._string_list(parsed.get("followups")),
            token_usage=self._token_usage(usage_payload),
        )
        return self._validate_model_generation(generated)

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
        self, question: str, chart: ChartFacts, hits: list[RetrievalHit]
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
        )

    @staticmethod
    def _validate_model_generation(generated: GenerationResult) -> GenerationResult:
        citations = [int(item) for item in re.findall(r"\[(\d+)\]", generated.answer)]
        generated.citations_validated = bool(citations)
        generated.uncertainty_validated = bool(generated.uncertainties)
        return generated


_HIGH_RISK_TERMS = (
    "癌",
    "肿瘤",
    "病",
    "健康",
    "寿命",
    "死亡",
    "死",
    "归西",
    "离世",
    "去世",
    "活多久",
    "灾祸",
    "受伤",
    "手术",
    "财",
    "赚钱",
    "赚多少",
    "收入",
    "彩票",
    "投资",
    "亏损",
    "升职",
    "失业",
    "职业",
    "职位",
    "考试",
    "录取",
    "上岸",
    "婚",
    "配偶",
    "夫妻",
    "感情",
    "恋爱",
    "姻缘",
    "官司",
    "诉讼",
    "大牢",
    "牢狱",
    "坐牢",
    "蹲牢",
    "判刑",
    "入狱",
    "拘留",
    "运势",
    "好运",
    "倒霉",
    "吉凶",
)
_PERSONAL_TERMS = (
    "我",
    "咱",
    "你会",
    "你将",
    "你在",
    "你的婚",
    "你的财",
    "他会",
    "他将",
    "他在",
    "他的婚",
    "他的财",
    "她会",
    "她将",
    "她在",
    "她的婚",
    "她的财",
    "本人",
    "自己",
    "这个人",
    "某人",
    "对象",
    "我们家",
    "家人",
    "家里人",
    "孩子",
    "父母",
    "伴侣",
    "老公",
    "老婆",
    "男友",
    "女友",
)
_SPECIFIC_TERMS = (
    "这场",
    "这次",
    "本次",
    "眼下",
)
_CONCRETE_PREDICTION_TERMS = (
    "断定",
    "断言",
    "预测",
    "预言",
    "算出",
    "一定",
    "肯定",
    "必然",
    "必定",
    "铁定",
    "注定",
    "保证",
    "稳赢",
    "胜券在握",
    "十拿九稳",
    "板上钉钉",
    "成定局",
    "哪年",
    "何年",
    "哪一年",
    "何时",
    "什么时候",
    "啥时候",
    "几时",
    "哪天",
    "何日",
    "几岁",
    "具体金额",
    "具体日期",
    "结果",
    "结局",
    "成败",
    "成功",
    "失败",
    "顺利",
    "不顺",
    "吉凶",
    "祸福",
    "发财",
    "暴富",
    "赚多少钱",
    "升职",
    "失业",
    "离婚",
    "婚变",
    "破裂",
    "得病",
    "患病",
    "重病",
    "归西",
    "离世",
    "去世",
    "坐牢",
    "蹲大牢",
    "入狱",
    "判刑",
)
_DIRECT_REQUEST_TERMS = (
    "请断定",
    "请预测",
    "请算",
    "帮我算",
    "告诉我",
    "算一算",
    "算算",
    "给我断",
    "替我看",
)
_RESEARCH_TERMS = (
    "原文",
    "古籍",
    "文本",
    "章节",
    "概念",
    "定义",
    "含义",
    "意思",
    "解释",
    "理解",
    "区别",
    "关系",
    "条件",
    "出处",
    "注释",
    "依据",
    "证据",
    "语境",
    "用法",
    "校勘",
    "版本",
    "历史",
    "是什么",
    "有哪些",
    "哪一柱",
    "哪章",
    "第几章",
    "哪一篇",
    "位于哪篇",
    "位于哪一篇",
    "何处",
    "哪里",
    "哪一段",
    "谈到",
    "记载",
    "出现",
    "几个",
)
_MAPPING_TERMS = (
    "一定",
    "必然",
    "断定",
    "断言",
    "预测",
    "推断",
    "直接判断",
    "导致",
    "就会",
    "就必",
    "解释成",
    "等于",
    "代表",
    "表示",
    "说明",
    "翻译成",
    "保证",
)
_SINGLE_BASIS_TERMS = ("只凭", "仅凭", "单凭", "只看", "看到")
_BAZI_SYMBOL_TERMS = (
    "八字",
    "命盘",
    "命里",
    "四柱",
    "五行",
    "阴阳",
    "天干",
    "地支",
    "支",
    "日主",
    "月令",
    "生克",
    "生剋",
    "旺衰",
    "身强",
    "身弱",
    "十神",
    "比肩",
    "劫财",
    "食神",
    "伤官",
    "偏财",
    "正财",
    "七杀",
    "正官",
    "偏印",
    "正印",
    "用神",
    "格局",
    "行运",
    "星辰",
    "神煞",
    "四吉神",
    "四凶神",
    "阳刃",
    "建禄",
    "月劫",
    "六亲",
    "外格",
    "喜神",
    "喜忌",
    "口诀",
    "木",
    "火",
    "土",
    "金",
    "水",
)
_META_QUESTION_TERMS = (
    "是否",
    "能否",
    "是不是",
    "可不可以",
    "可以不可以",
    "能不能",
    "会不会",
    "等不等于",
    "为什么不能",
    "为何不能",
    "应该避免",
)
_PREDICTION_STRUCTURE = re.compile(
    r"(?:哪|何|啥).{0,3}(?:年|月|日|天|时|时候)|"
    r"几(?:年|岁|时)|(?:19|20)\d{2}|今年|明年|后年|未来|以后|将来|"
    r"(?:会不会|能不能|是否|能否)|(?:会|能|要|将).{0,18}(?:吗|么|不|$)|"
    r"(?:多少|几多).{0,5}(?:钱|元|块|万|百万|千万|亿)?|"
    r"确定|肯定|必然|注定|保证"
)
_ASSERTIVE_PREDICTION = re.compile(
    r"(?:今年|明年|后年|未来|以后|将来|(?:19|20)\d{2}).{0,24}"
    r"(?:确定|肯定|一定|必然|注定|保证|将会|会在|能成|必成)"
)
_OUTPUT_BOUNDARY_TERMS = (
    "不能",
    "不可",
    "不应",
    "无法",
    "不宜",
    "不等于",
    "不代表",
    "不足以",
    "不支持",
    "不要",
    "避免",
    "禁止",
    "尚不能",
    "并非",
    "未必",
    "不作",
    "不做",
    "不用于",
    "不意味着",
    "不得",
)
_OUTPUT_PERSONAL_TERMS = _PERSONAL_TERMS + (
    "你",
    "您",
    "命主",
    "此人",
    "本命",
    "本盘",
    "该命",
)


def classify_question_policy(_question: str) -> QuestionPolicy:
    """Compatibility hook; prediction policy checks are disabled."""

    return "evidence_answer"


def requires_refusal(_question: str) -> bool:
    return False


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _response_status(exc: httpx.HTTPError) -> int | None:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code
    return None
