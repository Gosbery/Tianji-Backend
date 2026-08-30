from __future__ import annotations

import json
from dataclasses import dataclass

import httpx

from bazi_api.core.config import Settings
from bazi_api.modules.charts.schemas import ChartFacts
from bazi_api.modules.retrieval.schemas import RetrievalHit


@dataclass
class GenerationResult:
    answer: str
    uncertainties: list[str]
    followups: list[str]
    token_usage: int | None = None


class AnswerGenerator:
    def __init__(self, settings: Settings) -> None:
        self.api_key = settings.openai_api_key
        self.base_url = settings.openai_base_url.rstrip("/")
        self.model = settings.openai_chat_model

    async def generate(
        self, question: str, chart: ChartFacts, hits: list[RetrievalHit]
    ) -> GenerationResult:
        if not self.api_key:
            return self._extractive_demo(question, chart, hits)

        context = "\n\n".join(
            f"[{index}] {hit.document.title}\n"
            f"来源：{hit.document.source}\n"
            f"内容：{hit.document.text}"
            for index, hit in enumerate(hits, start=1)
        )
        chart_json = chart.model_dump_json(exclude={"birth": {"name"}})
        system = (
            "你是一名冷静的八字研究助手。命盘 JSON 是确定性事实，资料片段是可引用解释。"
            "禁止补造命盘信息，禁止把倾向写成确定事件，健康、财富、婚姻问题必须说明边界。"
            "每个实质性判断都用 [1] 形式引用资料。若证据不足，明确说不知道。"
            "只返回 JSON，字段为 answer、uncertainties、followups。"
        )
        user = f"命盘：\n{chart_json}\n\n用户问题：{question}\n\n资料：\n{context}"
        request_payload = {
            "model": self.model,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=request_payload,
            )
            if response.status_code in {400, 422}:
                request_payload.pop("response_format")
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=request_payload,
                )
            response.raise_for_status()
        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
        parsed = self._parse_model_json(content)
        usage = payload.get("usage", {}).get("total_tokens")
        return GenerationResult(
            answer=str(parsed.get("answer", "证据不足，暂时无法回答。")),
            uncertainties=[str(item) for item in parsed.get("uncertainties", [])],
            followups=[str(item) for item in parsed.get("followups", [])],
            token_usage=usage,
        )

    @staticmethod
    def _parse_model_json(content: str) -> dict[str, object]:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.removeprefix("```json").removeprefix("```")
            cleaned = cleaned.removesuffix("```").strip()
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            return {
                "answer": content,
                "uncertainties": ["当前模型未返回结构化 JSON，已保留其原始回答。"],
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
            )
        excerpts = []
        for index, hit in enumerate(hits[:3], start=1):
            excerpts.append(f"[{index}] **{hit.document.title}**：{hit.document.text}")
        answer = (
            f"你的日主是{chart.day_master}（{chart.day_master_yin_yang}{chart.day_master_element}）。"
            f"针对“{question}”，当前知识库支持先从以下几点理解：\n\n"
            + "\n\n".join(excerpts)
            + "\n\n这些是概念层面的解释，不能单凭某一个符号推导具体人生事件。"
        )
        return GenerationResult(
            answer=answer,
            uncertainties=["当前为无 API Key 的摘录演示模式，尚未进行模型综合推理。"],
            followups=[
                f"{chart.day_master}日主与其他天干的十神关系如何理解？",
                "月柱在基础解读中通常提供什么背景？",
            ],
        )
