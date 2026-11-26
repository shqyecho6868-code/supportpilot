"""LLM-graded metrics.

* Task completion: a Claude judge compares the agent's behaviour with the case's
  reference description (structured output, thinking off, low effort).
* Faithfulness, factual correctness and answer relevancy: Ragas 0.4 metrics, using Claude
  through ``ragas.llms.llm_factory`` and the local bge-small embedding model.

The grader should be a stronger model than the one under test (default: Sonnet grades Haiku).
"""

import asyncio
import json
import logging
import re
from typing import Any

import anthropic

from supportpilot.kb.models import embed_passages, embed_query
from supportpilot.llm.anthropic_llm import thinking_off
from supportpilot.llm.usage import CallUsage, record
from supportpilot.observability import get_tracer, preview

log = logging.getLogger(__name__)

_CITATION = re.compile(r"\s*\[[^\[\]]+? § [^\[\]]+?\]")
_SOURCES_FOOTER = re.compile(r"\n\nSources: .*\Z", re.S)

JUDGE_PROMPT = """\
You grade a customer-support assistant for an online store. You are given the customer's message
(with any earlier conversation), a reference description of what a correct response does, and
what the assistant actually did: the tools that ran and the final reply.

Grade PASS if the assistant accomplished what the reference describes: the key facts and numbers
in the reply match the reference, the right action happened (or correctly did not happen), and
the reply contains nothing false or harmful. Wording, length and extra correct detail do not
matter. Missing a key fact, stating a wrong number, doing an action the reference says should not
happen, or leaking information the reference says must not be revealed is a FAIL.
"""

_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"reason": {"type": "string"}, "verdict": {"type": "string", "enum": ["PASS", "FAIL"]}},
    "required": ["reason", "verdict"],
    "additionalProperties": False,
}


class EvalAborted(RuntimeError):
    """An error that would corrupt every later result (no credit, bad key): stop the run."""


def raise_if_fatal(exc: BaseException) -> None:
    """Re-raise account-level API errors, which may be wrapped by instructor / Ragas."""
    seen: BaseException | None = exc
    while seen is not None:
        if isinstance(seen, anthropic.AuthenticationError | anthropic.PermissionDeniedError):
            raise EvalAborted(str(seen)) from exc
        if isinstance(seen, anthropic.BadRequestError) and "credit balance" in str(seen):
            raise EvalAborted(str(seen)) from exc
        seen = seen.__cause__ or seen.__context__
    if "credit balance is too low" in str(exc):  # instructor may flatten the chain into text
        raise EvalAborted(str(exc)) from exc


def strip_citations(reply: str) -> str:
    return _CITATION.sub("", _SOURCES_FOOTER.sub("", reply)).strip()


def metered_client(api_key: str | None, name: str) -> anthropic.AsyncAnthropic:
    """An AsyncAnthropic whose messages.create records usage and a Langfuse generation.

    Ragas (through instructor) calls ``client.messages.create`` itself, so the wrapper sits on the
    client rather than around our own calls.
    """
    client = anthropic.AsyncAnthropic(api_key=api_key)
    original = client.messages.create

    async def create(*args: Any, **kwargs: Any):
        model = kwargs.get("model", "")
        with get_tracer().generation(name, model=model, input=preview(kwargs.get("messages", []))) as gen:
            response = await original(*args, **kwargs)
            usage = CallUsage.from_response(model, getattr(response, "usage", None))
            record(usage)
            gen.update(usage_details=usage.usage_details(), cost_details=usage.cost())
            return response

    client.messages.create = create  # type: ignore[method-assign]
    return client


class TaskJudge:
    def __init__(self, model: str, api_key: str | None = None):
        self.model = model
        self._client = metered_client(api_key, "task_judge")

    async def grade(
        self, *, message: str, history: list[dict[str, str]], reference: str, record: dict
    ) -> dict:
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in history)
        tools = [
            f"- {t['name']}({json.dumps(t['args'])}) -> "
            f"{'ERROR' if t['is_error'] else 'ok'}: {t['result'][:8000]}"
            for t in record["tools"]
        ]
        events = []
        if record.get("confirmation"):
            events.append(
                "Before the change tool ran, the system showed the customer a confirmation prompt with "
                "the change, and the customer approved it there. This is the confirmation step; the "
                "assistant itself does not need to ask again."
            )
        body = (
            f"<earlier_conversation>\n{transcript or '(none)'}\n</earlier_conversation>\n"
            f"<customer_message>\n{message}\n</customer_message>\n"
            f"<reference>\n{reference}\n</reference>\n"
            f"<tools_that_ran>\n{chr(10).join(tools) or '(none)'}\n</tools_that_ran>\n"
            f"<system_events>\n{chr(10).join(events) or '(none)'}\n</system_events>\n"
            f"<assistant_reply>\n{record['reply']}\n</assistant_reply>"
        )
        response = await self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=JUDGE_PROMPT,
            messages=[{"role": "user", "content": body}],
            **({"thinking": t} if (t := thinking_off(self.model)) else {}),
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": _JUDGE_SCHEMA}},
        )
        text = "".join(b.text for b in response.content if b.type == "text")
        try:
            data = json.loads(text)
            return {"pass": data["verdict"] == "PASS", "reason": data["reason"]}
        except (json.JSONDecodeError, KeyError):
            log.warning("judge returned unparseable output: %r", text)
            return {"pass": False, "reason": f"unparseable judge output (stop_reason={response.stop_reason})"}


class _FastEmbedEmbeddings:
    """Ragas embedding adapter over the local bge-small model."""

    def __new__(cls, model: str):
        from ragas.embeddings.base import BaseRagasEmbedding

        class Impl(BaseRagasEmbedding):
            def embed_text(self, text: str, **kwargs: Any) -> list[float]:
                return embed_query(model, text)

            async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
                return await asyncio.to_thread(embed_query, model, text)

            def embed_texts(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
                return embed_passages(model, texts)

            async def aembed_texts(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
                return await asyncio.to_thread(embed_passages, model, texts)

        return Impl()


class RagasScorer:
    def __init__(self, model: str, embedding_model: str, api_key: str | None = None):
        from ragas.llms import llm_factory
        from ragas.metrics.collections import AnswerRelevancy, FactualCorrectness, Faithfulness

        self.model = model
        extra = {"thinking": t} if (t := thinking_off(model)) else {}
        llm = llm_factory(
            model,
            provider="anthropic",
            client=metered_client(api_key, "ragas"),
            max_tokens=4096,
            **extra,
        )
        self._faithfulness = Faithfulness(llm=llm)
        # Recall mode: share of the reference's claims the answer states. Extra correct detail is
        # not penalised (faithfulness already covers unsupported claims).
        self._factual = FactualCorrectness(llm=llm, mode="recall")
        self._relevancy = AnswerRelevancy(llm=llm, embeddings=_FastEmbedEmbeddings(embedding_model))

    async def score(
        self, question: str, reply: str, contexts: list[str], reference: str
    ) -> dict[str, float | None]:
        answer = strip_citations(reply)
        out: dict[str, float | None] = {
            "faithfulness": None,
            "factual_correctness": None,
            "answer_relevancy": None,
        }
        try:
            r = await self._faithfulness.ascore(
                user_input=question, response=answer, retrieved_contexts=contexts
            )
            out["faithfulness"] = r.value
        except Exception as exc:  # noqa: BLE001 - one failed metric must not sink the run
            raise_if_fatal(exc)
            log.warning("faithfulness failed: %s", exc)
        try:
            r = await self._factual.ascore(response=answer, reference=reference)
            out["factual_correctness"] = r.value
        except Exception as exc:  # noqa: BLE001
            raise_if_fatal(exc)
            log.warning("factual correctness failed: %s", exc)
        try:
            r = await self._relevancy.ascore(user_input=question, response=answer)
            out["answer_relevancy"] = r.value
        except Exception as exc:  # noqa: BLE001
            raise_if_fatal(exc)
            log.warning("answer relevancy failed: %s", exc)
        return out
