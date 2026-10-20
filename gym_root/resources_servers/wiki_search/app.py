import asyncio
import re

from fastapi import FastAPI, Request, Response
from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.server_utils import request
from pydantic import BaseModel, ConfigDict

from wiki_search.scoring import em_check, extract_solution


class WikiSearchResourcesServerConfig(BaseResourcesServerConfig):
    retriever_base_url: str
    top_k: int = 3

    max_concurrency: int = 64
    max_queries_per_call: int = 3
    # Bounds the whole retriever call: nemo_gym's request() retries connection
    # errors forever (fine for in-mesh servers, fatal for an external retriever
    # that is down), so wait_for is the only reliable ceiling.
    retriever_timeout_s: float = 30.0
    # Context safety: episodes must stay inside the model window, and the
    # policy controls how many parallel tool calls it emits per turn - without
    # these caps observation volume is unbounded and eventually exceeds the
    # engine's max context (a hard 400 from the SGLang server).
    # An episode makes up to max_turns model calls (5 in
    # configs/wiki_search.yaml) and the budget check overshoots by one call,
    # so the hard bound is base + max_turns x max_new_tokens + (budget + one
    # call) - keep the sum well under the engine context length.
    max_doc_chars: int = 1200  # per retrieved passage
    max_result_chars: int = 3000  # per /search call
    episode_obs_char_budget: int = 8000  # per episode, tracked via cookie


class SearchRequest(BaseModel):
    query_list: list[str]


class SearchResponse(BaseModel):
    result: str


class WikiSearchVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    expected_answer: str | list[str]


class WikiSearchResourcesServer(SimpleResourcesServer):
    config: WikiSearchResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/search")(self.search)
        return app

    async def search(self, body: SearchRequest, request_ctx: Request, response_ctx: Response) -> SearchResponse:
        used = 0
        try:
            used = int(request_ctx.cookies.get("ws_obs_chars", "0"))
        except (TypeError, ValueError):
            pass
        if used >= self.config.episode_obs_char_budget:
            return SearchResponse(
                result="Search budget for this task is exhausted. Answer with the information you already have."
            )
        try:
            async with self._semaphore:
                resp = await asyncio.wait_for(
                    request(
                        "POST",
                        f"{self.config.retriever_base_url}/retrieve",
                        json={
                            "queries": body.query_list[: self.config.max_queries_per_call],
                            "topk": self.config.top_k,
                            "return_scores": True,
                        },
                    ),
                    timeout=self.config.retriever_timeout_s,
                )
                payload = await resp.json()
                raw = payload.get("result", [])
            if not raw:
                return SearchResponse(result="No search results found.")
            result = "\n---\n".join(passages2string(r, self.config.max_doc_chars) for r in raw)
            if len(result) > self.config.max_result_chars:
                result = result[: self.config.max_result_chars] + "\n[results truncated]"
            response_ctx.set_cookie("ws_obs_chars", str(used + len(result)))
            return SearchResponse(result=result)
        except Exception as e:
            # Degraded observation for the model, machine-readable flag for the
            # harness: the header value is a coarse category (exception class);
            # the exact message stays in the observation text for trajectory debug.
            response_ctx.headers["X-Env-Tool-Error"] = type(e).__name__
            return SearchResponse(result=f"Search error: {str(e) or type(e).__name__}")

    async def verify(self, body: WikiSearchVerifyRequest) -> BaseVerifyResponse:
        texts = []
        for item in body.response.output:
            for block in getattr(item, "content", None) or []:
                if getattr(block, "type", None) == "output_text":
                    texts.append(block.text)
        full_text = "\n".join(texts)
        full_text = re.sub(r"<think>.*?</think>", "", full_text, flags=re.DOTALL)
        answer = extract_solution(full_text)
        reward = float(em_check(answer, body.expected_answer)) if answer is not None else 0.0
        return BaseVerifyResponse(**body.model_dump(), reward=reward)

    def model_post_init(self, context):
        self._semaphore = asyncio.Semaphore(self.config.max_concurrency)


def passages2string(retrieval_result, max_doc_chars: int = 0) -> str:
    format_reference = ""
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item["document"]["contents"]
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        if max_doc_chars and len(text) > max_doc_chars:
            text = text[:max_doc_chars] + "..."
        format_reference += f"Doc {idx + 1} (Title: {title})\n{text}\n\n"
    return format_reference.strip()


if __name__ == "__main__":
    WikiSearchResourcesServer.run_webserver()
