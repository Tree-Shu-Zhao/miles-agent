from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi import Response
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from pytest import approx

from wiki_search.app import (
    SearchRequest,
    WikiSearchResourcesServer,
    WikiSearchResourcesServerConfig,
    WikiSearchVerifyRequest,
    passages2string,
)
from wiki_search.scoring import em_check, extract_solution, normalize_answer

RETRIEVER_URL = "http://retriever.test:8000"


def _make_server(**config_overrides) -> WikiSearchResourcesServer:
    config = WikiSearchResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        retriever_base_url=RETRIEVER_URL,
        **config_overrides,
    )
    return WikiSearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _ctx():
    """Gym 0.7.0 hands every endpoint the request and the response; search keeps its observation budget in a cookie."""
    return SimpleNamespace(cookies={}), Response()


def _make_message(text: str, msg_id: str = "msg") -> NeMoGymResponseOutputMessage:
    return NeMoGymResponseOutputMessage(
        id=msg_id,
        content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
        role="assistant",
        status="completed",
        type="message",
    )


def _make_response(*texts: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="test",
        created_at=0.0,
        model="test_model",
        object="response",
        output=[_make_message(text, msg_id=f"msg{i}") for i, text in enumerate(texts)],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )


def _verify_request(response: NeMoGymResponse, expected_answer) -> WikiSearchVerifyRequest:
    return WikiSearchVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        response=response,
        expected_answer=expected_answer,
    )


class _FakeAiohttpResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    async def json(self) -> dict:
        return self._payload


# ──────────────────────────────────────────────────────────
# Pure scoring functions
# ──────────────────────────────────────────────────────────


class TestScoring:
    def test_normalize_answer_combined(self) -> None:
        assert normalize_answer("  The   QUICK,   brown FOX!! ") == "quick brown fox"

    def test_em_check_string_gt(self) -> None:
        assert em_check("the Paris.", "paris") == 1

    def test_em_check_list_gt_second_matches(self) -> None:
        assert em_check("Paris", ["London", "the paris"]) == 1

    def test_em_check_mismatch(self) -> None:
        assert em_check("London", ["Paris"]) == 0

    def test_em_check_no_partial_credit(self) -> None:
        assert em_check("Paris is the capital", ["Paris"]) == 0

    def test_extract_solution_none_without_tags(self) -> None:
        assert extract_solution("no tags here") is None

    def test_extract_solution_takes_last(self) -> None:
        assert extract_solution("<answer>wrong</answer> hmm <answer> right </answer>") == "right"

    def test_extract_solution_empty_tag(self) -> None:
        assert extract_solution("<answer></answer>") == ""

    def test_extract_solution_multiline(self) -> None:
        assert extract_solution("<answer>\nParis\n</answer>") == "Paris"


# ──────────────────────────────────────────────────────────
# Observation formatting
# ──────────────────────────────────────────────────────────


class TestPassages2String:
    def test_title_body_split(self) -> None:
        docs = [{"document": {"contents": '"Title A"\nBody line 1\nBody line 2'}, "score": 0.9}]
        result = passages2string(docs)
        assert result == 'Doc 1 (Title: "Title A")\nBody line 1\nBody line 2'

    def test_doc_numbering(self) -> None:
        docs = [
            {"document": {"contents": '"A"\na text'}, "score": 0.9},
            {"document": {"contents": '"B"\nb text'}, "score": 0.8},
        ]
        result = passages2string(docs)
        assert 'Doc 1 (Title: "A")' in result
        assert 'Doc 2 (Title: "B")' in result

    def test_single_line_contents_edge(self) -> None:
        # No newline in contents: whole string becomes the title, no truncation.
        docs = [{"document": {"contents": "OnlyTitleNoBody"}, "score": 0.5}]
        assert passages2string(docs) == "Doc 1 (Title: OnlyTitleNoBody)"

    def test_empty_result(self) -> None:
        assert passages2string([]) == ""


# ──────────────────────────────────────────────────────────
# /search endpoint (retriever mocked)
# ──────────────────────────────────────────────────────────


class TestSearch:
    async def test_success_two_queries(self, monkeypatch) -> None:
        server = _make_server()
        captured = {}

        async def fake_request(method, url, **kwargs):
            captured["method"] = method
            captured["url"] = url
            captured["json"] = kwargs["json"]
            return _FakeAiohttpResponse(
                {
                    "result": [
                        [{"document": {"contents": '"Title A"\nBody A'}, "score": 0.9}],
                        [{"document": {"contents": '"Title B"\nBody B'}, "score": 0.8}],
                    ]
                }
            )

        monkeypatch.setattr("wiki_search.app.request", fake_request)
        response = await server.search(SearchRequest(query_list=["q1", "q2"]), *_ctx())

        assert captured["method"] == "POST"
        assert captured["url"] == f"{RETRIEVER_URL}/retrieve"
        assert captured["json"] == {"queries": ["q1", "q2"], "topk": 3, "return_scores": True}
        assert 'Doc 1 (Title: "Title A")\nBody A' in response.result
        assert 'Doc 1 (Title: "Title B")\nBody B' in response.result
        assert "\n---\n" in response.result

    async def test_query_list_capped(self, monkeypatch) -> None:
        server = _make_server(max_queries_per_call=2)

        async def fake_request(method, url, **kwargs):
            queries = kwargs["json"]["queries"]
            return _FakeAiohttpResponse(
                {"result": [[{"document": {"contents": f'"T{i}"\nbody'}, "score": 1.0}] for i in range(len(queries))]}
            )

        monkeypatch.setattr("wiki_search.app.request", fake_request)
        response = await server.search(SearchRequest(query_list=["q1", "q2", "q3", "q4"]), *_ctx())
        # Only 2 queries reach the retriever -> exactly 2 blocks, 1 separator.
        assert response.result.count("\n---\n") == 1

    async def test_empty_result(self, monkeypatch) -> None:
        server = _make_server()

        async def fake_request(method, url, **kwargs):
            return _FakeAiohttpResponse({"result": []})

        monkeypatch.setattr("wiki_search.app.request", fake_request)
        response = await server.search(SearchRequest(query_list=["q"]), *_ctx())
        assert response.result == "No search results found."

    async def test_hung_retriever_times_out_as_observation(self, monkeypatch) -> None:
        # nemo_gym's request() retries connection errors forever; the wait_for
        # ceiling must convert a hung/down retriever into an error observation.
        import asyncio

        server = _make_server(retriever_timeout_s=0.05)

        async def fake_request(method, url, **kwargs):
            await asyncio.sleep(5)

        monkeypatch.setattr("wiki_search.app.request", fake_request)
        response = await server.search(SearchRequest(query_list=["q"]), *_ctx())
        assert response.result.startswith("Search error:")

    async def test_retriever_error_returned_as_observation(self, monkeypatch) -> None:
        server = _make_server()

        async def fake_request(method, url, **kwargs):
            raise RuntimeError("connection refused")

        monkeypatch.setattr("wiki_search.app.request", fake_request)
        response = await server.search(SearchRequest(query_list=["q"]), *_ctx())
        assert response.result.startswith("Search error:")
        assert "connection refused" in response.result

    async def test_malformed_payload_returned_as_observation(self, monkeypatch) -> None:
        server = _make_server()

        async def fake_request(method, url, **kwargs):
            # Missing the "document" wrapper -> KeyError inside formatting.
            return _FakeAiohttpResponse({"result": [[{"contents": "bare doc"}]]})

        monkeypatch.setattr("wiki_search.app.request", fake_request)
        response = await server.search(SearchRequest(query_list=["q"]), *_ctx())
        assert response.result.startswith("Search error:")

    def test_search_route_registered(self) -> None:
        server = _make_server()
        app = server.setup_webserver()
        assert any(getattr(route, "path", None) == "/search" for route in app.routes)


# ──────────────────────────────────────────────────────────
# verify endpoint
# ──────────────────────────────────────────────────────────


class TestVerify:
    async def test_correct_answer(self) -> None:
        server = _make_server()
        result = await server.verify(_verify_request(_make_response("<answer>Paris</answer>"), "Paris"))
        assert result.reward == approx(1.0)

    async def test_correct_after_normalization(self) -> None:
        server = _make_server()
        result = await server.verify(_verify_request(_make_response("<answer> the Paris. </answer>"), "paris"))
        assert result.reward == approx(1.0)

    async def test_list_gt(self) -> None:
        server = _make_server()
        result = await server.verify(_verify_request(_make_response("<answer>Paris</answer>"), ["London", "Paris"]))
        assert result.reward == approx(1.0)

    async def test_wrong_answer(self) -> None:
        server = _make_server()
        result = await server.verify(_verify_request(_make_response("<answer>London</answer>"), "Paris"))
        assert result.reward == approx(0.0)

    async def test_missing_answer_tag(self) -> None:
        server = _make_server()
        result = await server.verify(_verify_request(_make_response("I think it is Paris."), "Paris"))
        assert result.reward == approx(0.0)

    async def test_empty_output(self) -> None:
        server = _make_server()
        result = await server.verify(_verify_request(_make_response(), "Paris"))
        assert result.reward == approx(0.0)

    async def test_answer_inside_think_does_not_count(self) -> None:
        server = _make_server()
        text = "<think>the answer is <answer>Paris</answer></think>"
        result = await server.verify(_verify_request(_make_response(text), "Paris"))
        assert result.reward == approx(0.0)

    async def test_last_answer_wins(self) -> None:
        server = _make_server()
        text = "<answer>London</answer> wait, let me reconsider. <answer>Paris</answer>"
        result = await server.verify(_verify_request(_make_response(text), "Paris"))
        assert result.reward == approx(1.0)

    async def test_multi_message_output_joined(self) -> None:
        # Multi-turn rollout: reasoning message first, final answer in the last message.
        server = _make_server()
        result = await server.verify(
            _verify_request(_make_response("Let me search for that.", "<answer>Paris</answer>"), "Paris")
        )
        assert result.reward == approx(1.0)

    async def test_response_passthrough(self) -> None:
        # BaseVerifyResponse keeps the base fields (response, create params);
        # extra request fields like expected_answer are dropped by design.
        server = _make_server()
        result = await server.verify(_verify_request(_make_response("<answer>Paris</answer>"), "Paris"))
        assert result.response.id == "test"
        assert result.reward == approx(1.0)
