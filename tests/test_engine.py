import time
from unittest.mock import MagicMock, patch

from app.inference.engine import QueryEngine, QueryResult


def test_query_result_fields():
    r = QueryResult(
        text="Hello world",
        ttft_ms=50.0,
        decode_tokens_per_sec=35.0,
        total_tokens=10,
    )
    assert r.text == "Hello world"
    assert r.ttft_ms == 50.0
    assert r.decode_tokens_per_sec == 35.0


def test_format_query_prompt():
    text = QueryEngine.format_query_prompt("What does this code do?")
    assert "What does this code do?" in text
    assert "</query>" in text
    assert "<query>" not in text  # opening tag is in the cached prefix, not here
