import numpy as np
import pytest
from conftest import FakeOpenAI

from cpurag.embeddings import Embedder, HashingEmbedder, OpenAICompatibleEmbedder


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b)


def test_deterministic_across_instances_and_normalised() -> None:
    a = HashingEmbedder(512).embed(["restart the nginx service"])
    b = HashingEmbedder(512).embed(["restart the nginx service"])
    assert np.array_equal(a, b)
    assert a.dtype == np.float32 and a.shape == (1, 512)
    assert np.isclose(np.linalg.norm(a[0]), 1.0, atol=1e-5)


def test_related_texts_are_closer_than_unrelated() -> None:
    e = HashingEmbedder(1024)
    q, rel, unrel = e.embed(
        ["rotate tls certificates", "renewing TLS certificate before expiry", "bake bread"]
    )
    assert cos(q, rel) > cos(q, unrel) + 0.1


def test_character_ngrams_survive_typos() -> None:
    with_chars, without = HashingEmbedder(1024), HashingEmbedder(1024, char_ngrams=False)
    a, b = with_chars.embed(["kubernetes readiness probe", "kubernets readines probe"])
    c, d = without.embed(["kubernetes readiness probe", "kubernets readines probe"])
    assert cos(a, b) > cos(c, d)


def test_idf_downweights_ubiquitous_terms() -> None:
    corpus = [f"server {w}" for w in ("alpha", "beta", "gamma", "delta", "epsilon")]
    e = HashingEmbedder(1024).fit(corpus)
    assert e.idf["w:server"] < e.idf["w:alpha"]


def test_empty_text_gives_zero_vector_not_nan() -> None:
    v = HashingEmbedder(64).embed([""])
    assert not np.isnan(v).any() and float(np.abs(v).sum()) == 0.0


@pytest.mark.parametrize("dim", [32, 1000])
def test_dim_must_be_power_of_two(dim: int) -> None:
    with pytest.raises(ValueError):
        HashingEmbedder(dim)


def test_protocol_conformance() -> None:
    assert isinstance(HashingEmbedder(64), Embedder)
    assert isinstance(OpenAICompatibleEmbedder("http://x", "m", 4), Embedder)


def test_openai_compatible_client(fake_openai: str) -> None:
    e = OpenAICompatibleEmbedder(fake_openai, "local-embed", 4, api_key="k")
    m = e.embed(["ab", "abcd"])
    assert m.shape == (2, 4)
    assert np.allclose(np.linalg.norm(m, axis=1), 1.0)
    assert m[1, 0] > m[0, 0], "rows must be re-ordered by index"
    assert FakeOpenAI.requests[0]["auth"] == "Bearer k"
    assert FakeOpenAI.requests[0]["model"] == "local-embed"


def test_openai_compatible_dimension_mismatch(fake_openai: str) -> None:
    with pytest.raises(ValueError, match="shape"):
        OpenAICompatibleEmbedder(fake_openai, "m", 8).embed(["x"])
