import itertools

import pytest

from cpurag.chunking import chunk_text

DOC = (
    "Preface text that sits before any heading and is long enough to keep.\n\n"
    "# Guide\nIntro paragraph for the guide section, describing what follows in detail.\n\n"
    "## Install\n" + "Install the package with pip and verify the version afterwards. " * 30 + "\n\n"
    "## Configure\nSet the environment variables and restart the service to apply them.\n"
)


def test_offsets_point_at_exact_source_text() -> None:
    for c in chunk_text("d", DOC, max_chars=300, overlap=60):
        assert DOC[c.start : c.end] == c.text


def test_sections_carry_heading_path() -> None:
    sections = {c.section for c in chunk_text("d", DOC, max_chars=300, overlap=60)}
    assert {"", "Guide", "Guide > Install", "Guide > Configure"} <= sections


def test_chunks_respect_size_and_never_cross_sections() -> None:
    chunks = chunk_text("d", DOC, max_chars=300, overlap=60)
    assert all(len(c.text) <= 300 for c in chunks)
    install = [c for c in chunks if c.section == "Guide > Install"]
    assert len(install) > 3
    assert not any("Set the environment" in c.text for c in install)


def test_consecutive_chunks_overlap_on_word_boundaries() -> None:
    install = [c for c in chunk_text("d", DOC, max_chars=300, overlap=60) if c.section == "Guide > Install"]
    for a, b in itertools.pairwise(install):
        assert b.start < a.end, "expected overlap between neighbours"
        assert DOC[b.start - 1] in " \n", "overlap must start on a word boundary"


def test_prefers_sentence_boundaries() -> None:
    install = [c for c in chunk_text("d", DOC, max_chars=300, overlap=0) if c.section == "Guide > Install"]
    assert all(c.text.endswith(".") for c in install[:-1])


def test_ids_metadata_and_min_chars() -> None:
    chunks = chunk_text("doc", "# A\ntiny\n# B\n" + "word " * 30, metadata={"lang": "en"})
    assert [c.chunk_id for c in chunks] == ["doc#0"]
    assert chunks[0].metadata == {"lang": "en"} and chunks[0].section == "B"


def test_plain_text_and_giant_tokens() -> None:
    chunks = chunk_text("p", "x" * 1000, max_chars=200, overlap=0)
    assert len(chunks) == 5 and all(len(c.text) == 200 for c in chunks)


@pytest.mark.parametrize(("max_chars", "overlap"), [(0, 0), (100, 100), (100, -1)])
def test_rejects_bad_parameters(max_chars: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_text("d", DOC, max_chars=max_chars, overlap=overlap)
