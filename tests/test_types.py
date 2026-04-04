from mcq.core.types import CorpusChunk, Corpus, TokenizedPrefix, ArtifactRef


def test_corpus_chunk_creation():
    chunk = CorpusChunk(
        source_path="readme.md",
        content="# Hello\n",
        byte_range=(0, 9),
    )
    assert chunk.source_path == "readme.md"
    assert chunk.content == "# Hello\n"
    assert chunk.byte_range == (0, 9)


def test_corpus_creation_and_hash_determinism():
    chunks_a = [
        CorpusChunk("b.txt", "beta", (0, 4)),
        CorpusChunk("a.txt", "alpha", (0, 5)),
    ]
    chunks_b = [
        CorpusChunk("a.txt", "alpha", (0, 5)),
        CorpusChunk("b.txt", "beta", (0, 4)),
    ]
    corpus_a = Corpus(name="test", chunks=chunks_a)
    corpus_b = Corpus(name="test", chunks=chunks_b)
    assert corpus_a.content_hash == corpus_b.content_hash
    assert len(corpus_a.content_hash) == 64  # sha256 hex


def test_corpus_hash_includes_paths():
    """Different paths with same content must produce different hashes."""
    chunk_a = [CorpusChunk("a.txt", "same content", (0, 12))]
    chunk_b = [CorpusChunk("b.txt", "same content", (0, 12))]
    corpus_a = Corpus(name="a", chunks=chunk_a)
    corpus_b = Corpus(name="b", chunks=chunk_b)
    assert corpus_a.content_hash != corpus_b.content_hash


def test_tokenized_prefix_creation():
    prefix = TokenizedPrefix(
        model_id="test-model",
        model_revision="abc123",
        tokens=[1, 2, 3, 4, 5],
        corpus_hash="deadbeef" * 8,
    )
    assert prefix.token_count == 5
    assert len(prefix.prefix_hash) == 64


def test_tokenized_prefix_hash_determinism():
    kwargs = dict(
        model_id="test-model",
        model_revision="abc123",
        tokens=[1, 2, 3, 4, 5],
        corpus_hash="deadbeef" * 8,
    )
    a = TokenizedPrefix(**kwargs)
    b = TokenizedPrefix(**kwargs)
    assert a.prefix_hash == b.prefix_hash


def test_artifact_ref_creation():
    ref = ArtifactRef(
        artifact_hash="abc" * 21 + "a",
        model_id="test-model",
        model_revision="abc123",
        corpus_hash="def" * 21 + "d",
        corpus_name="my-corpus",
        prefix_token_count=1000,
        prompt_template_version="v1",
        normalization_version="v1",
        build_timestamp="2026-04-03T12:00:00Z",
        file_size_bytes=1024,
        file_path="/tmp/test.safetensors",
        mlx_lm_version="0.22.0",
    )
    assert ref.corpus_name == "my-corpus"
    assert ref.prefix_token_count == 1000
