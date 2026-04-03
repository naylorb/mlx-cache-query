from app.core.hashing import sha256_hex, artifact_hash_from_prefix


def test_sha256_hex_bytes():
    result = sha256_hex(b"hello")
    assert len(result) == 64
    assert result == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


def test_sha256_hex_deterministic():
    assert sha256_hex(b"test") == sha256_hex(b"test")


def test_sha256_hex_different_inputs():
    assert sha256_hex(b"a") != sha256_hex(b"b")


def test_artifact_hash_from_prefix():
    h = artifact_hash_from_prefix(model_revision="rev123", prefix_tokens=[1, 2, 3])
    assert len(h) == 64


def test_artifact_hash_determinism():
    a = artifact_hash_from_prefix("rev", [10, 20])
    b = artifact_hash_from_prefix("rev", [10, 20])
    assert a == b


def test_artifact_hash_changes_with_revision():
    a = artifact_hash_from_prefix("rev1", [10, 20])
    b = artifact_hash_from_prefix("rev2", [10, 20])
    assert a != b


def test_artifact_hash_changes_with_tokens():
    a = artifact_hash_from_prefix("rev", [10, 20])
    b = artifact_hash_from_prefix("rev", [10, 21])
    assert a != b
