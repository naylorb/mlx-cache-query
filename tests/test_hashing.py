from mcq.core.hashing import sha256_hex, file_content_hash


def test_sha256_hex_bytes():
    result = sha256_hex(b"hello")
    assert len(result) == 64
    assert result == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


def test_sha256_hex_deterministic():
    assert sha256_hex(b"test") == sha256_hex(b"test")


def test_sha256_hex_different_inputs():
    assert sha256_hex(b"a") != sha256_hex(b"b")


def test_file_content_hash():
    h = file_content_hash("readme.md", "hello world")
    assert len(h) == 64


def test_file_content_hash_deterministic():
    a = file_content_hash("a.py", "content")
    b = file_content_hash("a.py", "content")
    assert a == b


def test_file_content_hash_changes_with_path():
    a = file_content_hash("a.py", "content")
    b = file_content_hash("b.py", "content")
    assert a != b


def test_file_content_hash_changes_with_content():
    a = file_content_hash("a.py", "content1")
    b = file_content_hash("a.py", "content2")
    assert a != b
