#!/usr/bin/env python3
"""Fail closed when a candidate public repository contains unsafe material."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


ALLOWED_SUFFIXES = {
    ".md",
    ".py",
    ".r",
    ".toml",
    ".txt",
}

ALLOWED_SUFFIXLESS_NAMES = {".gitignore"}

FORBIDDEN_SUFFIXES = {
    ".csv",
    ".gz",
    ".parquet",
    ".feather",
    ".h5",
    ".hdf5",
    ".joblib",
    ".pkl",
    ".pickle",
    ".zip",
    ".tar",
    ".docx",
    ".pdf",
    ".png",
    ".tif",
    ".tiff",
}

FORBIDDEN_NAMES = {
    ".DS_Store",
    ".env",
    "config.local.toml",
    "id_rsa",
    "id_ed25519",
}

TEXT_PATTERNS = {
    "personal absolute path": re.compile(r"/(?:Users|home)/[A-Za-z0-9._-]+/"),
    "mounted-volume absolute path": re.compile(r"/Volumes/[A-Za-z0-9._ -]+/"),
    "Windows user path": re.compile(r"[A-Za-z]:\\\\Users\\\\[^\\\\\s]+"),
    "credential assignment": re.compile(
        r"(?i)(?:api[_-]?key|access[_-]?token|secret|password|"
        r"aws_access_key_id|aws_secret_access_key)\s*[:=]\s*['\"]?[^'\"\s]+"
    ),
    "AWS access key": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "Bearer token": re.compile(
        r"(?i)\b(?:authorization\s*:\s*)?bearer\s+[A-Za-z0-9._~+/-]{16,}={0,2}"
    ),
    "JWT token": re.compile(
        r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"
    ),
    "signed URL": re.compile(
        r"(?i)https?://[^\s'\"<>]+[?&](?:x-amz-signature|x-goog-signature|"
        r"signature|access_token|token|sig)=[^&\s'\"<>]+"
    ),
    "private key": re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    "email address": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
}

IGNORED_PARTS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


def iter_entries(root: Path):
    for path in sorted(root.rglob("*")):
        if any(part in IGNORED_PARTS for part in path.parts):
            continue
        yield path


def iter_files(root: Path):
    for path in iter_entries(root):
        if path.is_symlink() or not path.is_file():
            continue
        yield path


def check(root: Path) -> list[str]:
    failures: list[str] = []
    for path in iter_entries(root):
        if path.is_symlink():
            failures.append(f"symbolic link is not allowed: {path.relative_to(root)}")
    for path in iter_files(root):
        relative = path.relative_to(root)
        if path.name in FORBIDDEN_NAMES:
            failures.append(f"forbidden file name: {relative}")
        suffix = path.suffix.lower()
        if suffix not in ALLOWED_SUFFIXES and path.name not in ALLOWED_SUFFIXLESS_NAMES:
            failures.append(f"unapproved release file type: {relative}")
            continue
        if suffix in FORBIDDEN_SUFFIXES:
            failures.append(f"forbidden release file type: {relative}")
            continue
        if path.stat().st_size > 2_000_000:
            failures.append(f"unexpected file larger than 2 MB: {relative}")
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            failures.append(f"non-UTF-8 or binary file: {relative}")
            continue
        for label, pattern in TEXT_PATTERNS.items():
            if pattern.search(text):
                failures.append(f"{label}: {relative}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    failures = check(root)
    if failures:
        print("RELEASE_CHECK_FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    count = sum(1 for _ in iter_files(root))
    print(f"RELEASE_CHECK_PASS files={count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
