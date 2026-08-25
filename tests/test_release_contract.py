#!/usr/bin/env python3
"""Static release-contract checks that require no protected data."""

from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"
sys.path.insert(0, str(ANALYSIS))
sys.path.insert(0, str(ROOT / "tools"))


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="c02-release-contract-") as private, tempfile.TemporaryDirectory(
        prefix="c02-approved-offload-"
    ) as approved_offload:
        os.environ["C02_PRIVATE_WORKSPACE"] = private
        from c02_runtime import configure_private_workspace, require_private_path
        from export_authoritative_result_map import EXPECTED_IDS
        from check_release import check

        configure_private_workspace(private)
        assert len(EXPECTED_IDS) == 22
        try:
            require_private_path(ROOT / "outputs", must_exist=False)
        except RuntimeError:
            pass
        else:
            raise AssertionError("protected output inside public code was accepted")

        escape = Path(private) / "escape_to_code"
        escape.symlink_to(ROOT, target_is_directory=True)
        try:
            require_private_path(escape / "would_be_restricted.csv.gz", must_exist=False)
        except RuntimeError:
            pass
        else:
            raise AssertionError("symlink escape into public code was accepted")

        try:
            require_private_path(
                Path(private) / "inside" / ".." / ".." / "escape.csv.gz",
                must_exist=False,
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("lexical parent-directory escape was accepted")

        configure_private_workspace(private, [approved_offload])
        approved_link = Path(private) / "approved_offload"
        approved_link.symlink_to(approved_offload, target_is_directory=True)
        accepted = require_private_path(
            approved_link / "protected.csv.gz", must_exist=False
        )
        assert accepted == approved_link / "protected.csv.gz"

        methods = (ROOT / "docs/METHODS_MAP.md").read_text(encoding="utf-8")
        references = set(re.findall(r"`((?:analysis|validation)/[^`]+\.(?:py|R))`", methods))
        missing = sorted(reference for reference in references if not (ROOT / reference).is_file())
        assert not missing, f"documentation references missing code: {missing}"

        for path in ANALYSIS.glob("*.py"):
            source = path.read_text(encoding="utf-8")
            if path.name != "c02_runtime.py":
                assert "ROOT = Path(__file__).resolve().parents[1]" not in source

        workflow_source = (
            ANALYSIS / "analyze_c02_clinical_workflow_deepening.py"
        ).read_text(encoding="utf-8")
        assert 'filterwarnings("ignore", category=RuntimeWarning)' not in workflow_source

        unknown = Path(private) / "unknown.release-extension"
        unknown.write_text("synthetic", encoding="utf-8")
        failures = check(Path(private))
        assert any("unapproved release file type" in failure for failure in failures)
        assert any("symbolic link is not allowed" in failure for failure in failures)

        sensitive_fixtures = {
            "aws.txt": "AWS_" + "ACCESS_KEY_ID='" + "AKIA" + "A" * 16 + "'",
            "bearer.md": "Authorization: " + "Bearer " + "a" * 32,
            "jwt.txt": "eyJ" + "a" * 16 + "." + "b" * 16 + "." + "c" * 16,
            "signed-url.txt": (
                "https://example.invalid/object?" + "X-Amz-" + "Signature=" + "d" * 32
            ),
            "mounted-path.toml": "path = '" + "/" + "Volumes/private/data.csv'",
        }
        expected_labels = {
            "AWS access key",
            "Bearer token",
            "JWT token",
            "signed URL",
            "mounted-volume absolute path",
        }
        for name, content in sensitive_fixtures.items():
            (Path(private) / name).write_text(content, encoding="utf-8")
        failures = check(Path(private))
        observed_labels = {failure.split(":", 1)[0] for failure in failures}
        assert expected_labels.issubset(observed_labels), (
            expected_labels - observed_labels
        )

    print("RELEASE_CONTRACT_TEST_PASS")


if __name__ == "__main__":
    main()
