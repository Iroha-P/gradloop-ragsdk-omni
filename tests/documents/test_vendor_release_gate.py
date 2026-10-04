from collections import Counter
from pathlib import Path

from scripts.public_release_scan import _scan_text


def test_pinned_public_parsers_pass_with_licenses_preserved():
    root = Path(__file__).resolve().parents[2]
    for path in (root / "app/ui/documents/vendor").iterdir():
        findings = Counter()
        _scan_text(path.read_text(encoding="utf-8"), path.relative_to(root).as_posix(), findings, denylist_terms=())
        assert not findings, (path.name, dict(findings))


def test_one_byte_vendor_change_does_not_inherit_public_literal_exemptions():
    root = Path(__file__).resolve().parents[2]
    relative = "app/ui/documents/vendor/pdf.worker.mjs"
    original = (root / relative).read_text(encoding="utf-8")
    findings = Counter()
    _scan_text(original + "\n" + "sk-" + "x" * 24, relative, findings, denylist_terms=())
    assert findings["secret"]
    assert findings["private_absolute_path"]
    assert findings["pii"]


def test_reusing_vendor_bytes_under_an_unapproved_path_is_not_exempt():
    root = Path(__file__).resolve().parents[2]
    findings = Counter()
    _scan_text((root / "app/ui/documents/vendor/pdf.worker.mjs").read_text(encoding="utf-8"), "app/ui/documents/vendor/unapproved.mjs", findings, denylist_terms=())
    assert findings["secret"]
