from pathlib import Path

from scripts.competition.build_public_site import build_public_site


def test_document_mode_is_local_first_and_explicitly_confirmed():
    root = Path(__file__).resolve().parents[2]
    page = (root / "app/ui/index.html").read_text(encoding="utf-8")
    assert 'id="documentFiles"' in page
    assert ".pdf,.doc,.docx" in page
    assert 'id="documentConsent"' in page
    assert 'id="documentPreview"' in page
    assert "parseDocuments" in page
    assert "sendDocumentText" in page
    assert "./documents/client.mjs" in page


def test_pages_package_contains_local_document_parsers(tmp_path):
    output = build_public_site(
        tmp_path / "site", proxy_url="wss://worker.example/ws", project_url="https://repo.example"
    )
    for name in ("client.mjs", "policy.mjs", "worker.mjs"):
        assert (output / "documents" / name).is_file()
    for name in ("pdf.mjs", "pdf.worker.mjs", "word.mjs", "THIRD_PARTY_NOTICES.md", "provenance.json"):
        assert (output / "documents/vendor" / name).is_file()
