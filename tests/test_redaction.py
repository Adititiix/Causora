import json

import pytest

from incident_copilot.db import mask_dsn
from incident_copilot.redaction import MASK, redact


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("dsn=postgresql://etl_user:FAKE-PASSWORD-123@pg.internal:5432/orders",
         f"dsn=postgresql://etl_user:{MASK}@pg.internal:5432/orders"),
        ("https://bob:s3cr3t@example.com/x", f"https://bob:{MASK}@example.com/x"),
        ("Calling upstream API with token=FAKE-TOKEN-9f3a1c and user=svc-etl",
         f"Calling upstream API with token={MASK} and user=svc-etl"),
        ("password=hunter2;", f"password={MASK};"),
        ("PASSWORD = 'two words'", f"PASSWORD = {MASK}"),
        ("api_key=abc123&x=1", f"api_key={MASK}&x=1"),
        ("apikey=abc123", f"apikey={MASK}"),
        ("access_token=eyJhbGciOi.payload.sig", f"access_token={MASK}"),
        ('{"password": "hunter2", "user": "bob"}', f'{{"password": "{MASK}", "user": "bob"}}'),
        ("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.e30.sig", f"Authorization: Bearer {MASK}"),
        ("key AKIAABCDEFGHIJKLMNOP in config", f"key {MASK} in config"),
        ("aws_secret_access_key=wJalrXUtnFEMI/K7MDENG", f"aws_secret_access_key={MASK}"),
    ],
)
def test_redacts_credentials(text, expected):
    assert redact(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Rotate the password in the secrets manager, then clear the task.",
        "The token bucket refills every second; tokens=5 per call.",
        "Connecting to postgresql://pg.internal:5432/orders",
        "http://host:8080/path",
        "KeyError: 'customer_tier'",
    ],
)
def test_leaves_ordinary_text_alone(text):
    assert redact(text) == text


def test_redaction_is_idempotent():
    once = redact("token=abc password=def postgresql://u:p@h/db")
    assert redact(once) == once


def test_mask_dsn_hides_password():
    masked = mask_dsn("postgresql://copilot:change-me-locally@db:5432/incident_copilot")
    assert masked == f"postgresql://copilot:{MASK}@db:5432/incident_copilot"
    assert "change-me-locally" not in masked


def test_fake_secret_logs_are_fully_redacted_and_nothing_else_changes(dataset):
    """Only the 8 flagged files change, and none of them keeps a FAKE- string."""
    root = dataset["root"]
    manifest = json.loads((root / "manifest.json").read_text())
    flagged = {e["path"] for e in manifest["files"] if e.get("has_fake_secret")}
    assert len(flagged) == 8
    changed = set()
    for entry in manifest["files"]:
        text = (root / entry["path"]).read_text(encoding="utf-8")
        cleaned = redact(text)
        assert "FAKE-" not in cleaned, entry["path"]
        if cleaned != text:
            changed.add(entry["path"])
    assert changed == flagged
