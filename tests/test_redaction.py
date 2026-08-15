from wordlehands.guardrails.redaction import REDACTED, redact, redact_dict


def test_redacts_api_key_like_tokens():
    assert redact("api_key sk-abcdef0123456789ABCDEF") == "api_key " + REDACTED


def test_redacts_bearer_token():
    text = "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload"
    assert REDACTED in redact(text)
    assert "eyJhbGciOiJIUzI1NiJ9" not in redact(text)


def test_redacts_email():
    assert redact("contact me at pruthvi@example.com") == f"contact me at {REDACTED}"


def test_leaves_ordinary_text_untouched():
    text = "guess=CRANE result=elsewhere,absent,absent,correct,absent"
    assert redact(text) == text


def test_redact_dict_walks_nested_structures():
    data = {"user": {"email": "a@b.com"}, "notes": ["fine", "token sk-1234567890abcdef1234"]}
    out = redact_dict(data)
    assert out["user"]["email"] == REDACTED
    assert out["notes"][0] == "fine"
    assert REDACTED in out["notes"][1]
