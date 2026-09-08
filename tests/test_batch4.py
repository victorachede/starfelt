from starfelt.core.doctor import collect_doctor_rows
from starfelt.core.hosted import clear_auth, load_auth, save_auth


def test_doctor_rows_include_python():
    rows = collect_doctor_rows()
    names = {r[0] for r in rows}
    assert "python" in names
    assert ".starfelt/" in names


def test_auth_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "starfelt.core.hosted.auth_path",
        lambda: tmp_path / "auth.json",
    )
    clear_auth()
    save_auth({"supabase_url": "https://example.supabase.co", "supabase_key": "test"})
    auth = load_auth()
    assert auth is not None
    assert auth["supabase_url"].startswith("https://")
    clear_auth()
    assert load_auth() is None
