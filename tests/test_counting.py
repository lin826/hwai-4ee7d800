import os

from health_context.counting import load_dotenv


def test_load_dotenv_sets_missing_keys_only(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# comment\nNEW_KEY="abc"\nexport OTHER=1\nKEEP=from-file\n')
    monkeypatch.delenv("NEW_KEY", raising=False)
    monkeypatch.delenv("OTHER", raising=False)
    monkeypatch.setenv("KEEP", "from-env")
    load_dotenv(env)
    assert os.environ["NEW_KEY"] == "abc"
    assert os.environ["OTHER"] == "1"
    assert os.environ["KEEP"] == "from-env"
