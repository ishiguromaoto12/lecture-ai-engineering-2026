"""会話の継続・自動保存・再開を、実際のAPIを呼ばずに確認する。"""

from unittest.mock import Mock

import pytest
from google.genai import errors, models, types

from app import ask, chat_history


def response(text="テスト回答", signature=None):
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(
            role="model", parts=[types.Part(text=text, thought_signature=signature)],
        ))],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=10, candidates_token_count=5,
        ),
    )


@pytest.fixture
def fake_api(monkeypatch):
    # チャットの履歴管理は本物のSDKを使い、通信する箇所だけ差し替える。
    generate = Mock(return_value=response())
    monkeypatch.setattr(models.Models, "generate_content", generate)
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")
    return generate


def test_autosave_and_resume_keep_context_and_model(tmp_path, monkeypatch, fake_api, capsys):
    path = tmp_path / "session.json"
    monkeypatch.setattr(ask, "new_history_path", lambda: path)
    fake_api.side_effect = [response("はるさんですね。", b"\xff\x00\x80"), response("はるさんです。")]
    inputs = iter(["私の名前は、はるです。", "私の名前は？", "/exit"])

    def next_input(prompt):
        question = next(inputs)
        if question != "私の名前は、はるです。":
            # 終了時だけでなく、次の入力を待つ前に保存済み。
            assert path.exists()
        return question

    monkeypatch.setattr("builtins.input", next_input)
    ask.run_chat()
    assert len(fake_api.call_args_list[0].kwargs["contents"]) == 1
    assert len(fake_api.call_args_list[1].kwargs["contents"]) == 3
    model, history = chat_history.load_history(path)
    assert len(history) == 4
    assert history[1].parts[0].thought_signature == b"\xff\x00\x80"

    # 環境変数側のモデルが変わっても、保存した会話のモデルで再開する。
    monkeypatch.setattr(ask, "MODEL", "another-model")
    fake_api.side_effect = None
    inputs = iter(["前の会話の名前を教えて。", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
    ask.run_chat(path)
    sent = fake_api.call_args.kwargs
    assert sent["model"] == model
    assert len(sent["contents"]) == 5
    assert sent["contents"][0].parts[0].text == "私の名前は、はるです。"
    assert sent["contents"][1].parts[0].thought_signature == b"\xff\x00\x80"
    assert len(chat_history.load_history(path)[1]) == 6
    assert "[再開] 2往復" in capsys.readouterr().out


@pytest.mark.parametrize("ending", ["/exit", EOFError(), KeyboardInterrupt()])
def test_empty_input_and_exit_do_not_call_api_or_save(tmp_path, monkeypatch, fake_api, ending):
    path = tmp_path / "unused.json"
    monkeypatch.setattr(ask, "new_history_path", lambda: path)
    inputs = iter(["   ", ending])

    def next_input(prompt):
        value = next(inputs)
        if isinstance(value, BaseException):
            raise value
        return value

    monkeypatch.setattr("builtins.input", next_input)
    ask.run_chat()
    fake_api.assert_not_called()
    assert not path.exists()


def test_api_failure_can_retry_without_saving_failed_question(tmp_path, monkeypatch, fake_api):
    path = tmp_path / "session.json"
    monkeypatch.setattr(ask, "new_history_path", lambda: path)
    fake_api.side_effect = [response(), errors.ClientError(429, {"message": "test"}), response()]
    inputs = iter(["最初の質問", "失敗する質問", "やり直し", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
    ask.run_chat()
    history = chat_history.load_history(path)[1]
    assert [c.parts[0].text for c in history[::2]] == ["最初の質問", "やり直し"]
    assert len(fake_api.call_args.kwargs["contents"]) == 3


def test_empty_response_is_not_added_to_next_turn(tmp_path, monkeypatch, fake_api):
    path = tmp_path / "session.json"
    monkeypatch.setattr(ask, "new_history_path", lambda: path)
    fake_api.side_effect = [response(), types.GenerateContentResponse(), response()]
    inputs = iter(["最初の質問", "回答なし", "やり直し", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
    ask.run_chat()
    assert len(chat_history.load_history(path)[1]) == 4
    assert len(fake_api.call_args.kwargs["contents"]) == 3


@pytest.mark.parametrize("data", [
    "not JSON", "[]", '{"version": 2}',
    '{"version": 1, "model": "m", "history": [{}]}',
    '{"version": 1, "model": "m", "history": [{"role":"model"}, {"role":"user"}]}',
])
def test_invalid_history_is_rejected_before_api(tmp_path, monkeypatch, fake_api, data):
    path = tmp_path / "broken.json"
    path.write_text(data, encoding="utf-8")
    with pytest.raises(SystemExit, match="会話履歴を読み込めません"):
        ask.run_chat(path)
    fake_api.assert_not_called()


def test_failed_replace_preserves_previous_file(tmp_path, monkeypatch):
    path = tmp_path / "session.json"
    chat_history.save_history(path, "m", [])
    original = path.read_bytes()

    def fail_replace(source, destination):
        raise OSError("disk error")

    monkeypatch.setattr(chat_history.os, "replace", fail_replace)
    with pytest.raises(OSError):
        chat_history.save_history(path, "changed", [])
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_save_failure_stops_chat_with_clear_message(tmp_path, monkeypatch, fake_api):
    monkeypatch.setattr(ask, "new_history_path", lambda: tmp_path / "session.json")
    monkeypatch.setattr("builtins.input", lambda prompt: "質問")
    monkeypatch.setattr(ask, "save_history", Mock(side_effect=OSError("disk error")))
    with pytest.raises(SystemExit, match="今回の回答は保存できていません"):
        ask.run_chat()
    assert fake_api.call_count == 1


def test_one_shot_command_still_works(monkeypatch, fake_api, capsys):
    clock = iter([100.0, 101.25])
    monkeypatch.setattr(ask, "perf_counter", lambda: next(clock))
    monkeypatch.setattr("sys.argv", ["app.ask", "LoRAとは？"])
    ask.main()
    assert fake_api.call_args.kwargs["contents"] == "LoRAとは？"
    output = capsys.readouterr().out
    assert "[応答時間（秒）] 1.250" in output
    assert "[入力トークン数] 10" in output
    assert "[出力トークン数] 5" in output
    assert "テスト回答" in output


@pytest.mark.parametrize("args", [["--resume", "history.json"], ["--chat", "質問"]])
def test_invalid_command_options_do_not_call_api(monkeypatch, fake_api, args):
    monkeypatch.setattr("sys.argv", ["app.ask", *args])
    with pytest.raises(SystemExit) as e:
        ask.main()
    assert e.value.code == 2
    fake_api.assert_not_called()
