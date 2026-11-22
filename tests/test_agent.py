"""エージェントの配線テスト。

TestModel は実際のLLMを呼ばない偽モデルで、登録されたツールを全部呼んでから
ダミーの回答を返す。APIキーも利用枠も使わずに、
「ツールが登録され、呼ばれ、結果がモデルに戻るか」を確かめられる。
"""

import json

import pytest
from pydantic_ai import DeferredToolRequests
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai.models.test import TestModel

from app import tools
from app.agent import agent, ask_tool_approvals

BASE_TOOLS = {"search_materials"}


def _parts(result, kind):
    return [p for m in result.all_messages() for p in m.parts if isinstance(p, kind)]


def test_base_tools_are_registered_and_called(monkeypatch):
    monkeypatch.setattr("app.retrieval.search", lambda *args, **kwargs: [])
    model = TestModel()
    with agent.override(model=model):
        result = agent.run_sync("LoRAとは？")

    offered = {t.name for t in model.last_model_request_parameters.function_tools}
    assert BASE_TOOLS <= offered  # 演習でツールを足しても壊れないよう「含む」で判定

    called = {p.tool_name for p in _parts(result, ToolCallPart)}
    assert BASE_TOOLS <= called
    # 承認が必要なツール（requires_approval=True）は、実行されずに承認待ちになるので戻り値が無い
    waiting = len(result.output.approvals) if isinstance(result.output, DeferredToolRequests) else 0
    assert len(_parts(result, ToolReturnPart)) + waiting == len(_parts(result, ToolCallPart))
    assert result.output


def test_material_result_reaches_the_model(monkeypatch):
    hit = {"page_id": "lec02-main-p052", "document_id": "lec02-main", "lecture": 2, "page": 52, "content": "元の重みは固定"}
    neighbor = {"page_id": "lec02-main-p051", "document_id": "lec02-main", "lecture": 2, "page": 51, "content": "小さな追加部分だけを更新"}
    monkeypatch.setattr("app.retrieval.search", lambda *args, **kwargs: [hit])
    monkeypatch.setattr("app.retrieval.load_pages", lambda *_: [hit, neighbor])
    model = TestModel(call_tools=["search_materials"])
    with agent.override(model=model):
        result = agent.run_sync("RAGとは？")
    returns = _parts(result, ToolReturnPart)
    assert returns and returns[0].tool_name == "search_materials"
    assert json.loads(returns[0].content)["results"] == [hit, {**neighbor, "neighbor_of": hit["page_id"]}]


@pytest.mark.parametrize("answer,approved", [("y", True), ("n", False), ("", False), (None, False)])
def test_learning_log_is_saved_only_after_approval(tmp_path, monkeypatch, capsys, answer, approved):
    monkeypatch.setattr(tools, "DATA_DIR", tmp_path / "data")
    model = TestModel(call_tools=["save_learning_log"])
    with agent.override(model=model):
        pending = agent.run_sync("学習内容を記録してください")
        assert isinstance(pending.output, DeferredToolRequests)
        assert len(pending.output.approvals) == 1
        assert not (tmp_path / "learning_log").exists()
        call = pending.output.approvals[0]

        def respond(prompt):
            # 確認画面の入力時点でも、保存はまだ行われていない。
            assert not (tmp_path / "learning_log").exists()
            if answer is None:
                raise EOFError
            return answer

        monkeypatch.setattr("builtins.input", respond)
        decisions = ask_tool_approvals(pending.output)
        assert decisions.approvals[call.tool_call_id] is approved
        assert "learning_log/" in capsys.readouterr().out
        assert not (tmp_path / "learning_log").exists()

        finished = agent.run_sync(
            message_history=pending.all_messages(),
            deferred_tool_results=decisions,
        )
        assert isinstance(finished.output, str)

    files = list((tmp_path / "learning_log").glob("*.json"))
    assert len(files) == int(approved)
    if approved:
        record = json.loads(files[0].read_text(encoding="utf-8"))
        args = call.args_as_dict()
        assert record["topic"] == args["topic"]
        assert record["note"] == args["note"]
        assert record["saved_at"]


@pytest.mark.parametrize("answer,expected_files", [("y", 1), ("n", 0)])
def test_cli_preserves_approval_with_retrieval_modes(tmp_path, monkeypatch, capsys, answer, expected_files):
    from app import agent as agent_module

    monkeypatch.setattr(tools, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(agent_module, "build_model", lambda: TestModel(call_tools=["save_learning_log"]))
    monkeypatch.setattr("sys.argv", ["app.agent", "--trace", "学習内容を保存してください"])
    monkeypatch.setattr("builtins.input", lambda _: answer)

    agent_module.main()

    assert len(list((tmp_path / "learning_log").glob("*.json"))) == expected_files
    output = capsys.readouterr().out
    assert "承認待ちのツール：save_learning_log" in output
    assert "[time]" in output
    assert "[tool call]" in output
