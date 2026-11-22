"""第4回: 採点（app/judge.py）の配線テスト。API キーは使わない。

--model test と同じ偽モデル（規則で判定する FunctionModel）で、評価セット 10 件の
「採点 → 人のラベルとの比較 → 要確認・不一致の集計」が通ることを確かめる。
"""

from types import SimpleNamespace

import pytest
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior

from app import agent as agent_module
from app import judge
from app.judge import (
    ALL_CRITERIA,
    DEFAULT_CASES,
    DEFAULT_RUBRIC,
    build_judge,
    build_test_model,
    check_lecture_mark,
    judge_cases,
    load_labels,
    overall,
    read_jsonl,
    summarize,
)

INSTRUCTOR_LABELS = judge.ROOT / "lectures" / "04" / "reference" / "labels_instructor.jsonl"


def test_lecture_mark_is_checked_by_code():
    assert check_lecture_mark("LoRA は…\n\n関係する回：第2回") == "pass"
    assert check_lecture_mark("第2回で扱う LoRA は…\n\n以上です。") == "fail"  # 末尾に無ければ fail
    assert check_lecture_mark("") == "fail"


def test_overall_requires_all_required_criteria():
    base = {"grounding": "pass", "scope": "pass", "hint": "pass", "lecture_mark": "pass", "clarity": "fail"}
    assert overall(base) == "pass"  # 分かりやすさは総合に入れない
    assert overall({**base, "hint": "fail"}) == "fail"
    assert overall({**base, "grounding": "unsure"}) == "unsure"


def test_cases_and_rubric_are_consistent():
    cases = read_jsonl(DEFAULT_CASES)
    assert len(cases) == 10
    assert len({c["id"] for c in cases}) == 10
    for c in cases:
        assert c["question"] and c["answer"] and c["reference"]["expected"]
    rubric = DEFAULT_RUBRIC.read_text(encoding="utf-8")
    for key, name in ALL_CRITERIA[:-1]:  # 回の表示はコードで判定するので rubric には書かない
        assert f"{name}（{key}）" in rubric
    labels = load_labels(INSTRUCTOR_LABELS)
    assert set(labels) == {c["id"] for c in cases}


def test_judge_pipeline_with_fake_model():
    cases = read_jsonl(DEFAULT_CASES)
    labels = load_labels(INSTRUCTOR_LABELS)
    agent = build_judge(build_test_model(), DEFAULT_RUBRIC.read_text(encoding="utf-8"))

    verdicts = judge_cases(agent, cases)
    assert set(verdicts) == {c["id"] for c in cases}
    for v in verdicts.values():
        for key, _ in ALL_CRITERIA:
            assert v[key]["verdict"] in ("pass", "fail", "unsure")
            assert v[key]["reason"]

    summary = summarize(cases, verdicts, labels)
    assert summary["n"] == 10
    # unsure にした項目は一致率の分母から外れ、要確認に入る
    excluded = sum(p["total"] - p["compared"] for p in summary["per_criterion"].values())
    assert excluded == len(summary["review"])
    assert summary["coverage"]["covered"] + len({r["id"] for r in summary["review"]}) == 10
    for m in summary["mismatches"]:
        assert m["kind"] in ("見逃し", "過検出")


def test_generation_keeps_progress_and_resumes_after_rate_limit(tmp_path, monkeypatch):
    cases = [{"id": str(i), "question": f"question{i}", "answer": ""} for i in range(3)]
    path = tmp_path / "cases.jsonl"
    judge.write_jsonl(path, cases)
    calls, waits = [], []

    def generate(question, *, model):
        calls.append(question)
        if len(calls) == 2:
            raise ModelHTTPError(429, "test", body={"message": "rate limited"})
        return SimpleNamespace(output=f"answer to {question}")

    monkeypatch.setattr(agent_module, "agent", SimpleNamespace(run_sync=generate))
    monkeypatch.setattr(agent_module, "build_model", lambda: object())
    monkeypatch.setattr(judge.time, "sleep", waits.append)

    with pytest.raises(ModelHTTPError):
        judge.generate_answers(cases, wait=15, checkpoint=path)
    saved = judge.read_jsonl(path)
    assert [bool(c["answer"]) for c in saved] == [True, False, False]

    judge.generate_answers(saved, wait=15, checkpoint=path)
    assert calls == ["question0", "question1", "question1", "question2"]
    assert waits == [15, 15]
    assert all(c["answer"] for c in judge.read_jsonl(path))

    # 完了後の再実行はAPIを呼ばない。明示的に作り直す場合だけ再生成する。
    completed = judge.read_jsonl(path)
    judge.generate_answers(completed, checkpoint=path)
    assert len(calls) == 4
    judge.generate_answers(completed, wait=0, checkpoint=path, regenerate=True)
    assert calls[4:] == ["question0", "question1", "question2"]


@pytest.mark.parametrize("wait_option,expected_wait", [([], 15.0), (["--wait", "30"], 30.0)])
def test_generate_cli_passes_wait_and_checkpoint(tmp_path, monkeypatch, wait_option, expected_wait):
    path = tmp_path / "cases.jsonl"
    judge.write_jsonl(path, [{"id": "01", "question": "質問", "answer": ""}])
    received = {}

    def generate(cases, **kwargs):
        received.update(kwargs)
        return cases

    monkeypatch.setattr(judge, "generate_answers", generate)
    judge.main(["--generate", "--cases", str(path), *wait_option])
    assert received == {"wait": expected_wait, "checkpoint": path, "regenerate": False}


def test_generation_does_not_save_pending_approval_as_an_answer(tmp_path, monkeypatch):
    from pydantic_ai import DeferredToolRequests

    cases = [{"id": "01", "question": "保存してください", "answer": ""}]
    path = tmp_path / "cases.jsonl"
    judge.write_jsonl(path, cases)
    monkeypatch.setattr(agent_module, "build_model", lambda: object())
    monkeypatch.setattr(
        agent_module, "agent", SimpleNamespace(run_sync=lambda *args, **kwargs: SimpleNamespace(output=DeferredToolRequests()))
    )
    with pytest.raises(UnexpectedModelBehavior, match="承認待ち"):
        judge.generate_answers(cases, checkpoint=path)
    assert judge.read_jsonl(path)[0]["answer"] == ""
