"""発展編の入力分離、実際の検索、採点の未確定状態、Colab への持ち運びを確認する。"""

import ast
import copy
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

import glossary_experiment as experiment


ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def inputs(tmp_path):
    shutil.copy2(ROOT / "data/glossary.json", tmp_path / "glossary.json")
    for name in ["glossary_sft.jsonl", "evaluation.json"]:
        shutil.copy2(ROOT / "lectures/02/experiments/glossary" / name, tmp_path / name)
    return experiment.load_inputs(tmp_path)


def test_questions_keep_training_repetition_and_unseen_wording_separate(inputs):
    glossary, records, questions = inputs
    assert len(glossary["terms"]) == 58
    assert len(records) == 116
    assert [sum(q["category"] == c for q in questions) for c in experiment.CATEGORIES] == [6, 6, 3]
    changed = copy.deepcopy(questions)
    changed[0]["category"] = "paraphrase"
    with pytest.raises(ValueError, match="学習データと重複"):
        experiment.validate_inputs(glossary, records, changed)


def test_reference_information_never_reaches_generator(inputs, tmp_path):
    _, _, questions = inputs
    calls = []

    def generate(question):
        calls.append(experiment.build_messages(question))
        return "未確認の回答"

    summary, rows = experiment.evaluate_model(generate, questions, "before", tmp_path / "before.json")
    assert calls == [[{"role": "user", "content": q["question"]}] for q in questions]
    assert len(rows) == 15
    assert summary["manual_correct_rate"] is None
    assert json.loads((tmp_path / "before.json").read_text())["rows"][0]["reference_answer"]


def test_unlisted_terms_can_receive_credit_for_correct_answers(inputs, tmp_path):
    _, _, questions = inputs
    answers = {q["question"]: q["reference_answer"] for q in questions}
    summary, _ = experiment.evaluate_model(answers.__getitem__, questions, "after", tmp_path / "after.json")
    assert summary["by_category"]["not_in_glossary"]["keyword_rate"] == 1
    assert summary["manual_correct_rate"] is None


def test_real_search_reports_missing_terms_without_inventing_definitions(inputs, tmp_path):
    glossary, _, questions = inputs
    search = experiment.load_search_function(ROOT / "app/tools.py", tmp_path)
    summary, rows = experiment.evaluate_search(search, questions, glossary, tmp_path / "search.json")
    assert summary["listed_top1_hit_rate"] == 1
    assert summary["unlisted_not_found_rate"] == 1
    assert all(not r["hits"] and "見つかりません" in r["raw"] for r in rows if r["category"] == "not_in_glossary")
    assert rows[0]["found_term_id"] == "lora"
    assert len(rows[0]["hits"]) == 3  # 実際の関数は関連候補も返す。全候補を保持する。


def test_keyword_match_does_not_become_a_semantic_correctness_claim():
    row = {"id": "one", "category": "seen", "raw": "元の重みは固定しません。低ランクではありません。",
           **experiment.check_keywords("固定しません。低ランクではありません。", [["固定"], ["低ランク"]]),
           "manual_correct": None}
    assert row["keyword_ok"] is True
    assert experiment.summarize([row], "before", 0)["manual_correct_rate"] is None
    judged = experiment.apply_manual_review([row], {"one": False})
    assert experiment.summarize(judged, "before", 0)["manual_correct_rate"] == 0


def test_partial_review_stays_pending_and_invalid_verdicts_are_rejected():
    rows = [{"id": name, "category": "seen", "keyword_ok": True, "manual_correct": None} for name in ["a", "b"]]
    reviewed = experiment.apply_manual_review(rows, {"a": True})
    summary = experiment.summarize(reviewed, "before", 0)
    assert summary["manual_reviewed"] == 1
    assert summary["manual_correct_rate"] is None
    with pytest.raises(ValueError, match="不明な問題"):
        experiment.apply_manual_review(rows, {"typo": True})
    with pytest.raises(ValueError, match="True / False / None"):
        experiment.apply_manual_review(rows, {"a": "True"})


@pytest.mark.parametrize("with_reference", [False, True])
def test_notebook_bundle_loads_without_repo_access_and_exports_pending_review(tmp_path, with_reference):
    name = "lora_glossary_with_reference.ipynb" if with_reference else "lora_glossary_tuning.ipynb"
    notebook = json.loads((ROOT / "lectures/02/experiments/notebooks" / name).read_text())
    code_cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    for i, source in enumerate(code_cells):
        compile("\n".join(line for line in source.splitlines() if not line.lstrip().startswith("%")), f"cell{i}", "exec")
    namespace = {"WORK": tmp_path, "json": json, "hashlib": hashlib, "importlib": __import__("importlib")}
    bundle = next(source for source in code_cells if "BUNDLED_FILES = " in source)
    exec(bundle, namespace)
    portable = namespace["experiment"]
    glossary, records, questions = namespace["glossary"], namespace["train_records"], namespace["questions"]
    assert len(records) == 116
    assert namespace["search_glossary"]("LoRA")
    answers = {q["question"]: q["reference_answer"] for q in questions}
    b_summary, b_rows = portable.evaluate_model(answers.__getitem__, questions, "before", tmp_path / "eval_before.json")
    a_summary, a_rows = portable.evaluate_model(answers.__getitem__, questions, "after", tmp_path / "eval_after.json")
    s_summary, s_rows = portable.evaluate_search(namespace["search_glossary"], questions, glossary, tmp_path / "eval_search.json")
    fake_version = SimpleNamespace(__name__="smoke_test", __version__="not_a_real_training_run")
    namespace.update({
        "CONFIG": {"num_train_epochs": 20 if with_reference else 5}, "transformers": fake_version, "peft": fake_version,
        "trl": fake_version, "torch": fake_version, "TRAINING_SECONDS": 0.0,
        "trainer": SimpleNamespace(state=SimpleNamespace(global_step=0, epoch=0)),
        "LAST_LOGGED_TRAINING_LOSS": None, "LOSS_HISTORY": [], "TRAIN_LOGGING_STEPS": 5,
        "before_summary": b_summary, "before_rows": b_rows, "after_summary": a_summary, "after_rows": a_rows,
        "search_summary": s_summary, "search_rows": s_rows, "comparison_rows": [],
        "keyword_regressed": [], "MANUAL_VERDICTS": {"before": {}, "after": {}}, "USE_DRIVE": False,
    })
    if with_reference:
        calls = []

        def generate_answer(model, question, reference_text=None):
            calls.append(portable.build_messages(question, reference_text))
            return answers[question] if reference_text is not None else "資料なしのテスト用回答"

        namespace.update({
            "model": object(), "tuned": object(), "generate_answer": generate_answer, "COMPARE_WITHOUT_REFERENCE": True,
            "pd": SimpleNamespace(DataFrame=lambda rows: rows), "display": lambda value: None,
            "REFERENCE_PROMPT_MAX_TOKENS": 5000, "MODEL_CONTEXT_LIMIT": 32768,
        })
        before_source = next(source for source in code_cells if "def generate_answer(" in source)
        before_assignment = next(node for node in ast.parse(before_source).body
                                 if isinstance(node, ast.Assign)
                                 and any(isinstance(t, ast.Name) and t.id == "before_summary"
                                         for target in node.targets for t in ast.walk(target)))
        exec(compile(ast.Module(body=[before_assignment], type_ignores=[]), "before_reference", "exec"), namespace)
        compare = next(source for source in code_cells if source.startswith("after_summary, after_rows ="))
        exec(compare, namespace)
        assert len(calls) == 45
        assert all(namespace["REFERENCE_TEXT"] in messages[0]["content"] for messages in calls[:30])
        assert calls[:15] == calls[15:30]  # 学習前後に同じ資料と質問を渡す。
        assert calls[30:] == [[{"role": "user", "content": q["question"]}] for q in questions]
        assert namespace["after_summary"]["keyword_rate"] == 1
        assert namespace["no_reference_summary"]["keyword_rate"] == 0
        review = next(source for source in code_cells if source.startswith("MANUAL_VERDICTS ="))
        exec(review, namespace)
    export = next(source for source in code_cells if source.startswith("result = {"))
    exec(export, namespace)
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["model_receives_reference_at_inference"] is with_reference
    assert result["before"]["manual_correct_rate"] is None
    assert result["after"]["manual_correct_rate"] is None
    assert result["search"]["unlisted_not_found_rate"] == 1
    for name in ["evaluation.json", "comparison.json", "manual_review.json", "eval_before.json", "eval_after.json", "eval_search.json"]:
        assert (tmp_path / name).exists()
    if with_reference:
        assert result["config"]["num_train_epochs"] == 20
        assert result["training_receives_reference"] is False
        assert result["reference"]["terms"] == 58
        assert result["after_without_reference"]["n"] == 15
        assert (tmp_path / "eval_after_without_reference.json").exists()
        comparison = json.loads((tmp_path / "comparison.json").read_text())
        assert all(r["学習後・資料なし"] == "資料なしのテスト用回答" for r in comparison)


@pytest.mark.parametrize("name", ["lora_glossary_tuning.ipynb", "lora_glossary_with_reference.ipynb"])
def test_token_count_uses_token_ids_not_the_number_of_encoding_fields(name):
    notebook = json.loads((ROOT / "lectures/02/experiments/notebooks" / name).read_text())
    source = next("".join(c["source"]) for c in notebook["cells"] if "lengths = " in "".join(c.get("source", [])))
    tree = ast.parse(source)
    statements = [node for node in tree.body if isinstance(node, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id in {"training_texts", "lengths"} for t in node.targets)]

    class Tokenizer:
        def apply_chat_template(self, messages, tokenize, enable_thinking):
            text = messages[0]["content"]
            return self(text, False) if tokenize else text

        def __call__(self, text, add_special_tokens):
            assert add_special_tokens is False
            ids = list(range(len(text.split())))
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    namespace = {"tok": Tokenizer(), "train_records": [
        {"messages": [{"content": "one two three four"}]},
        {"messages": [{"content": "one two three four five six seven eight"}]},
    ]}
    exec(compile(ast.Module(body=statements, type_ignores=[]), "token_count", "exec"), namespace)
    assert namespace["lengths"] == [4, 8]


def test_reference_includes_all_glossary_entries_but_not_outside_gold_answers(inputs):
    glossary, _, questions = inputs
    text = experiment.format_glossary_reference(glossary)
    messages = experiment.build_messages("「LoRA」とは何ですか？", text)
    assert len(messages) == 1 and messages[0]["role"] == "user"
    assert messages[0]["content"].endswith("質問: 「LoRA」とは何ですか？")
    for term in glossary["terms"]:
        assert f"- {term['term']}: {term['definition']}" in messages[0]["content"]
    for q in questions:
        if q["category"] == "not_in_glossary":
            assert q["reference_answer"] not in messages[0]["content"]
    with pytest.raises(ValueError, match="資料は空でない"):
        experiment.build_messages("LoRAとは？", "")
