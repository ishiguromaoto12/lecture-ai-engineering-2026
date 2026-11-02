"""第2回・発展編のデータ検証、自然文の評価、用語集検索の比較。

モデルに渡す入力と、採点用の正解を分離する。キーワード判定は一次確認で、
意味の正しさは回答原文を見て手動判定する。GPU や外部 API は使わない。
"""

import importlib.util
import json
import time
import unicodedata
from pathlib import Path


CATEGORIES = ("seen", "paraphrase", "not_in_glossary")
CATEGORY_LABELS = {
    "seen": "学習と同じ質問",
    "paraphrase": "言い換えた質問",
    "not_in_glossary": "用語集にない語",
}


def normalize(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).casefold().split())


def format_glossary_reference(glossary: dict) -> str:
    """掲載された全用語の名前と定義を資料にする。評価問題・正解は使わない。"""
    return "\n".join(f"- {term['term']}: {term['definition']}" for term in glossary["terms"])


def build_messages(question: str, reference_text: str | None = None) -> list[dict]:
    """通常は質問のみ。資料ありの比較では用語集全文も渡す。採点用の正解は渡さない。"""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("質問は空でない文字列にしてください")
    if reference_text is not None:
        if not isinstance(reference_text, str) or not reference_text.strip():
            raise ValueError("資料は空でない文字列にしてください")
        content = f"用語集を参考に、質問に普通の文章で答えてください。\n\n用語集:\n{reference_text}\n\n質問: {question}"
    else:
        content = question
    return [{"role": "user", "content": content}]


def load_inputs(folder: Path) -> tuple[dict, list[dict], list[dict]]:
    glossary = json.loads((folder / "glossary.json").read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (folder / "glossary_sft.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    questions = json.loads((folder / "evaluation.json").read_text(encoding="utf-8"))
    validate_inputs(glossary, records, questions)
    return glossary, records, questions


def validate_inputs(glossary: dict, records: list[dict], questions: list[dict]) -> None:
    terms = glossary["terms"]
    by_id = {t["id"]: t for t in terms}
    if len(by_id) != len(terms) or not records or not questions:
        raise ValueError("用語 ID の重複、または空のデータがあります")
    train_questions = set()
    definitions = {t["definition"] for t in terms}
    for record in records:
        messages = record["messages"]
        if [m["role"] for m in messages] != ["user", "assistant"]:
            raise ValueError("学習データは user と assistant の 2 メッセージです")
        if any(not isinstance(m["content"], str) or not m["content"].strip() for m in messages):
            raise ValueError("空の質問・回答があります")
        if messages[1]["content"] not in definitions:
            raise ValueError("用語集にない定義が学習データに含まれています")
        train_questions.add(messages[0]["content"])
    if len(train_questions) != len(records):
        raise ValueError("学習用の質問文が重複しています")

    all_names = {
        normalize(name)
        for term in terms
        for name in [term["term"], term["id"], *term["aliases"]]
    }
    ids, texts = set(), set()
    for q in questions:
        if q["id"] in ids or q["question"] in texts:
            raise ValueError("評価問題の ID・質問文が重複しています")
        ids.add(q["id"])
        texts.add(q["question"])
        if q["category"] not in CATEGORIES:
            raise ValueError("不明な評価カテゴリです")
        if not q["query"] or normalize(q["query"]) not in normalize(q["question"]):
            raise ValueError("検索語は問題文に実際に書かれた語にしてください")
        groups = q["keyword_groups"]
        if not groups or any(not g or any(not w for w in g) for g in groups):
            raise ValueError("キーワードの判定条件が空です")
        if not isinstance(q["reference_answer"], str) or not q["reference_answer"]:
            raise ValueError("参照回答がありません")
        if q["category"] == "not_in_glossary":
            if q["expected_term_id"] is not None or normalize(q["query"]) in all_names:
                raise ValueError("未掲載の語が用語集の名前・別名と一致しています")
            if q["question"] in train_questions:
                raise ValueError("未掲載語の質問が学習に混ざっています")
        else:
            term = by_id[q["expected_term_id"]]
            if q["reference_answer"] != term["definition"]:
                raise ValueError("掲載語の参照回答が用語集と一致しません")
            if normalize(q["query"]) not in {
                normalize(name) for name in [term["term"], term["id"], *term["aliases"]]
            }:
                raise ValueError("検索語が対象用語の名前・別名に一致しません")
        if q["category"] == "seen" and q["question"] not in train_questions:
            raise ValueError("seen の質問が学習データにありません")
        if q["category"] == "paraphrase" and q["question"] in train_questions:
            raise ValueError("言い換え問題が学習データと重複しています")


def check_keywords(answer: str, groups: list[list[str]]) -> dict:
    """各グループから少なくとも 1 語があるか。意味の正解を保証しない。"""
    text = normalize(answer)
    missing = [group for group in groups if not any(normalize(word) in text for word in group)]
    return {"keyword_ok": not missing, "missing_keyword_groups": missing}


def summarize(rows: list[dict], label: str, seconds: float) -> dict:
    def stats(items):
        judged = [r for r in items if r.get("manual_correct") is not None]
        return {
            "n": len(items),
            "keyword_rate": sum(r["keyword_ok"] for r in items) / len(items) if items else None,
            "manual_reviewed": len(judged),
            # 未判定を誤答や正答として数えない。全問判定後だけ正解率を出す。
            "manual_correct_rate": (
                sum(r["manual_correct"] for r in judged) / len(items)
                if items and len(judged) == len(items) else None
            ),
        }
    return {
        "label": label,
        **stats(rows),
        "seconds": round(seconds, 3),
        "by_category": {category: stats([r for r in rows if r["category"] == category]) for category in CATEGORIES},
    }


def evaluate_model(generate, questions: list[dict], label: str, output: Path) -> tuple[dict, list[dict]]:
    rows = []
    started = time.perf_counter()
    for q in questions:
        # generate に渡せるのは質問の文字列だけ。正解は生成後に採点に使う。
        answer = generate(q["question"])
        rows.append({
            "id": q["id"], "category": q["category"], "question": q["question"],
            "raw": answer, "reference_answer": q["reference_answer"],
            **check_keywords(answer, q["keyword_groups"]), "manual_correct": None,
        })
        print(f"{label}: {len(rows)}/{len(questions)} {q['id']}")
    summary = summarize(rows, label, time.perf_counter() - started)
    write_json(output, {"summary": summary, "rows": rows})
    return summary, rows


def apply_manual_review(rows: list[dict], verdicts: dict) -> list[dict]:
    known = {r["id"] for r in rows}
    if set(verdicts) - known:
        raise ValueError("手動判定に不明な問題 ID があります")
    if any(value is not None and type(value) is not bool for value in verdicts.values()):
        raise ValueError("手動判定は True / False / None で指定してください")
    return [{**r, "manual_correct": verdicts.get(r["id"])} for r in rows]


def load_search_function(tools_file: Path, data_dir: Path):
    """既存 app/tools.py をそのまま読み、用語集の保存先だけを差し替える。"""
    spec = importlib.util.spec_from_file_location("glossary_search_snapshot", tools_file)
    if spec is None or spec.loader is None:
        raise ValueError("検索用の Python ファイルを読み込めません")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.DATA_DIR = data_dir
    module.load_glossary.cache_clear()
    return module.search_glossary


def evaluate_search(search, questions: list[dict], glossary: dict, output: Path) -> tuple[dict, list[dict]]:
    by_name = {t["term"]: t["id"] for t in glossary["terms"]}
    started, rows = time.perf_counter(), []
    for q in questions:
        # query は問題文に書かれた用語。expected_term_id や参照回答は検索に渡さない。
        raw = search(q["query"])
        try:
            hits = json.loads(raw)
        except json.JSONDecodeError:
            hits = []
        first = hits[0] if hits else None
        found_id = by_name[first["term"]] if first else None
        rows.append({
            "id": q["id"], "category": q["category"], "question": q["question"],
            "query": q["query"], "raw_search_result": raw,
            "raw": first["definition"] if first else raw,
            "hits": hits, "found_term_id": found_id,
            "retrieval_ok": found_id == q["expected_term_id"] if q["expected_term_id"] else None,
        })
    listed = [r for r in rows if r["category"] != "not_in_glossary"]
    unlisted = [r for r in rows if r["category"] == "not_in_glossary"]
    summary = {
        "label": "search", "n": len(rows), "seconds": round(time.perf_counter() - started, 3),
        "listed_n": len(listed),
        "listed_top1_hit_rate": sum(r["retrieval_ok"] for r in listed) / len(listed) if listed else None,
        "unlisted_n": len(unlisted),
        "unlisted_not_found_rate": sum(not r["hits"] for r in unlisted) / len(unlisted) if unlisted else None,
    }
    write_json(output, {"summary": summary, "rows": rows})
    return summary, rows


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
