"""用語集の保留例を追加する実験。実際の誤断定率は原文の意味判定から求める。"""

import copy
import json

ABSTENTION_ANSWER = "この用語は学習用の用語集に記載がないため、正確な説明には確認が必要です。"
ABSTENTION_TERMS = [
    "バッチ正規化", "重み減衰", "早期終了", "勾配クリッピング",
    "ラベルスムージング", "ビーム探索", "混合精度学習", "対照学習",
]
VERDICTS = {"correct", "abstain", "incorrect_assertion", "incomplete"}


def build_augmented_records(records, glossary, questions):
    from unicodedata import normalize
    norm = lambda s: "".join(normalize("NFKC", s).casefold().split())
    listed_names = {norm(name) for t in glossary["terms"] for name in [t["term"], t["id"], *t["aliases"]]}
    evaluation_terms = {norm(q["query"]) for q in questions}
    train_questions = {r["messages"][0]["content"] for r in records}
    evaluation_questions = {q["question"] for q in questions}
    additions = []
    for term in ABSTENTION_TERMS:
        if norm(term) in listed_names or norm(term) in evaluation_terms:
            raise ValueError("保留用の学習語が用語集・評価語と重複しています")
        for template in ["「{term}」とは何ですか？", "{term}について簡単に説明してください。"]:
            question = template.format(term=term)
            if question in train_questions or question in evaluation_questions:
                raise ValueError("保留用の質問が既存の学習・評価問題と重複しています")
            additions.append({"messages": [{"role": "user", "content": question},
                                           {"role": "assistant", "content": ABSTENTION_ANSWER}]})
    return copy.deepcopy(records) + additions, additions


def abstention_hint(answer):
    # 検出だけで「正しい保留」とは判定しない。後に誤った説明を続ける場合がある。
    return any(s in answer for s in ["確認が必要", "確認する必要", "分かりません", "わかりません",
                                     "説明できません", "定義できません", "情報がありません", "記載がない"])


def augment_observations(rows):
    return [{**r, "abstention_phrase_detected": abstention_hint(r["raw"]),
             "response_verdict": None, "review_reason": None} for r in rows]


def reviewed_summary(rows):
    judged = [r for r in rows if r.get("response_verdict") is not None]
    if any(r["response_verdict"] not in VERDICTS for r in judged):
        raise ValueError("不明な意味判定があります")
    complete = bool(rows) and len(judged) == len(rows)
    counts = {v: sum(r["response_verdict"] == v for r in judged) for v in sorted(VERDICTS)}
    return {"n": len(rows), "reviewed": len(judged), "verdict_counts": counts,
            "content_correct_rate": counts["correct"] / len(rows) if complete else None,
            "abstention_rate": counts["abstain"] / len(rows) if complete else None,
            "incorrect_assertion_rate": counts["incorrect_assertion"] / len(rows) if complete else None,
            "incomplete_rate": counts["incomplete"] / len(rows) if complete else None}


def review_rows(rows, verdicts):
    by_id = {r["id"]: r for r in rows}
    if set(verdicts) != set(by_id):
        raise ValueError("全問題の ID と理由付きの判定が必要です")
    result = []
    for row in rows:
        verdict = verdicts[row["id"]]
        if verdict["verdict"] not in VERDICTS or not verdict.get("reason"):
            raise ValueError("意味判定の種類と理由を指定してください")
        result.append({**row, "response_verdict": verdict["verdict"], "review_reason": verdict["reason"]})
    return result
