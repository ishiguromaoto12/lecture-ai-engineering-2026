import json
from pathlib import Path

import pytest

from glossary_abstention import build_augmented_records, review_rows, reviewed_summary, augment_observations

ROOT = Path(__file__).resolve().parents[4]


def test_abstention_training_never_uses_evaluation_terms():
    glossary = json.loads((ROOT / 'data/glossary.json').read_text())
    records = [json.loads(line) for line in (ROOT / 'lectures/02/experiments/glossary/glossary_sft.jsonl').read_text().splitlines()]
    questions = json.loads((ROOT / 'lectures/02/experiments/glossary/evaluation.json').read_text())
    result, additions = build_augmented_records(records, glossary, questions)
    assert len(result) == 132 and len(additions) == 16
    assert result[:116] == records
    texts = '\n'.join(r['messages'][0]['content'] for r in additions)
    assert all(q['query'] not in texts for q in questions if q['category'] == 'not_in_glossary')
    changed = [dict(questions[0], query='バッチ正規化')]
    with pytest.raises(ValueError, match='評価語と重複'):
        build_augmented_records(records, glossary, changed)


def test_abstaining_is_not_a_factual_answer_and_not_an_incorrect_assertion():
    rows = augment_observations([{'id': 'a', 'raw': '確認が必要です。'}, {'id': 'b', 'raw': '誤った定義です。'}])
    pending = reviewed_summary(rows)
    assert pending['content_correct_rate'] is None and pending['incorrect_assertion_rate'] is None
    reviewed = review_rows(rows, {'a': {'verdict': 'abstain', 'reason': '定義を作っていない'},
                                  'b': {'verdict': 'incorrect_assertion', 'reason': '定義に誤りがある'}})
    summary = reviewed_summary(reviewed)
    assert summary['content_correct_rate'] == 0
    assert summary['abstention_rate'] == .5
    assert summary['incorrect_assertion_rate'] == .5


def test_abstention_phrase_does_not_hide_a_false_definition():
    rows = augment_observations([{'id': 'a', 'raw': '確認が必要ですが、勾配降下法は植物の栽培法です。'}])
    assert rows[0]['abstention_phrase_detected']
    reviewed = review_rows(rows, {'a': {'verdict': 'incorrect_assertion', 'reason': '保留語句の後に誤った断定'}})
    assert reviewed_summary(reviewed)['incorrect_assertion_rate'] == 1
