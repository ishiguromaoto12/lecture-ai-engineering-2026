"""保留例追加実験の入力ファイルと、再実行用の Colab ノートブックを作る。"""

import hashlib
import json
import shutil
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DEST = ROOT / "lectures/02/experiments/glossary/abstention20"
NOTEBOOK = ROOT / "lectures/02/experiments/notebooks/lora_glossary_abstention.ipynb"


def main():
    if NOTEBOOK.exists():
        n = json.loads(NOTEBOOK.read_text())
        if any(c.get("outputs") or c.get("execution_count") is not None for c in n["cells"]):
            raise SystemExit("実行済みのノートブックは上書きしません")
    inputs = DEST / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    files = {
        "inputs/glossary.json": (ROOT / "data/glossary.json").read_text(),
        "inputs/glossary_sft.jsonl": (ROOT / "lectures/02/experiments/glossary/glossary_sft.jsonl").read_text(),
        "inputs/evaluation.json": (ROOT / "lectures/02/experiments/glossary/evaluation.json").read_text(),
        "inputs/glossary_experiment.py": (ROOT / "lectures/02/experiments/helpers/glossary_experiment.py").read_text(),
        "inputs/glossary_abstention.py": (ROOT / "lectures/02/experiments/helpers/glossary_abstention.py").read_text(),
        "run_experiment.py": (ROOT / "lectures/02/experiments/scripts/run_glossary_abstention.py").read_text(),
    }
    config = {"base_model": "Qwen/Qwen3.5-0.8B", "revision": "2fc06364715b967f1860aea9cf38778875588b17",
              "num_train_epochs": 20, "seed": 42, "lr": 2e-4, "lora_r": 16, "lora_alpha": 32,
              "max_seq_len": 512, "max_new_tokens": 256}
    files["inputs/config.json"] = json.dumps(config, ensure_ascii=False, indent=2) + "\n"
    hashes = {name: hashlib.sha256(text.encode()).hexdigest() for name, text in files.items()}
    baseline = ROOT / "lectures/02/experiments/glossary/epochs20_with_reference"
    if not (DEST / "result.json").exists():
        for name, text in files.items():
            (DEST / name).write_text(text)
        (DEST / "input_manifest.json").write_text(json.dumps(hashes, indent=2) + "\n")
        for source, name in [("result.json", "baseline_result.json"), ("eval_after.json", "baseline_with_reference.json"),
                             ("eval_after_without_reference.json", "baseline_without_reference.json")]:
            shutil.copy2(baseline / source, DEST / name)
    else:
        print('保存済みの実験入力・実行コード・比較元は変更しません:', DEST.relative_to(ROOT))
    cells = []
    def add(kind, text):
        c = {"cell_type": kind, "id": uuid.uuid4().hex[:8], "metadata": {}, "source": text.strip().splitlines(keepends=True)}
        if kind == "code": c.update(execution_count=None, outputs=[])
        cells.append(c)
    add("markdown", """# 発展編・保留例を追加する実験

前回と同じ用語集 Q&A 116 件に、別の未掲載語 8 語 × 2 通り＝16 件の保留例を追加する。
評価用のドロップアウト・勾配降下法・知識蒸留は学習に使わない。合計 132 件を基盤モデルから 20 エポック学習する。
この訓練は「学習資料にない語で、説明を作らず確認を求める」例を教えるもので、モデルの内面の確信度を直接測るものではない。

学習後は資料あり・資料なしで同じ 15 問に答え、前回の定義例だけの 20 エポックと比較する。
**内容の正解・保留・誤断定・回答不足を別々に判定する。保留は内容正解には数えないが、誤断定にも数えない。**
語句検出だけで誤断定率を決めず、原文の意味と根拠を確認する。用語集にある質問まで保留していないかも見る。

Colab の T4 でカーネルを再起動し、上から実行する。学習は約十数分を見込み、実測値を保存する。
今回の Codex による実行は `run_experiment.py` を Colab 端末で実行し、ログと結果を保存する。
""")
    add("markdown", "## 0. 環境\n\n元の演習と同じ版のライブラリを導入する。")
    original = json.loads((ROOT / "notebooks/lora_format_tuning.ipynb").read_text())
    pins = next("".join(c["source"]) for c in original["cells"] if "PINS = {" in "".join(c.get("source", [])))
    add("code", pins)
    add("markdown", "## 1. 入力を展開する\n\n用語集・Q&A・評価問題・実行用コードを同梱済み。元の結果とは別の保存先を使う。")
    payload = json.dumps(files, ensure_ascii=False, indent=2)
    assert "'''" not in payload
    add("code", "from pathlib import Path\nimport hashlib, importlib.util, json\nWORK = Path('/content/glossary_abstention20_lab')\nWORK.mkdir(parents=True, exist_ok=True)\nBUNDLE = json.loads(r'''" + payload + "''')\nEXPECTED_SHA256 = " + repr(hashes) + "\n" + """
for name, text in BUNDLE.items():
    data = text.encode('utf-8')
    assert hashlib.sha256(data).hexdigest() == EXPECTED_SHA256[name]
    p = WORK / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
print('保存先:', WORK)
print((WORK / 'inputs/config.json').read_text())
""")
    add("markdown", "## 2. 追加する保留例を見る\n\nこの 16 件に、評価用の 3 語は含まれない。正解の定義を作らず、確認を求める例を追加する。")
    add("code", """
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
experiment = load('experiment', WORK / 'inputs/glossary_experiment.py')
abstention = load('abstention', WORK / 'inputs/glossary_abstention.py')
glossary, records, questions = experiment.load_inputs(WORK / 'inputs')
augmented, additions = abstention.build_augmented_records(records, glossary, questions)
print('定義:', len(records), '保留:', len(additions), '合計:', len(augmented))
for r in additions:
    print(r['messages'][0]['content'], '→', r['messages'][1]['content'])
""")
    add("markdown", "## 3. 学習・資料ありと資料なしの評価\n\n新しい基盤モデルから学習し、実測値と回答原文を保存する。結果が既にある場合は上書きせず停止する。")
    add("code", "runner = load('runner', WORK / 'run_experiment.py')\nrunner.run(WORK, executed_training_myself=True)")
    add("markdown", "## 4. 結果を見る\n\nキーワード一致と保留語句の有無は補助。誤断定率は原文を読んで確認した後に求める。")
    add("code", """
import pandas as pd
pd.set_option('display.max_colwidth', None)
result = json.loads((WORK / 'result.json').read_text())
print(json.dumps(result, ensure_ascii=False, indent=2))
for label in ['with_reference', 'without_reference']:
    data = json.loads((WORK / f'eval_{label}.json').read_text())
    print(label)
    display(pd.DataFrame(data['rows'])[['id', 'question', 'raw', 'reference_answer', 'abstention_phrase_detected']])
""")
    add("markdown", """## 5. 意味を確認する基準

- correct：質問に必要な内容を正しく説明し、重要な誤りを含まない。
- abstain：定義を作らず、分からない・確認が必要と明示する。
- incorrect_assertion：根拠のない定義や、誤った事実を説明する。保留の語句があっても、その後に誤った定義を続ければこちら。
- incomplete：正しい断片はあるが質問に十分答えず、明確な誤断定とも言えない。

分母は全問題数。内容正解率＝correct / 全問、保留率＝abstain / 全問、誤断定率＝incorrect_assertion / 全問。
掲載語 12 問と未掲載語 3 問を分けて集計し、必要な回答まで保留していないかも確認する。
未判定のまま、語句検出から内容正解率や誤断定率を確定しない。
""")
    add("markdown", "## 6. 比較\n\nローカルの `baseline_with_reference.json` と `baseline_without_reference.json` は前回の 20 エポックの結果。質問と用語集は同じ。今回は保留例の追加で総件数と学習ステップ数も増えるため、その影響も含む比較になる。")
    add("markdown", "## 7. 保存\n\n`/content/glossary_abstention20_lab` の JSON・`execution.log`・学習済み adapter を保存する。ローカルの保存先は `lectures/02/experiments/glossary/abstention20/`。Codex が代行した実行か、本人がノートブックから実行したかも記録する。")
    notebook = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
                "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                             "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                             "language_info": {"name": "python"}}}
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
    print('作成:', NOTEBOOK.relative_to(ROOT), '入力:', DEST.relative_to(ROOT))


if __name__ == '__main__':
    main()
