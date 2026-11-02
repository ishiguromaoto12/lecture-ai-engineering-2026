"""第2回・発展編の評価問題と、Colab / VS Code で開くノートブックを作る。

実行: python lectures/02/experiments/scripts/build_glossary_notebook.py
資料あり・20エポック版: python lectures/02/experiments/scripts/build_glossary_notebook.py --with-reference
学習は実行しない。入力ファイルをノートブックに同梱するため、Colab への
個別アップロードや、未コミットのファイルを GitHub から取得する操作は不要。
"""

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
DEST = ROOT / "lectures/02/experiments/glossary"
NOTEBOOK = ROOT / "lectures/02/experiments/notebooks/lora_glossary_tuning.ipynb"
REFERENCE_NOTEBOOK = ROOT / "lectures/02/experiments/notebooks/lora_glossary_with_reference.ipynb"


def evaluation_questions(glossary):
    by_id = {t["id"]: t for t in glossary["terms"]}
    selected = [
        ("lora", "LoRAでは元の重みをどう扱い、何を学習しますか？",
         [["固定", "更新しない", "凍結"], ["低ランク", "小さな行列", "低いランク"]]),
        ("qlora", "QLoRAはどのようにメモリを節約して追加学習しますか？",
         [["量子化"], ["4ビット", "4bit", "4-bit"], ["LoRA"]]),
        ("overfitting", "過学習が起きると、学習に使った問題と未知の問題への対応はどうなりますか？",
         [["学習データ", "訓練データ", "学習に使った"], ["未知", "新しい", "未学習", "汎化"], ["落ち", "低下", "悪く"]]),
        ("lec-start-tag", "lecNN-start タグを使うと、前の回を終えていなくても何ができますか？",
         [["タグ"], ["開始", "始め", "スタート"], ["演習", "コード"]]),
        ("homework-levels", "宿題の3レベルには何があり、全部取り組む必要がありますか？",
         [["基礎"], ["実装"], ["発展"], ["1つ", "一つ", "どれか", "いずれか"]]),
        ("answer-md", "answer.mdには何を書き、どうやって提出しますか？",
         [["演習", "宿題"], ["結果", "気づき", "考察"], ["PR", "プルリクエスト"]]),
    ]
    result = []
    for term_id, paraphrase, groups in selected:
        term = by_id[term_id]
        for category, question in [
            ("seen", f"「{term['term']}」とは何ですか？"),
            ("paraphrase", paraphrase),
        ]:
            result.append({
                "id": f"{term_id}_{category}", "category": category,
                "domain": "general" if term_id in {"lora", "qlora", "overfitting"} else "course_specific",
                "question": question, "query": term["term"],
                "expected_term_id": term_id, "reference_answer": term["definition"],
                "keyword_groups": groups, "reference_source": "data/glossary.json",
            })
    # 未掲載という理由では誤答にしない。正解の概念を外部の一次資料から別途用意。
    outside = [
        ("dropout", "ドロップアウト", "学習中にニューロンなどをランダムに無効化し、過学習を抑える正則化手法。",
         [["ランダム", "確率的"], ["無効", "除外", "落と", "ゼロ", "使わない"], ["過学習", "正則化"]],
         "https://www.jmlr.org/papers/v15/srivastava14a.html"),
        ("gradient-descent", "勾配降下法", "損失関数の勾配を使い、損失が小さくなる方向へ重みなどのパラメータを繰り返し更新する最適化手法。",
         [["勾配", "微分"], ["損失", "誤差", "目的関数"], ["更新", "調整"], ["減ら", "小さ", "最小", "下が", "減少"]],
         "https://developers.google.com/machine-learning/crash-course/linear-regression/gradient-descent"),
        ("distillation", "知識蒸留", "教師モデルの出力などを使って生徒モデルを学習させ、知識を移す手法。モデルを小さくする用途などに使う。",
         [["教師", "teacher"], ["生徒", "student", "小さいモデル", "小さなモデル"], ["学習", "訓練"], ["出力", "予測", "分布", "ソフトラベル"]],
         "https://arxiv.org/abs/1503.02531"),
    ]
    for term_id, term, reference, groups, source in outside:
        result.append({
            "id": f"{term_id}_outside", "category": "not_in_glossary", "domain": "general",
            "question": f"「{term}」とは何ですか？", "query": term,
            "expected_term_id": None, "reference_answer": reference,
            "keyword_groups": groups, "reference_source": source,
        })
    return result


def cell(kind, source):
    result = {"cell_type": kind, "id": uuid.uuid4().hex[:8], "metadata": {}, "source": source.strip().splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def reference_variant(notebook):
    """学習データを変えず、20 エポックと回答時の用語集全文を比較する版。"""
    notebook = copy.deepcopy(notebook)
    intro = """# 第2回・発展編の追加実験：20 エポック＋回答時に用語集全文を渡す

普通の文章で回答します。学習には前回と同じ 116 件の「質問→定義文」を使い、20 エポック学習します。
**学習前・学習後の評価では、同じ用語集 58 語の全文と質問を渡します。**
資料は検索で絞り込まず、すべての用語名と定義文です。評価専用の正解や未掲載語の説明は資料に追加しません。

20 エポック学習後には「質問だけで答える」条件も測ります。同じモデルで資料の有無を比べることで、
資料を渡す効果を確かめられます。`COMPARE_WITHOUT_REFERENCE=False` なら追加評価を省略できます。
元の 5 エポック・資料なしの結果は `lectures/02/experiments/glossary/epochs5/` に保存済みです。

**使い方**: このノートブックで Colab の T4 カーネルを選び、カーネルを再起動してセクション 0〜7 を順に実行します。
データは同梱済みで、個別アップロードは不要です。
20 エポックは前回の 5 エポックより学習回数が多く、用語集全文を渡す評価も入力が長くなるため、所要時間は実測してください。
"""
    notes = {
        1: """## 1. 設定

20 エポックにします。学習データは同じ 116 件で、LoRA のランク・学習率などは前回と同じです。
`COMPARE_WITHOUT_REFERENCE=True` なら、学習後に資料なしでも同じ 15 問へ回答します。
出力先は `/content/glossary_lora_lab_with_reference` で、前回分と分けて保存します。
""",
        2: """## 2. 学習データ・評価問題・資料全文・検索の準備

同梱ファイルを展開します。学習には `glossary_sft.jsonl` の 116 件をそのまま使います。
**用語集全文は、モデルに回答してもらうときに渡す資料です。学習用の質問には追加しません。**
`REFERENCE_TEXT` は `glossary.json` の全 58 語の名前と定義から作ります。検索結果や評価専用の参照回答は使いません。
評価問題は前回と同じ 15 問です。用語集にない 3 語の参照回答は、採点のためだけに使います。
""",
        3: """## 3. モデルの読み込みと学習前の回答（資料あり）

資料全文と質問を 1 つの user メッセージにして渡します。各質問で同じ資料を使い、途中で切り捨てません。
学習例の長さと、資料付きの入力長は別に確認します。入力と出力の合計がモデルの上限を超える場合は停止します。
正解 ID や採点用の語句、未掲載語の参照回答はモデルに渡しません。
キーワード一致は一次確認であり、意味の内容正解率ではありません。
""",
        4: """## 4. 同じ用語集 Q&A で 20 エポック学習

学習データは前回と同じ「質問→定義文」の 116 件です。資料全文を付けた学習データには変更しません。
前回から学習条件で変更するのはエポック数だけです。学習時間・loss・ステップ数を記録します。
この学習の後に、同じモデルを資料あり・資料なしの両条件で評価します。
""",
        5: """## 5. 学習後の回答と資料の有無・検索を比較する

最初に学習後のモデルへ資料全文と質問を渡します。追加で、同じモデルに質問だけを渡します。
これにより「学習前→学習後（どちらも資料あり）」と「20 エポック学習後の資料あり→なし」を比較できます。
前回の 5 エポック・資料なしと今回の資料ありを比べると、エポック数と入力資料の両方が変わることに注意してください。

検索版は前回と同じ文字列検索で、問題に書かれた用語を検索し、先頭の定義をそのまま表示します。
検索対象は用語集だけで、モデルが参照する資料全文も同じ用語集から作っています。
未掲載語については、資料にないことを示せるか、説明する場合に内容が正しいかを原文で確認します。
""",
        6: """## 6. 原文を読んで内容を確認する

資料ありの学習前・学習後、追加評価の資料なしを読み比べます。
正しい語句があっても、否定・矛盾・資料にない創作が混ざっていないかを確認してください。
`MANUAL_VERDICTS` に問題 ID と `True` / `False` を書くと手動判定を保存できます。未確認は空か `None` のままにします。
全問を手動確認するまでは内容正解率を `null` として保存します。キーワードの一致率とは別の指標です。
""",
        7: """## 7. 比較結果を保存する

`result.json` に、20 エポック・資料ありという条件、資料の語数と入力長、学習時間、loss、各条件の評価を保存します。
`eval_before.json` と `eval_after.json` は資料あり、`eval_after_without_reference.json` は学習後の資料なしです。
回答原文の比較は `comparison.json`、手動判定は `manual_review.json` に保存します。
保存先は `/content/glossary_lora_lab_with_reference` です。実行後はノートブックと結果ファイルを保存してください。
ローカルの提出用フォルダーは `lectures/02/experiments/glossary/epochs20_with_reference/` として、前回の結果を残せます。
""",
    }
    for c in notebook["cells"]:
        source = "".join(c["source"])
        if c["cell_type"] == "markdown":
            if source.startswith("# 第2回"):
                source = intro
            else:
                for section, text in notes.items():
                    if source.startswith(f"## {section}."):
                        source = text
                        break
        else:
            c.update(execution_count=None, outputs=[])
            if "CONFIG = {" in source:
                source = source.replace('"num_train_epochs": 5,', '"num_train_epochs": 20,')
                source += "\nCOMPARE_WITHOUT_REFERENCE = True\n"
            if "BUNDLED_FILES = " in source:
                source += '\nREFERENCE_TEXT = experiment.format_glossary_reference(glossary)\nprint("資料に含まれる用語数:", len(glossary["terms"]), "文字数:", len(REFERENCE_TEXT))\n'
            if "def generate_answer(" in source:
                source = source.replace("def generate_answer(model, question):", "def generate_answer(model, question, reference_text=None):")
                source = source.replace("experiment.build_messages(question)", "experiment.build_messages(question, reference_text)")
                source = source.replace('tok(prompt, return_tensors="pt")', 'tok(prompt, return_tensors="pt", truncation=False)')
                source = source.replace(
                    "before_summary, before_rows = experiment.evaluate_model(",
                    '''reference_prompts = [tok.apply_chat_template(
    experiment.build_messages(q["question"], REFERENCE_TEXT),
    tokenize=False, add_generation_prompt=True, enable_thinking=False,
) for q in questions]
REFERENCE_PROMPT_MAX_TOKENS = max(len(tok(p, add_special_tokens=False)["input_ids"]) for p in reference_prompts)
text_config = getattr(model.config, "text_config", model.config)
MODEL_CONTEXT_LIMIT = getattr(text_config, "max_position_embeddings", None)
print("資料付き入力の最大トークン数:", REFERENCE_PROMPT_MAX_TOKENS, "モデル上限:", MODEL_CONTEXT_LIMIT)
if MODEL_CONTEXT_LIMIT is not None and REFERENCE_PROMPT_MAX_TOKENS + CONFIG["max_new_tokens"] > MODEL_CONTEXT_LIMIT:
    raise ValueError("用語集全文と回答がモデルの入力上限を超えます。資料は自動では切り捨てません。")
before_summary, before_rows = experiment.evaluate_model(''',
                )
                source = source.replace("lambda q: generate_answer(model, q),", "lambda q: generate_answer(model, q, REFERENCE_TEXT),")
            if source.startswith("after_summary, after_rows ="):
                source = source.replace("lambda q: generate_answer(tuned, q),", "lambda q: generate_answer(tuned, q, REFERENCE_TEXT),")
                source = source.replace(
                    "search_summary, search_rows = experiment.evaluate_search(",
                    '''no_reference_summary, no_reference_rows = None, None
if COMPARE_WITHOUT_REFERENCE:
    no_reference_summary, no_reference_rows = experiment.evaluate_model(
        lambda q: generate_answer(tuned, q), questions, "after_without_reference",
        WORK / "eval_after_without_reference.json",
    )
search_summary, search_rows = experiment.evaluate_search(''',
                )
                source = source.replace('"条件": summary["label"],', '"条件": {"before": "学習前・資料あり", "after": "学習後・資料あり", "after_without_reference": "学習後・資料なし"}[summary["label"]],')
                source = source.replace("for summary in [before_summary, after_summary]", "for summary in [before_summary, after_summary] + ([no_reference_summary] if no_reference_summary else [])")
                source = source.replace('"学習前": b["raw"], "学習後": a["raw"],', '"学習前・資料あり": b["raw"], "学習後・資料あり": a["raw"],\n    "学習後・資料なし": no_reference_rows[i]["raw"] if no_reference_rows is not None else None,')
                source = source.replace("for q, b, a, s in zip(questions, before_rows, after_rows, search_rows)", "for i, (q, b, a, s) in enumerate(zip(questions, before_rows, after_rows, search_rows))")
            if source.startswith("MANUAL_VERDICTS ="):
                source = source.replace('"after": {},', '"after": {},\n    "after_without_reference": {},')
                source = source.replace('"学習前": b["raw"], "学習後": a["raw"],', '"学習前・資料あり": b["raw"], "学習後・資料あり": a["raw"],')
                source += '''
if no_reference_rows is not None:
    no_reference_rows = experiment.apply_manual_review(no_reference_rows, MANUAL_VERDICTS["after_without_reference"])
    no_reference_summary = experiment.summarize(no_reference_rows, "after_without_reference", no_reference_summary["seconds"])
    display(pd.DataFrame([{"id": r["id"], "質問": r["question"], "学習後・資料なし": r["raw"],
                           "参照回答": r["reference_answer"], "手動判定": r["manual_correct"]} for r in no_reference_rows]))
'''
            if source.startswith("result = {"):
                source = source.replace('"model_receives_reference_at_inference": False,', '"model_receives_reference_at_inference": True,\n    "training_receives_reference": False,\n    "reference": {"terms": len(glossary["terms"]), "characters": len(REFERENCE_TEXT),\n                  "max_prompt_tokens": REFERENCE_PROMPT_MAX_TOKENS, "model_context_limit": MODEL_CONTEXT_LIMIT},\n    "after_without_reference": no_reference_summary,')
                source = source.replace('"search_query": "問題文に書かれた用語。正解 ID や参照回答は検索に渡さない。",', '"search_query": "問題文に書かれた用語。正解 ID や参照回答は検索に渡さない。",\n        "model_reference": "学習前後とも用語集全文を渡す。評価専用の正解は追加しない。",\n        "comparison": "資料の有無は同じ20エポック学習後のモデルでも比較する。",')
                source = source.replace('if USE_DRIVE:\n', '''if no_reference_rows is not None:
    experiment.write_json(WORK / "eval_after_without_reference.json", {"summary": no_reference_summary, "rows": no_reference_rows})
if USE_DRIVE:
''')
        source = source.replace("/glossary_lora_lab", "/glossary_lora_lab_with_reference")
        # 上の説明文は既に保存先の名前を含むので、二重の接尾辞を避ける。
        source = source.replace("glossary_lora_lab_with_reference_with_reference", "glossary_lora_lab_with_reference")
        c["source"] = source.strip().splitlines(keepends=True)
    return notebook


def main(with_reference=False):
    output_notebook = REFERENCE_NOTEBOOK if with_reference else NOTEBOOK
    if output_notebook.exists():
        previous = json.loads(output_notebook.read_text(encoding="utf-8"))
        if any(c.get("outputs") or c.get("execution_count") is not None for c in previous["cells"]):
            raise SystemExit("実行結果のある発展ノートブックは上書きしません。先に別名で保存してください。")
    DEST.mkdir(parents=True, exist_ok=True)
    glossary_text = (ROOT / "data/glossary.json").read_text(encoding="utf-8")
    glossary = json.loads(glossary_text)
    generated = subprocess.run(
        [sys.executable, str(ROOT / "scripts/make_sft_dataset.py")],
        check=True, capture_output=True, text=True,
    ).stdout
    sft_path = DEST / "glossary_sft.jsonl"
    if sft_path.exists() and sft_path.read_text(encoding="utf-8") != generated:
        raise SystemExit("既存の glossary_sft.jsonl がスクリプトの出力と異なります。自動では上書きしません。")
    sft_path.write_text(generated, encoding="utf-8")
    evaluation = evaluation_questions(glossary)
    evaluation_text = json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n"
    (DEST / "evaluation.json").write_text(evaluation_text, encoding="utf-8")
    files = {
        "glossary.json": glossary_text,
        "glossary_sft.jsonl": generated,
        "evaluation.json": evaluation_text,
        "glossary_experiment.py": (ROOT / "lectures/02/experiments/helpers/glossary_experiment.py").read_text(encoding="utf-8"),
        "search_tools.py": (ROOT / "app/tools.py").read_text(encoding="utf-8"),
    }
    manifest = {name: hashlib.sha256(value.encode()).hexdigest() for name, value in files.items()}
    # 入力データの JSON と Python ソースの両方を、文字列として同梱する。
    payload = json.dumps(files, ensure_ascii=False, indent=2)
    assert "'''" not in payload
    original = json.loads((ROOT / "notebooks/lora_format_tuning.ipynb").read_text(encoding="utf-8"))
    pins = next("".join(c["source"]) for c in original["cells"] if "PINS = {" in "".join(c.get("source", [])))
    cells = []
    def md(text): cells.append(cell("markdown", text))
    def code(text): cells.append(cell("code", text))

    md("""
# 第2回・発展編：用語集を学習する方法と、検索する方法を比べる

講座の用語集 58 語から作った Q&A 116 件で LoRA 学習します。回答は普通の文章です。
**学習前・学習後とも、回答時のモデル入力は質問だけです。用語集・正解・採点用語句は渡しません。**

比較するのは、①元のモデル、②学習後のモデル、③用語集の文字列検索です。
検索版はモデルを使わず、`app/tools.py` の検索結果の先頭の定義をそのまま返します。

**使い方**: Colab の T4 カーネルを選び、カーネルを再起動してセクション 0〜7 を上から実行してください。
入力ファイルはこのノートブックに同梱済みです。VS Code の Colab 接続でも個別アップロードは不要です。
元ファイルは `lectures/02/experiments/glossary/glossary_sft.jsonl`、`data/glossary.json`、
`lectures/02/experiments/glossary/evaluation.json` です。同梱時点の内容と SHA-256 を結果に残します。

学習はまず **5 エポック**にしています。116 件あるため、図書館版の 20 エポックをそのまま使わず、
短い実験から始めるための設定です。性能や学習時間は実測してください。結果の保存先は `/content/glossary_lora_lab` です。

評価は 15 問です。「学習と同じ質問」6 問は記憶の確認、「言い換え」6 問は別の聞き方への対応、
「用語集にない語」3 問は既存の知識や断定の変化を観察します。未掲載でも正しく説明できれば正解です。
""")
    md("""
## 0. GPU とライブラリ

最初のセルで T4 が表示されることを確認します。ライブラリは元の演習と同じ版です。
インストール中の `cudf` / `pyarrow` の依存警告については元の演習と同じ扱いです。
""")
    code("""
import subprocess
try:
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"],
                         capture_output=True, text=True, check=True).stdout)
except (FileNotFoundError, subprocess.CalledProcessError):
    raise RuntimeError("Colab の T4 カーネルを選び、カーネルを再起動してください。")
""")
    code(pins)
    md("""
## 1. 設定

5 エポック・量子化なしの LoRA です。回答は自然文なので、生成上限は 256 トークンにします。
学習前後の生成条件は同じです。設定を変えて再学習するときはカーネルを再起動してください。
""")
    code("""
from pathlib import Path
import hashlib, importlib.util, json, math, time

CONFIG = {
    "base_model": "Qwen/Qwen3.5-0.8B",
    "revision": "2fc06364715b967f1860aea9cf38778875588b17",
    "seed": 42,
    "num_train_epochs": 5,
    "warmup_steps": 6,
    "lr": 2e-4,
    "lora_r": 16,
    "lora_alpha": 32,
    "max_seq_len": 512,
    "max_new_tokens": 256,
    "gen": {"do_sample": False},
    "work_dir": "/content/glossary_lora_lab",
    "drive_dir": "/content/drive/MyDrive/aie2026/glossary_lora_lab",
}
WORK = Path(CONFIG["work_dir"])
WORK.mkdir(parents=True, exist_ok=True)
USE_DRIVE = False
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    Path(CONFIG["drive_dir"]).mkdir(parents=True, exist_ok=True)
print(json.dumps(CONFIG, ensure_ascii=False, indent=2))
""")
    md("""
## 2. 学習データ・評価問題・検索の準備

次の大きなセルは、元のファイルを Colab に持ち込むための同梱データです。
通常は編集せず実行してください。`inputs/` に展開し、ファイルのハッシュを確認します。
`glossary_sft.jsonl` は `scripts/make_sft_dataset.py` の出力そのものです。

学習時には質問と定義文の対を使います。評価問題やその正解は学習に追加しません。
ただし「学習と同じ質問」の 6 問は、意図的に学習問題を再利用しています。
言い換えと未掲載語の質問文は、116 件の学習問題と重複しないことを確認します。
未掲載語の参照回答の出典は `evaluation.json` に記録しています。
""")
    code("# 元ファイルの内容を文字列として同梱。ここではモデルに入力しません。\n"
         + "BUNDLED_FILES = json.loads(r'''" + payload + "''')\n"
         + "EXPECTED_SHA256 = " + repr(manifest) + "\n"
         + """
INPUT_DIR = WORK / "inputs"
INPUT_DIR.mkdir(parents=True, exist_ok=True)
for name, text in BUNDLED_FILES.items():
    data = text.encode("utf-8")
    assert hashlib.sha256(data).hexdigest() == EXPECTED_SHA256[name], name
    (INPUT_DIR / name).write_bytes(data)
spec = importlib.util.spec_from_file_location("glossary_experiment", INPUT_DIR / "glossary_experiment.py")
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)
glossary, train_records, questions = experiment.load_inputs(INPUT_DIR)
search_glossary = experiment.load_search_function(INPUT_DIR / "search_tools.py", INPUT_DIR)
print(f"用語数={len(glossary['terms'])} / 学習例={len(train_records)} / 評価問題={len(questions)}")
print("学習例:", json.dumps(train_records[0], ensure_ascii=False, indent=2))
""")
    code("""
import pandas as pd
pd.set_option("display.max_colwidth", None)
display(pd.DataFrame([{
    "id": q["id"], "種類": experiment.CATEGORY_LABELS[q["category"]],
    "質問": q["question"], "検索語": q["query"], "参照回答": q["reference_answer"],
} for q in questions]))
print("検索例:", search_glossary("LoRA"))
""")
    md("""
## 3. モデルの読み込みと学習前の回答

モデルに渡すのは `[{"role": "user", "content": 質問}]` だけです。
正解は生成後に確認します。キーワードの一致率は一次確認であり、内容正解率ではありません。
否定や矛盾があっても語句だけは含められるので、セクション 6 で原文も読みます。
""")
    code("""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
if not torch.cuda.is_available():
    raise RuntimeError("このノートブックは Colab の CUDA GPU を使って実行してください。")
set_seed(CONFIG["seed"])
tok = AutoTokenizer.from_pretrained(CONFIG["base_model"], revision=CONFIG["revision"])
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
training_texts = [tok.apply_chat_template(r["messages"], tokenize=False, enable_thinking=False)
                  for r in train_records]
lengths = [len(tok(text, add_special_tokens=False)["input_ids"]) for text in training_texts]
print("学習例の最大トークン数:", max(lengths), "/ 上限:", CONFIG["max_seq_len"])
if max(lengths) > CONFIG["max_seq_len"]:
    raise ValueError("定義文を切り捨てないよう max_seq_len を増やして、上から実行し直してください。")
model = AutoModelForCausalLM.from_pretrained(
    CONFIG["base_model"], revision=CONFIG["revision"], dtype=torch.float32,
).to("cuda")
model.config.use_cache = True
model.eval()
BASE_MEMORY_GB = round(model.get_memory_footprint() / 1e9, 2)
print("loaded:", CONFIG["base_model"], "| 重みのメモリ(GB):", BASE_MEMORY_GB)

def generate_answer(model, question):
    messages = experiment.build_messages(question)
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=CONFIG["max_new_tokens"],
                             pad_token_id=tok.pad_token_id, **CONFIG["gen"])
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

before_summary, before_rows = experiment.evaluate_model(
    lambda q: generate_answer(model, q), questions, "before", WORK / "eval_before.json",
)
print(json.dumps(before_summary, ensure_ascii=False, indent=2))
display(pd.DataFrame(before_rows)[["id", "question", "raw", "keyword_ok"]])
""")
    md("""
## 4. 用語集 Q&A で LoRA 学習

116 件の `messages` をそのまま `Dataset.from_list` に渡します。
図書館版の FAQ や JSON のお手本は使いません。未掲載語の参照回答も学習には使いません。
学習時間・最終ログの loss・ステップ数を変数に残し、セクション 7 で保存します。
学習できなかった場合は、学習後の結果を作らず、実行できた範囲を記録してください。
""")
    code("""
from datasets import Dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

train_ds = Dataset.from_list(train_records)
model.config.use_cache = False
lora = LoraConfig(
    r=CONFIG["lora_r"], lora_alpha=CONFIG["lora_alpha"], lora_dropout=0.05,
    bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
TRAIN_LOGGING_STEPS = 5
sft = SFTConfig(
    output_dir=str(WORK / "ckpt"), num_train_epochs=CONFIG["num_train_epochs"],
    per_device_train_batch_size=4, gradient_accumulation_steps=2,
    learning_rate=CONFIG["lr"], lr_scheduler_type="cosine", warmup_steps=CONFIG["warmup_steps"],
    logging_steps=TRAIN_LOGGING_STEPS, save_steps=10, save_total_limit=2,
    fp16=False, max_length=CONFIG["max_seq_len"], packing=False, report_to="none", seed=CONFIG["seed"],
)
trainer = SFTTrainer(model=model, args=sft, train_dataset=train_ds,
                     peft_config=lora, processing_class=tok)
trainer.model.print_trainable_parameters()
started = time.perf_counter()
trainer.train()
TRAINING_SECONDS = round(time.perf_counter() - started, 1)
LOSS_HISTORY = [{"step": row["step"], "training_loss": row["loss"]}
                for row in trainer.state.log_history if "loss" in row]
LAST_LOGGED_TRAINING_LOSS = LOSS_HISTORY[-1]["training_loss"] if LOSS_HISTORY else None
print(f"学習時間: {TRAINING_SECONDS} 秒 | ステップ数: {trainer.state.global_step} | エポック: {trainer.state.epoch}")
ADAPTER_DIR = WORK / "adapter"
trainer.model.save_pretrained(ADAPTER_DIR)
tok.save_pretrained(ADAPTER_DIR)
tuned = trainer.model
tuned.config.use_cache = True
tuned.eval();
""")
    md("""
## 5. 学習後と用語集検索を比較する

学習後も質問だけを渡します。検索版は、問題文に書かれた用語を `query` として既存の検索関数に渡し、
先頭に返った定義を表示します。自然文から検索語を抽出する能力や、意味検索の性能は測りません。
検索に参照回答や正解 ID は渡しません。候補は最大 3 件で、全候補も結果ファイルに残します。

モデルの「キーワード一致率」と、検索の「正しい用語を先頭に見つけた割合」は別の指標です。
検索にない語は「見つからない」が期待されますが、モデルが正しく説明できる可能性はあります。
""")
    code("""
after_summary, after_rows = experiment.evaluate_model(
    lambda q: generate_answer(tuned, q), questions, "after", WORK / "eval_after.json",
)
search_summary, search_rows = experiment.evaluate_search(
    search_glossary, questions, glossary, WORK / "eval_search.json",
)
display(pd.DataFrame([{
    "条件": summary["label"],
    "同じ質問のキーワード一致率": summary["by_category"]["seen"]["keyword_rate"],
    "言い換えのキーワード一致率": summary["by_category"]["paraphrase"]["keyword_rate"],
    "未掲載語のキーワード一致率": summary["by_category"]["not_in_glossary"]["keyword_rate"],
    "回答生成時間（秒）": summary["seconds"],
} for summary in [before_summary, after_summary]]))
print("検索結果:", json.dumps(search_summary, ensure_ascii=False, indent=2))
comparison_rows = [{
    "id": q["id"], "種類": experiment.CATEGORY_LABELS[q["category"]], "質問": q["question"],
    "学習前": b["raw"], "学習後": a["raw"], "検索": s["raw"], "参照回答": q["reference_answer"],
} for q, b, a, s in zip(questions, before_rows, after_rows, search_rows)]
display(pd.DataFrame(comparison_rows))
""")
    md("""
## 6. 原文を読んで内容を確認する

参照回答はお手本です。一字一句の一致ではなく、意味が合っているか、誤った説明が混ざっていないかを確認します。
「学習と同じ質問」での改善と「言い換え」での改善を分けて見てください。
未掲載語では、正しい説明・分からないという応答・誤った断定を区別します。
元のモデルが知っていた場合もあるため、未掲載という理由だけで誤答にしません。

下の `MANUAL_VERDICTS` は任意で記入できます。例: `"lora_seen": True`。
`True` は内容が正しい、`False` は誤りまたは回答不足です。未確認は書かないか `None` にします。
正しい保留と誤った断定の違いは原文にも記録されています。
**全問の手動確認が終わるまでは、内容正解率を `null`（未確定）として保存します。**
手動確認を後で行う場合は、このセルとセクション 7 だけ再実行できます。
""")
    code("""
MANUAL_VERDICTS = {
    "before": {},  # 例: "lora_seen": True
    "after": {},
}
before_rows = experiment.apply_manual_review(before_rows, MANUAL_VERDICTS["before"])
after_rows = experiment.apply_manual_review(after_rows, MANUAL_VERDICTS["after"])
before_summary = experiment.summarize(before_rows, "before", before_summary["seconds"])
after_summary = experiment.summarize(after_rows, "after", after_summary["seconds"])
display(pd.DataFrame([{
    "id": q["id"], "質問": q["question"], "学習前": b["raw"], "学習後": a["raw"],
    "参照回答": q["reference_answer"], "学習前の手動判定": b["manual_correct"],
    "学習後の手動判定": a["manual_correct"],
} for q, b, a in zip(questions, before_rows, after_rows)]))
keyword_regressed = [b["id"] for b, a in zip(before_rows, after_rows)
                     if b["keyword_ok"] and not a["keyword_ok"]]
print("キーワード判定が通過から不通過になった問題（意味の退行かは原文確認）:", keyword_regressed)
""")
    md("""
## 7. 結果を保存する

`result.json` に条件・学習時間・loss・学習前後の要約・検索の要約を記録します。
各回答原文は `eval_before.json` / `eval_after.json` / `eval_search.json`、
評価問題と正解は `evaluation.json`、手動判定は `manual_review.json` に保存します。

保存先は `/content/glossary_lora_lab` です。実行後はノートブックを保存し、結果ファイルもローカルに保存してください。
提出用には `lectures/02/experiments/glossary/` に置きます。`USE_DRIVE=True` の場合は Drive にもコピーします。

考察では、定義の正確さ・言い換えへの対応・未掲載語の応答を比較し、用語集を更新したときに
「再学習する方法」と「検索するデータを書き換える方法」で何が違うかを説明します。
この小さな評価だけで、全用語・全質問への正しさを保証することはできません。
""")
    code("""
result = {
    "config": {k: v for k, v in CONFIG.items() if k not in {"work_dir", "drive_dir"}},
    "versions": {m.__name__: m.__version__ for m in [transformers, peft, trl, torch]},
    "method": "LoRA（量子化なし）", "dataset_topic": "講座の用語集",
    "answer_style": "自然文", "model_receives_reference_at_inference": False,
    "dataset": {"terms": len(glossary["terms"]), "train_examples": len(train_records),
                "evaluation_questions": len(questions), "input_sha256": EXPECTED_SHA256},
    "evaluation_notes": {
        "keyword_rate": "語句による一次確認。意味の内容正解率ではない。",
        "manual_correct_rate": "全問を手動確認するまで null。",
        "seen": "学習と同じ質問。未知の質問への汎化とは区別する。",
        "not_in_glossary": "未掲載でも正しく説明できれば正解。検索の未検出とは別に評価する。",
        "search_query": "問題文に書かれた用語。正解 ID や参照回答は検索に渡さない。",
    },
    "training": {"seconds": TRAINING_SECONDS, "global_step": trainer.state.global_step,
                 "epochs": trainer.state.epoch, "last_logged_training_loss": LAST_LOGGED_TRAINING_LOSS,
                 "loss_history": LOSS_HISTORY, "logging_steps": TRAIN_LOGGING_STEPS},
    "before": before_summary, "after": after_summary, "search": search_summary,
    "keyword_regressed": keyword_regressed, "executed_training_myself": True,
}
experiment.write_json(WORK / "result.json", result)
experiment.write_json(WORK / "eval_before.json", {"summary": before_summary, "rows": before_rows})
experiment.write_json(WORK / "eval_after.json", {"summary": after_summary, "rows": after_rows})
experiment.write_json(WORK / "evaluation.json", questions)
experiment.write_json(WORK / "manual_review.json", MANUAL_VERDICTS)
experiment.write_json(WORK / "comparison.json", comparison_rows)
if USE_DRIVE:
    import shutil
    target = Path(CONFIG["drive_dir"])
    for path in WORK.glob("*.json"):
        shutil.copy2(path, target / path.name)
    shutil.copytree(INPUT_DIR, target / "inputs", dirs_exist_ok=True)
    shutil.copytree(ADAPTER_DIR, target / "adapter", dirs_exist_ok=True)
print(json.dumps(result, ensure_ascii=False, indent=2))
print("保存先:", WORK)
""")
    notebook = {
        "cells": cells, "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                     "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
    }
    if with_reference:
        notebook = reference_variant(notebook)
    output_notebook.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"作成: {output_notebook.relative_to(ROOT)}")
    print(f"入力: {len(glossary['terms'])} 語 / {len(generated.splitlines())} 学習例 / {len(evaluation)} 評価問題")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-reference", action="store_true", help="20エポック・回答時に用語集全文を渡す版を作る")
    args = parser.parse_args()
    main(with_reference=args.with_reference)
