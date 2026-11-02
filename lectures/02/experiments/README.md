# 第2回で追加した作業

このフォルダーに、今回の課題で追加したノートブック・コード・データ・実験結果を集約した。提出する文章は一つ上の [answer.md](../answer.md)。配布された [元のノートブック](../../../notebooks/lora_format_tuning.ipynb)、`data/glossary.json`、`app/tools.py`、`scripts/make_sft_dataset.py` は教材側に残している。

## ノートブックと実験の対応

| 課題・実験 | 作業用ノートブック | 結果 |
| --- | --- | --- |
| 基礎：講座 FAQ、20 / 10 エポック | [lora_faq_tuning.ipynb](notebooks/lora_faq_tuning.ipynb) | [20 エポック](basic/epochs20/) / [10 エポック](basic/epochs10/) |
| 実装：図書館 FAQ、20 エポック | [lora_library_tuning.ipynb](notebooks/lora_library_tuning.ipynb) | [library/](library/) |
| 発展：用語集、5 エポック、資料なし | [lora_glossary_tuning.ipynb](notebooks/lora_glossary_tuning.ipynb) | [glossary/epochs5/](glossary/epochs5/) |
| 発展：用語集、20 エポック、資料あり / なし | [lora_glossary_with_reference.ipynb](notebooks/lora_glossary_with_reference.ipynb) | [glossary/epochs20_with_reference/](glossary/epochs20_with_reference/) |
| 最新の追加実験：保留例も学習、20 エポック | [lora_glossary_abstention.ipynb](notebooks/lora_glossary_abstention.ipynb) | [glossary/abstention20/](glossary/abstention20/README.md) |

最新の保留例追加実験は、接続中の Colab T4 で Codex が端末から実行した。対応ノートブックは再実行用で、実行出力は入れていない。それ以前の作業用ノートブックには本人が実行した出力がある。基礎のノートブックは最後の作業状態を残したもので、20 / 10 エポックの確定結果は各結果フォルダーを見る。

## ファイルの役割

| 場所 | 役割 | 編集するとき |
| --- | --- | --- |
| `notebooks/` | Colab で開く個人用ノートブック | 新しい実験の作業先にする |
| `helpers/` | 用語集の評価、保留例作成、集計の現行コード | 処理を変えるときに編集する |
| `scripts/` | ノートブック生成と Colab 用学習処理 | 必要に応じて編集する |
| `tests/` | 入力の分離、採点、ノートブックの持ち運びの検証 | 処理を変えたら確認する |
| `basic/`・`library/`・`glossary/` | 学習条件別の結果、回答、ログ | 過去の結果を上書きせず、新しい条件は別のフォルダーへ保存する |
| `glossary/glossary_sft.jsonl` | 元の用語集から作った 116 件の学習データ | データ作成スクリプトから生成する |
| `glossary/evaluation.json` | 共通の評価 15 問と参照回答 | 問題を変えた場合は、過去の比較と区別する |

結果フォルダー内の `inputs/`、`run_experiment.py`、`executed_notebook.ipynb`、ZIP は実行時の記録。現行コードと同じ名前があっても、過去の入力・処理を確認するために残している。これらを編集して現在の処理を変更しない。保存当時のパス表記もそのまま残してある。

保留例追加実験の `*_unreviewed.json` は取得時の原本、接尾辞なしの評価 JSON は同じ回答に意味判定と理由を付けたもの。`baseline_*.json` は比較に使った前回の結果のコピーである。

## 再生成・検証

リポジトリのルートから実行する。

```bash
python lectures/02/experiments/scripts/build_glossary_notebook.py
python lectures/02/experiments/scripts/build_glossary_notebook.py --with-reference
python lectures/02/experiments/scripts/build_glossary_abstention.py
python -m pytest -q
```

出力のあるノートブックは再生成で上書きしない。必要なら先に別名で保存する。保留例追加実験は、保存済みの実験コード・入力・比較元も保護する。

## 整理前の場所との対応

| 整理前 | 整理後 |
| --- | --- |
| `lectures/02/result*.json`・`eval*.json`・`training*.json` | `basic/epochs20/` または `basic/epochs10/`。各フォルダー内では同じ名前に統一 |
| `lectures/02/library/` | `library/` |
| `lectures/02/advanced/` | `glossary/` |
| `notebooks/lora_format_tuning.ipynb` の個人の実行内容 | `notebooks/lora_faq_tuning.ipynb`。教材側は元の内容に復元 |
| 追加した `notebooks/lora_glossary*.ipynb`・`lora_library_tuning.ipynb` | `notebooks/` |
| 追加した `app/glossary_*.py` | `helpers/` |
| 追加したノートブック生成・実行スクリプト | `scripts/` |
| 追加した `tests/test_glossary_*.py` | `tests/` |
