# 基礎編の実行結果

| フォルダー | 学習設定 | 内容 |
| --- | --- | --- |
| `epochs20/` | 元の 20 エポック | 設定・回答・評価・学習時間・loss |
| `epochs10/` | 10 エポックに変更 | 同じ項目を保存し、20 エポックと比較 |

各フォルダーの `result.json` が集計、`eval_before.json`・`eval_after.json` が回答と採点、`training.json` が学習記録。

[作業用ノートブック](../notebooks/lora_faq_tuning.ipynb) は最後の作業状態。`session_closed_notebook.ipynb` は、整理中にエディターが旧パスへ遅延保存したコピーで、最後のセルにカーネル終了の表示を含む。別の学習実験ではなく、履歴として残した。比較に使う確定値は上の各結果フォルダーと [answer.md](../../answer.md#基礎) を参照。
