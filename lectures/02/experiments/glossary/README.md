# 第2回・発展編の実験

[発展編のノートブック](../notebooks/lora_glossary_tuning.ipynb)を、Colab の T4 カーネルで開く。カーネルを再起動して、セクション 0〜7 を順に実行する。

回答は普通の文章。学習前・学習後とも、回答時には質問だけを渡し、用語集や参照回答は渡さない。検索版は `app/tools.py` の文字列検索を使い、先頭の定義をそのまま返す。モデルの回答と並べて読む。

入力ファイルと Python の補助処理はノートブックに同梱しているため、VS Code の Colab 接続でも個別アップロードは不要。同梱時点の各ファイルの SHA-256 を `result.json` に記録する。

| ファイル | 内容 |
| --- | --- |
| `glossary_sft.jsonl` | `scripts/make_sft_dataset.py` が作った学習用 Q&A。58 語 × 質問 2 通り＝116 件 |
| `evaluation.json` | 評価問題、検索語、参照回答、判定用の語句、参照元 |
| `data/glossary.json`（リポジトリ直下からのパス） | 元の用語集 |

評価は、学習と同じ質問 6 問・言い換えた質問 6 問・未掲載語 3 問の計 15 問。一般的な用語（LoRA・QLoRA・過学習）と講座固有の用語（開始タグ・宿題の3レベル・answer.md）を含めた。学習と同じ質問での成績は、未知の質問に答える能力とは分けて見る。

未掲載語はドロップアウト・勾配降下法・知識蒸留。モデルが元から知っている可能性があるため、未掲載という理由だけで誤答にはしない。参照回答は [Dropout の原論文](https://www.jmlr.org/papers/v15/srivastava14a.html)、[Google の勾配降下法の説明](https://developers.google.com/machine-learning/crash-course/linear-regression/gradient-descent)、[知識蒸留の原論文](https://arxiv.org/abs/1503.02531)から概念を確認して用意した。

キーワードの一致は一次確認であり、内容の正しさは原文を読んで確認する。セクション 6 の `MANUAL_VERDICTS` に `True` / `False` を記入できる。全問を手動確認するまでは、内容正解率を `null`（未確定）として記録する。検索については、掲載語を先頭に取得できた割合と、未掲載語を未検出とした割合を別に測る。検索語は問題文に書かれた用語で、自然文から用語を抽出する機能は評価しない。

学習は、116 件で短い実験から始めるため 5 エポックを初期値にした。実際の所要時間と性能は実行して確認する。設定を変えて再学習する場合はカーネルを再起動する。

結果は `/content/glossary_lora_lab/` の `result.json`・`eval_before.json`・`eval_after.json`・`eval_search.json`・`evaluation.json`・`manual_review.json`・`comparison.json` に保存する。学習時間と loss は `result.json` の `training` に含まれる。実行後はノートブックと結果ファイルを保存し、このディレクトリに結果を追加する。

入力や評価の設計を変更してノートブックを作り直すときは、リポジトリ直下で `python lectures/02/experiments/scripts/build_glossary_notebook.py` を実行する。実行出力のあるノートブックは上書きしないので、先に別名で保存する。

## 5 エポックの実行結果

結果 JSON と回答原文は [epochs5/](epochs5/)、修正前のコードと出力を含む実行済みノートブックは [epochs5/executed_notebook.ipynb](epochs5/executed_notebook.ipynb) に保存した。元の結果をまとめた ZIP は `results_epochs5.zip`。

この実行では、入力長チェックがチャットテンプレートの返した辞書の項目数を数えてしまい、「最大 2 トークン」と誤表示した。実行後に同じ tokenizer・同じ revision・同じ 116 例を使い、チャット文字列を token ID に変換し直して確認したところ、最小 45・最大 118 トークン、上限 512 を超える例は 0 件だった。監査結果は [epochs5/token_length_audit_epochs5.json](epochs5/token_length_audit_epochs5.json) に保存した。学習には別途 `SFTTrainer` が元の `messages` を使っており、誤った表示値「2」は渡していない。

現在のノートブックと生成スクリプトの入力長チェックは修正済み。現在のノートブックには過去の出力を残しているため、セクション 3 の保存済み出力「2」は修正前の表示である。

## 追加実験：20 エポック・回答時に資料あり

[20 エポック・資料ありのノートブック](../notebooks/lora_glossary_with_reference.ipynb)を使用する。Colab の T4 カーネルを選び、カーネルを再起動してセクション 0〜7 を実行する。

学習データは前回と同じ 116 件の Q&A で、学習時の質問には資料を追加しない。エポック数を 20 にし、回答時には用語集 58 語の名前と定義の全文を渡す。資料ありの学習前後で同じ入力を使い、学習後には質問だけの条件でも同じ 15 問に答えさせる。

この追加評価で、同じ 20 エポック学習後のモデルについて資料の有無を比較できる。前回の 5 エポック・資料なしと今回の 20 エポック・資料ありの差には、学習回数と入力資料の両方の影響が含まれる。`COMPARE_WITHOUT_REFERENCE=False` にすれば、追加の資料なし評価は省略できる。

用語集全文は途中で切り捨てず、入力長を確認する。評価専用の参照回答や未掲載語の説明を資料に混ぜない。回答は普通の文章で、キーワード一致と内容の手動確認を分けて記録する。

出力先は `/content/glossary_lora_lab_with_reference/`。`eval_before.json`・`eval_after.json` は資料あり、`eval_after_without_reference.json` は学習後の資料なし。設定・資料の語数・入力長・loss・学習時間・各条件の要約は `result.json` に保存する。ローカルの結果保存先は `lectures/02/experiments/glossary/epochs20_with_reference/` として、5 エポックの結果を残す。

再生成する場合は `python lectures/02/experiments/scripts/build_glossary_notebook.py --with-reference` を実行する。既存の 5 エポック用ノートブックには書き込まない。

## 追加実験：保留する回答も学習する

元の定義例 116 件に、評価用とは異なる未掲載語 8 語の保留例を 16 件加え、合計 132 件で 20 エポック学習した。利用者が接続していた Colab T4 で Codex が実行し、結果を取得済み。資料なしでは未掲載語 3 問の保留が 0 問から 2 問に増え、誤断定が 3 問から 1 問に減った。ただし掲載語の内容正解は 12 問中 10 問から 9 問に下がった。資料ありでは未掲載語の保留は 1 問だった。

実行条件・判定基準・比較表・回答原文・ログは [abstention20/README.md](abstention20/README.md) にまとめた。内容の正解と保留を別に数え、前回の回答も同じ意味判定基準で確認した。これは Codex による原文の確認であり、キーワードの自動一致だけによる結果とは区別する。
