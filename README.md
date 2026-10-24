# AIE2026 演習・宿題リポジトリ

AIエンジニアリング実践講座（AIE2026）の演習と宿題で使うリポジトリです。
講座を通して、**AIエンジニアリングの学習を支援するエージェント**を少しずつ育てていきます。

## 想定する環境

演習は **ブラウザだけで** 進められるように作っています。自分の PC への導入は不要です。

| 使うもの | 用途 | 注意 |
|---|---|---|
| GitHub の **個人** アカウント | このリポジトリを fork し、作業ブランチと PR で提出する | 会社や学校が管理するアカウントでは Codespaces・Copilot が制限されることがあります |
| GitHub Codespaces | 第1・3・5回などの開発環境。`.devcontainer/` の設定で Python 3.12・uv・Node.js が入ったコンテナが自動で用意されます | 個人アカウントの無料枠は月 120 コア時間（2 コアで約 60 時間）。**使い終わったら停止**してください |
| GitHub Copilot（Free でよい） | コードの補完とチャット。第5回の発展課題では MCP クライアントとしても使います | Free は補完・チャットの回数に月の上限があります |
| Gemini API（Google AI Studio の無料枠） | LLM の呼び出し。**個人**の Google アカウントでキーを発行し、Codespaces secret `GEMINI_API_KEY` で渡します | 無料枠には 1 分あたり・1 日あたりの上限があります（下の「前提と制約」） |
| Google Colab（T4 GPU、無料枠） | 第2回の LoRA 学習だけに使います | GPU の利用時間に上限があります。終わったらランタイムを削除してください |

自分の PC で動かす場合は、Python 3.12 と [uv](https://docs.astral.sh/uv/) を入れて `uv sync --frozen` を実行し、環境変数 `GEMINI_API_KEY` を設定してください。
プロジェクト直下の `.env` に `GEMINI_API_KEY=発行したキー` を保存しても使えます。`.env` は実行時に自動で読み込まれるため、`uv run` に `--env-file .env` を付ける必要はありません。すでに設定されている環境変数が優先されます。

## はじめに（第1回）

1. このリポジトリを自分のアカウントに **fork** する
2. [Google AI Studio](https://aistudio.google.com/apikey) で Gemini API キーを作る
3. GitHub の **Settings → Codespaces → Secrets** に `GEMINI_API_KEY` を登録し、使う repository に自分の fork を選ぶ
4. 自分の fork で **Code → Codespaces → Create codespace on main**
5. ターミナルで動作確認

```bash
uv run pytest -q
uv run python -m app.ask "LoRAとは何ですか？"
```

詳しい手順は [lectures/01/problem.md](lectures/01/problem.md) を見てください。

### 会話モード（第1回 実装編）

`--chat` を付けると、過去の質問と回答を踏まえて会話を続けられます。
`--chat` を付けない場合は、1回だけ質問し、回答とトークン数を表示して終了します。この場合、会話履歴は保存しません。

```bash
uv run python -m app.ask --chat
```

起動後の `あなた>` に質問を入力します。`/exit`、Ctrl+C、入力の終了（Ctrl+D）で終了します。空行は送信しません。各回答の入力・出力トークン数も表示します。

回答ごとに、プロジェクトの `lectures/01/learning_log/sessions/` 内のJSONファイルに自動保存します。保存先は画面に表示されます。新しく起動するたびに別の会話ファイルを作り、質問する前に終了した場合はファイルを作りません。

再開するときは、表示された保存先を `--resume` に指定してください。

```bash
uv run python -m app.ask --chat --resume lectures/01/learning_log/sessions/chat-日時-識別子.json
```

再開時は、保存したモデルと履歴を読み込み、同じファイルへの自動保存を続けます。モデルは履歴に記録された値を使い、現在の `AIE_MODEL` より優先します。再開後の画面には過去の全文を再表示せず、読み込んだ往復数を表示します。

保存するのはモデル名と会話履歴で、APIキーは含みません。普段の会話ファイルはGitの対象外です。提出用の動作確認は [第1回の解答](lectures/01/answer.md) にまとめています。

### 応答時間の比較（第1回 発展編）

1回だけ質問する場合も、会話モードでも、回答ごとに `[応答時間（秒）]` を表示します。SDKのAPI呼び出し直前から回答を受け取るまでを測ります。通信時間を含み、再試行が有効な設定ならその待ち時間も含みます。Pythonの起動・画面表示・履歴保存の時間は含みません。ストリーミングで最初の文字が届くまでの時間を測るものではありません。

発展編の実験は、次のコマンドで2モデル×2質問×各3回を順番に実行できます。実際のGemini APIを呼び出します。

```bash
uv run python scripts/benchmark_ask.py
```

結果は `lectures/01/learning_log/advanced/` 内の `execution.log` と `results.json` に保存します。既に結果がある場合は上書きせず終了するので、再実行する場合は前回の結果を別の場所へ移してください。実験条件と考察は [第1回の解答](lectures/01/answer.md) にまとめています。

SDKのHTTP送信・応答を2回の質問で確認する追加実験は、`uv run python -m scripts.check_sdk_retries` で実行できます。結果は同じ `advanced/` 内の `retry-check.log` に保存します。再試行設定はアプリと同じ未指定のまま観測し、APIキーやHTTPヘッダーは記録しません。こちらも既存のログを上書きしません。

ログは次のように整理しています。`basic/`・`implementation/`・`advanced/` 内の提出用ファイルはGitに追加でき、普段の会話を保存する `sessions/` はGitの対象外です。実行ログ中のパスは実行当時の記録として残しています。

```text
lectures/01/learning_log/
├── basic/
│   └── execution.log
├── implementation/
│   ├── execution.log
│   └── sample-session.json
├── advanced/
│   ├── execution.log
│   ├── results.json
│   └── retry-check.log
└── sessions/
    └── chat-日時-識別子.json
```

## 構成

```
.devcontainer/   Codespaces の環境定義（Python 3.12、uv、Node.js。起動時に uv sync --frozen）
.github/workflows/test.yml  PR ごとにテストを実行（fork では Actions タブで有効化が必要）
app/
  config.py      モデル名とAPIキーの読み込み
  ask.py         第1回：LLMに質問する
  chat_history.py 第1回：会話履歴の保存・読み込み
  tools.py       第3回：エージェントのツール（用語集の検索など）
  agent.py       第3回：Pydantic AI の学習支援エージェント（第5回：--mcp で MCP 経由）
  mcp_server.py  第5回：用語集を公開する MCP サーバー
  mcp_servers.json 第5回：使う MCP サーバーの設定（mcpServers 形式）
data/
  glossary.json  講座の用語集
  lectures/      各回の要点
lectures/NN/
  problem.md     その回の演習と宿題
  answer.md      あなたが記入して提出するファイル
notebooks/       第2回：Google Colab で動かす LoRA のノートブック（lora_format_tuning.ipynb）
scripts/         補助スクリプト
tests/           テスト（APIキー不要）
pyproject.toml / uv.lock  依存パッケージとその固定版
```

## 提出のしかた

1. 回ごとに作業ブランチを作る（例：`git switch -c lec01`）
2. 演習・宿題を行い、`lectures/NN/answer.md` に記入する
3. push して、**自分の fork の中で** 作業ブランチ → `main` の PR を作る

宿題は「基礎・実装・発展」の3レベルがあり、どれか1つに取り組めば完了です。

## 前提と制約

- **依存パッケージの版は固定**しています（`uv.lock`。Python 3.12、pydantic-ai-slim 2.57.0、google-genai 2.31.0）。講座の期間中は上げません。`uv sync --frozen` でそのまま入れてください。
- **既定のモデルは `gemini-3.5-flash-lite`** です。環境変数 `AIE_MODEL` で切り替えられます（下の「モデルの切り替え」）。ほかのモデルや自前契約の API を使ってもかまいません
- **Gemini API の無料枠にはレート制限**があります。エージェント（第3回以降）は 1 つの質問で LLM を複数回呼ぶので、第1回より早く上限に達します。`レート制限に達しました` と出たら 1 分ほど待ってください。`モデルが混み合っています（503）` が続くときは `AIE_MODEL=gemini-3.1-flash-lite` を試してください
- **API キーは環境変数 `GEMINI_API_KEY` だけ**で渡します。コード・コミット・チャットに貼らないでください。漏れたら AI Studio で削除して作り直します
- Codespaces からインターネットへは出られますが、外から Codespaces の中へは接続できません（ポート転送は既定で非公開）。第5回の MCP サーバーは Codespaces の中で動かします
- テスト（`uv run pytest -q`）は API キーなしで動きます（LLM を呼ばない TestModel を使用）

## 守ってほしいこと

- **APIキーをコードやコミットに書かない。** Codespaces secret で渡します
- **使い終わったら Codespaces を停止する。** 無料枠を使い切ると起動できなくなります
- AIツール（Copilot など）は使ってかまいません。提案されたコードは、読んで理解してから採用してください

## モデルの切り替え

既定モデルは `gemini-3.5-flash-lite` です。環境変数 `AIE_MODEL` で切り替えられます。

```bash
AIE_MODEL=gemini-3.5-flash uv run python -m app.ask "LoRAとは何ですか？"
```

無料枠には1分あたり・1日あたりのリクエスト上限があります。上限は [Gemini API のレート制限のページ](https://ai.google.dev/gemini-api/docs/rate-limits) で確認してください。
