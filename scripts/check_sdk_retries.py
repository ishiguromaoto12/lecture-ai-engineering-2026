"""第1回 発展編: Geminiへの2回の質問でHTTP送信・応答を観測する。

実行: uv run python -m scripts.check_sdk_retries
再試行設定はアプリと同じ未指定。APIキーやHTTPヘッダーはログに出さない。
"""

import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import httpx
from google import genai
from google.genai import errors, types

from app.config import get_api_key

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "lectures/01/learning_log/advanced/retry-check.log"
MODEL = "gemini-3.8-flash"
QUESTION = "LoRAとは何ですか？Python初心者向けに100字程度で説明してください。"


class HTTPTrace:
    """送信と応答の時刻・HTTPステータスだけを記録する。"""

    def __init__(self, emit):
        self.emit = emit
        self.started = time.perf_counter()
        self.requests = 0
        self.statuses = []

    def request(self, request: httpx.Request) -> None:
        self.requests += 1
        request.extensions["trace_number"] = self.requests
        self.emit(f"HTTP送信 {self.requests}: 経過{time.perf_counter() - self.started:.3f}秒")

    def response(self, response: httpx.Response) -> None:
        number = response.request.extensions["trace_number"]
        self.statuses.append(response.status_code)
        self.emit(f"HTTP応答 {number}: status={response.status_code}, ヘッダー受信までの経過{time.perf_counter() - self.started:.3f}秒")


def main() -> None:
    if LOG_PATH.exists():
        raise SystemExit("確認ログが既にあります。前回の記録を移動してから再実行してください。")
    key = get_api_key()
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("x", encoding="utf-8") as log:
        def emit(message):
            safe = str(message).replace(key, "[REDACTED]")
            print(safe, flush=True)
            log.write(safe + "\n")
            log.flush()

        emit("第1回 発展編 SDKのHTTP応答・再試行の追加確認")
        emit("実行: uv run python -m scripts.check_sdk_retries")
        emit(f"SDK: google-genai {version('google-genai')}")
        emit(f"モデル: {MODEL}\n質問: {QUESTION}")
        emit("2回のみ。履歴・temperature・retry_optionsは未指定。各回別のclient。")
        emit("HTTPの待機上限は120秒。開始間隔は最低15秒。失敗しても追加の質問は送らない。")
        emit("HTTPイベントは送信時と応答ヘッダー受信時。ヘッダー・URL・APIキーは記録しない。")
        previous_start = None
        for trial in (1, 2):
            if previous_start is not None:
                time.sleep(max(0, 15 - (time.perf_counter() - previous_start)))
            emit(f"\n=== 試行{trial} ===\n開始: {datetime.now().astimezone().isoformat()}")
            trace = HTTPTrace(emit)
            options = types.HttpOptions(
                timeout=120_000,
                client_args={"event_hooks": {"request": [trace.request], "response": [trace.response]}},
            )
            with genai.Client(api_key=key, http_options=options) as client:
                started = time.perf_counter()
                previous_start = started
                try:
                    response = client.models.generate_content(model=MODEL, contents=QUESTION)
                except (errors.APIError, httpx.HTTPError) as e:
                    emit(f"最終結果: {type(e).__name__}, HTTPコード={getattr(e, 'code', None)}")
                    emit(f"応答またはエラーまで: {time.perf_counter() - started:.3f}秒")
                else:
                    emit(f"最終結果: 成功\nSDK応答時間: {time.perf_counter() - started:.3f}秒")
                    usage = response.usage_metadata
                    emit(f"入力トークン数: {usage.prompt_token_count if usage else None}")
                    emit(f"出力トークン数: {usage.candidates_token_count if usage else None}")
                    emit(f"回答:\n{response.text}")
            emit(f"HTTP送信回数: {trace.requests}\nHTTP応答一覧: {trace.statuses}")
            emit(f"終了: {datetime.now().astimezone().isoformat()}")
        emit("\n確認終了。上の最終結果は各試行の成否を表し、スクリプトの完了とは区別する。")


if __name__ == "__main__":
    main()
