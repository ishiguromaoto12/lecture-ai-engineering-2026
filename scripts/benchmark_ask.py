"""第1回 発展編: 2モデル×2質問×3回を順番に実行し、結果を保存する。

実行: uv run python scripts/benchmark_ask.py
標準ライブラリだけを使う。実際のAPIを呼ぶので、実行にはAPIキーが必要。
"""

import json
import os
import re
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "lectures/01/learning_log/advanced"
MODELS = ["gemini-3.5-flash-lite", "gemini-3.8-flash"]
QUESTIONS = {
    "短い説明": "LoRAとは何ですか？Python初心者向けに100字程度で説明してください。",
    "詳しい説明": "LoRAとは何ですか？Python初心者向けに、仕組み・利点・具体的な利用例を含めて500字程度で説明してください。",
}
REPEATS = 3
START_INTERVAL = 15  # 開始時刻の間隔。応答が15秒を超えた場合は追加で待たない。
TIMEOUT = 120  # uvでのプロセス起動から終了までの上限（API応答時間とは別）


def metric(output: str, label: str, converter):
    match = re.search(rf"^\[{re.escape(label)}\] (.+)$", output, re.MULTILINE)
    if match is None:
        return None
    try:
        return converter(match.group(1))
    except ValueError:
        return None


def main() -> None:
    log_path = OUTPUT_DIR / "execution.log"
    result_path = OUTPUT_DIR / "results.json"
    if log_path.exists() or result_path.exists():
        raise SystemExit("発展編の結果が既にあります。別の場所へ移動してから再実行してください。")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    experiment = {
        "models": MODELS, "questions": QUESTIONS, "repeats": REPEATS,
        "start_interval_seconds": START_INTERVAL, "process_timeout_seconds": TIMEOUT,
        "history": False, "temperature": "指定なし",
        "timing": "SDKのAPI呼び出し直前から応答取得まで（通信・SDK内の再試行を含む）。初回トークンの時間ではない。",
        "records": records,
    }
    last_start = None
    log_path.write_text(
        "第1回 発展編 実行ログ\n"
        "実行: uv run python scripts/benchmark_ask.py\n"
        f"開始間隔: 最低{START_INTERVAL}秒。並列実行・実験スクリプトによる再試行なし。\n"
        "各回で別プロセスを起動。履歴・temperatureの指定なし。\n"
        "応答時間はSDK呼び出しを計測。プロセス全体の時間とは区別する。\n",
        encoding="utf-8",
    )
    for trial in range(1, REPEATS + 1):
        for label, question in QUESTIONS.items():
            for model in MODELS:
                if last_start is not None:
                    time.sleep(max(0, START_INTERVAL - (time.perf_counter() - last_start)))
                command = ["uv", "run", "python", "-m", "app.ask", question]
                record = {
                    "model": model, "question_label": label, "question": question,
                    "trial": trial, "started_at": datetime.now().astimezone().isoformat(),
                    "command": f"AIE_MODEL={model} {shlex.join(command)}",
                }
                print(f"開始: {model} / {label} / 試行{trial}", flush=True)
                last_start = time.perf_counter()
                try:
                    result = subprocess.run(
                        command, cwd=ROOT, capture_output=True, text=True, timeout=TIMEOUT,
                        env={**os.environ, "AIE_MODEL": model, "PYTHONUNBUFFERED": "1"},
                    )
                    code, stdout, stderr = result.returncode, result.stdout, result.stderr
                except subprocess.TimeoutExpired as e:
                    code = 124
                    stdout = e.stdout or b""
                    stderr = e.stderr or b""
                    stdout = stdout.decode(errors="replace") if isinstance(stdout, bytes) else stdout
                    stderr = stderr.decode(errors="replace") if isinstance(stderr, bytes) else stderr
                    stderr += f"\n実験スクリプトが{TIMEOUT}秒の上限で停止。APIのHTTPステータスではない。\n"
                record.update(
                    ended_at=datetime.now().astimezone().isoformat(), exit_code=code,
                    process_seconds=round(time.perf_counter() - last_start, 3),
                    response_seconds=metric(stdout, "応答時間（秒）", float),
                    input_tokens=metric(stdout, "入力トークン数", int),
                    output_tokens=metric(stdout, "出力トークン数", int),
                    stdout=stdout, stderr=stderr,
                )
                records.append(record)
                result_path.write_text(json.dumps(experiment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(f"\n=== {model} / {label} / 試行{trial} ===\n")
                    for field in ("started_at", "ended_at", "command", "exit_code", "process_seconds", "response_seconds", "input_tokens", "output_tokens"):
                        f.write(f"{field}: {record[field]}\n")
                    f.write(f"標準出力:\n{stdout}\n標準エラー:\n{stderr}\n")
                print(f"終了: code={code}, API応答={record['response_seconds']}秒, 入力={record['input_tokens']}, 出力={record['output_tokens']}", flush=True)
    print(f"保存先: {log_path}\n集計用: {result_path}", flush=True)


if __name__ == "__main__":
    main()
