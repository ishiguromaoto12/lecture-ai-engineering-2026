"""第1回: 素のLLMへの質問と、会話履歴を保ったチャット。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする
    uv run python -m app.ask --chat
    uv run python -m app.ask --chat --resume 保存したJSONファイル

通常はツールも会話履歴も持たない「素の」呼び出し。--chat では履歴を保持する。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
from pathlib import Path
from time import perf_counter

from google import genai
from google.genai import errors, types

from app.chat_history import load_history, new_history_path, save_history
from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def print_usage(response: types.GenerateContentResponse) -> None:
    usage = response.usage_metadata
    input_tokens = usage.prompt_token_count if usage is not None else None
    output_tokens = usage.candidates_token_count if usage is not None else None
    print(f"[入力トークン数] {input_tokens if input_tokens is not None else '不明'}")
    print(f"[出力トークン数] {output_tokens if output_tokens is not None else '不明'}\n")


def ask(question: str) -> str | None:
    with genai.Client(api_key=get_api_key()) as client:
        started = perf_counter()
        response = client.models.generate_content(model=MODEL, contents=question)
        elapsed = perf_counter() - started
    print(f"[応答時間（秒）] {elapsed:.3f}")
    print_usage(response)
    return response.text


def run_chat(resume_path: Path | None = None) -> None:
    model, history = MODEL, []
    if resume_path is not None:
        path = resume_path.expanduser().resolve()
        try:
            model, history = load_history(path)
        except (OSError, ValueError) as e:
            raise SystemExit(f"会話履歴を読み込めません: {e}") from e
    else:
        path = new_history_path()

    print(f"[model] {model}")
    print(f"[保存先] {path}")
    if resume_path is not None:
        print(f"[再開] {len(history) // 2}往復の会話を読み込みました。")
    print("会話モードです。/exit または Ctrl+C で終了します。回答ごとに自動保存します。\n")

    with genai.Client(api_key=get_api_key()) as client:
        chat = client.chats.create(model=model, history=history)
        while True:
            try:
                question = input("あなた> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n会話を終了します。")
                break
            if question == "/exit":
                print("会話を終了します。")
                break
            if not question:
                continue

            try:
                started = perf_counter()
                response = chat.send_message(question)
                elapsed = perf_counter() - started
            except errors.APIError as e:
                print(api_error_message(e.code, e.message))
                print("今回の質問は保存していません。もう一度入力できます。\n")
                continue
            except KeyboardInterrupt:
                print("\n応答の待機を終了しました。直前に自動保存した会話から再開できます。")
                break

            print(f"[応答時間（秒）] {elapsed:.3f}")
            print_usage(response)
            if not response.text:
                print("回答の文章を取得できませんでした。今回の質問は保存しません。\n")
                # 回答がなかった質問を、次回の会話に混ぜない。
                chat = client.chats.create(model=model, history=history)
                continue

            print(f"AI> {response.text}\n")
            history = list(chat.get_history(curated=True))
            try:
                save_history(path, model, history)
            except OSError as e:
                raise SystemExit(
                    f"自動保存に失敗しました: {e}\n"
                    "今回の回答は保存できていません。画面の回答を控えてください。"
                ) from e
            print(f"[自動保存] {path}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", help="1回だけ質問する内容")
    parser.add_argument("--chat", action="store_true", help="会話モードで起動する")
    parser.add_argument("--resume", type=Path, metavar="FILE", help="保存した会話を再開する（--chat と併用）")
    args = parser.parse_args()
    if args.resume is not None and not args.chat:
        parser.error("--resume は --chat と一緒に指定してください。")
    if args.chat and args.question is not None:
        parser.error("--chat では起動後に質問を入力してください。")
    if args.chat:
        run_chat(args.resume)
        return

    question = args.question if args.question is not None else DEFAULT_QUESTION
    print(f"[model] {MODEL}")
    print(f"[質問] {question}\n")
    try:
        print(ask(question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
