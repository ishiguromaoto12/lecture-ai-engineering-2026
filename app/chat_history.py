"""第1回: 会話履歴をJSONファイルに保存し、次回の起動時に読み込む。"""

import json
import os
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

from google.genai import types

HISTORY_DIR = Path(__file__).resolve().parent.parent / "lectures/01/learning_log/sessions"


def new_history_path() -> Path:
    """新しい会話ごとに別のファイル名を作る。"""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return HISTORY_DIR / f"chat-{timestamp}-{uuid4().hex[:8]}.json"


def save_history(path: Path, model: str, history: list[types.Content]) -> None:
    """回答まで完了した履歴を保存する。書き込み中も前回のファイルを残す。"""
    data = {
        "version": 1,
        "model": model,
        "history": [content.model_dump(mode="json", exclude_none=True) for content in history],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
            temporary_path = Path(f.name)
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        # 完成したファイルに置き換え、途中までのJSONを残さない。
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_history(path: Path) -> tuple[str, list[types.Content]]:
    """保存されたモデル名と履歴を読み込む。不正な形式ならValueError。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("version") != 1:
        raise ValueError("対応していない履歴ファイルです。")
    model = data.get("model")
    messages = data.get("history")
    if not isinstance(model, str) or not model.strip() or not isinstance(messages, list):
        raise ValueError("モデル名または会話履歴がありません。")
    if len(messages) % 2 != 0:
        raise ValueError("質問と回答が対になっていません。")

    history = [types.Content.model_validate(message) for message in messages]
    for index, content in enumerate(history):
        expected_role = "user" if index % 2 == 0 else "model"
        if content.role != expected_role or not content.parts:
            raise ValueError("会話履歴の発言順または内容が不正です。")
    return model, history
