"""前後ページの取得・引用情報・重複除去をAPIなしで確認する。"""

import json

from app import retrieval
from scripts.build_rag_index import build_index


def test_neighbors_with_real_index_keep_hits_and_do_not_embed_extra_pages(tmp_path, monkeypatch):
    source = tmp_path / "lectures.json"
    source.write_text(json.dumps([{
        "id": "lec03-main", "title": "講義3", "content": [{
            "section": "メモリ", "content": [
                {"id": f"lec03-main-p{n:03d}", "page": n, "title": f"ページ{n}", "content": f"本文{n}"}
                for n in range(1, 5)
            ],
        }],
    }], ensure_ascii=False), encoding="utf-8")
    vectors = {1: [0, 1], 2: [1, 0], 3: [0.9, 0.1], 4: [-1, 0]}
    database = tmp_path / "index.sqlite3"
    build_index(source, database, embedder=lambda text, _: vectors[int(text[-1])])
    before = database.read_bytes()
    monkeypatch.setattr(retrieval, "RAG_DB", database)
    monkeypatch.setattr(retrieval, "RAG_SOURCE", source)
    monkeypatch.setattr(retrieval, "RAG_TOP_K", 2)
    embedding_calls = []

    def embed(query, task):
        embedding_calls.append((query, task))
        return [1, 0]

    monkeypatch.setattr(retrieval, "embed_text", embed)
    results = retrieval.search_with_neighbors("メモリの質問")

    assert [r["page"] for r in results] == [2, 3, 1, 4]
    assert len({r["page_id"] for r in results}) == 4
    assert all(r["document_id"] == "lec03-main" and r["section"] == "メモリ" for r in results)
    assert all(r["content"] == f"本文{r['page']}" for r in results)
    assert all("distance" in r and "neighbor_of" not in r for r in results[:2])
    assert results[2]["neighbor_of"] == "lec03-main-p002"
    assert results[3]["neighbor_of"] == "lec03-main-p003"
    assert all("distance" not in r for r in results[2:])
    assert embedding_calls == [("メモリの質問", "RETRIEVAL_QUERY")]
    assert database.read_bytes() == before


def test_neighbors_stay_in_same_document_and_stop_at_boundaries(monkeypatch):
    pages = [
        {"document_id": doc, "page": n, "page_id": f"{doc}-p{n:03d}", "content": f"{doc}:{n}"}
        for doc in ["lec01-main", "lec02-main"] for n in range(1, 4)
    ]
    hits = [pages[0], pages[-1]]
    monkeypatch.setattr(retrieval, "search", lambda _: hits)
    monkeypatch.setattr(retrieval, "load_pages", lambda *_: pages)

    results = retrieval.search_with_neighbors("質問")

    assert [r["page_id"] for r in results] == [
        "lec01-main-p001", "lec02-main-p003", "lec01-main-p002", "lec02-main-p002",
    ]
    assert results[2]["neighbor_of"] == "lec01-main-p001"
    assert results[3]["neighbor_of"] == "lec02-main-p003"
    assert all("neighbor_of" not in hit for hit in hits)


def test_neighbors_do_not_jump_across_missing_page_numbers(monkeypatch):
    pages = [{"document_id": "lec01-main", "page": n, "page_id": f"lec01-main-p{n:03d}"} for n in [1, 3]]
    monkeypatch.setattr(retrieval, "search", lambda _: [pages[0]])
    monkeypatch.setattr(retrieval, "load_pages", lambda *_: pages)
    assert retrieval.search_with_neighbors("質問") == [pages[0]]


def test_empty_search_result_needs_no_materials(monkeypatch):
    monkeypatch.setattr(retrieval, "search", lambda _: [])

    def unexpected_read(*_):
        raise AssertionError("検索結果が空なら資料を読み直す必要はない")

    monkeypatch.setattr(retrieval, "load_pages", unexpected_read)
    assert retrieval.search_with_neighbors("質問") == []
