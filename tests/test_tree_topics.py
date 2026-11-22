"""用語付きTreeと比較モードを、APIを呼ばずに確認する。"""

import json

import pytest
from pydantic_ai import Tool
from pydantic_ai.models.test import TestModel

from app import agent as agent_module
from app import retrieval, tools


@pytest.fixture
def topic_materials(tmp_path, monkeypatch):
    def page(doc, number):
        return {"id": f"{doc}-p{number:03d}", "page": number,
                "title": f"ページ{number}", "content": f"先頭文{number}。本文の続き。"}

    documents = [
        {"id": "lec01-main", "title": "講義1", "content": [
            {"section": "章A", "content": [page("lec01-main", 1), page("lec01-main", 2)]},
            {"section": "章B", "content": [page("lec01-main", 3)]},
        ]},
        {"id": "lec02-main", "title": "講義2", "content": [{"section": "章C", "content": [page("lec02-main", 1)]}]},
        {"id": "lec05-main", "title": "対象外", "content": [{"section": "章D", "content": [page("lec05-main", 1)]}]},
    ]
    tree = {"id": "root", "kind": "root", "title": "資料", "children": []}
    for document in documents:
        node = {"id": document["id"], "kind": "document", "title": document["title"], "children": []}
        for section in document["content"]:
            node["children"].append({
                "id": f"{document['id']}-{section['section']}", "kind": "section", "title": section["section"],
                "children": [{"id": p["id"], "kind": "page", "title": p["title"]} for p in section["content"]],
            })
        tree["children"].append(node)

    mentions = [
        ("lec01-main-p001", "alpha"), ("lec01-main-p001", "alpha"), ("lec01-main-p001", "alpha"),
        ("lec01-main-p001", "beta"), ("lec01-main-p002", "beta"),
        ("lec01-main-p002", "common"), ("lec01-main-p003", "gamma"),
        ("lec02-main-p001", "other"), ("lec05-main-p001", "outside"),
    ]
    graph = {
        "nodes": [{"id": f"term:{name}", "kind": "term", "title": name.title()}
                  for name in ["alpha", "beta", "common", "gamma", "other", "outside"]],
        "edges": [{"source": page_id, "target": f"term:{term}", "relation": "mentions"} for page_id, term in mentions],
    }
    for attribute, name, value in [("RAG_SOURCE", "lectures", documents), ("RAG_TREE", "tree", tree), ("RAG_GRAPH", "graph", graph)]:
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        monkeypatch.setattr(retrieval, attribute, path)
    return tree


def test_topics_belong_only_to_the_candidate_subtree(topic_materials):
    result = retrieval.browse_tree(include_topics=True)
    assert [c["id"] for c in result["children"]] == ["lec01-main", "lec02-main"]
    first, second = result["children"]
    assert first["topics"] == ["Beta", "Alpha", "Common", "Gamma"]
    assert second["topics"] == ["Other"]
    assert "Outside" not in json.dumps(result)
    chapters = retrieval.browse_tree("lec01-main", include_topics=True)["children"]
    assert chapters[0]["topics"] == ["Beta", "Alpha", "Common"]
    assert chapters[1]["topics"] == ["Gamma"]


def test_topic_limit_counts_pages_not_duplicate_edges(topic_materials):
    page_ids = {p["page_id"] for p in retrieval.load_pages(retrieval.RAG_SOURCE)}
    hints = retrieval._tree_topic_hints(topic_materials, page_ids, limit=1)
    assert hints["lec01-main-章A"] == ["Beta"]


@pytest.mark.parametrize("node_id", [None, "lec01-main", "lec01-main-章A", "lec01-main-p001"])
def test_topic_variant_preserves_tree_and_page_contents(topic_materials, node_id):
    normal = retrieval.browse_tree(node_id)
    enriched = retrieval.browse_tree(node_id, include_topics=True)
    without_topics = {"node": enriched["node"], "children": [
        {key: value for key, value in child.items() if key != "topics"} for child in enriched["children"]
    ]}
    assert normal == without_topics
    assert "topics" not in enriched["node"]
    assert all("topics" not in child for child in enriched["children"] if child["kind"] == "page")


def test_original_tree_does_not_depend_on_graph(topic_materials, monkeypatch, tmp_path):
    monkeypatch.setattr(retrieval, "RAG_GRAPH", tmp_path / "missing.json")
    assert retrieval.browse_tree()["children"]
    assert "error" in json.loads(tools.browse_tree_with_topics())


def test_basic_rag_returns_hits_without_reading_neighbor_pages(monkeypatch):
    hit = {"page_id": "lec02-main-p052", "content": "根拠"}
    monkeypatch.setattr(retrieval, "search", lambda _: [hit])

    def unexpected_read(*_):
        raise AssertionError("基本RAGでは前後ページを追加しない")

    monkeypatch.setattr(retrieval, "load_pages", unexpected_read)
    assert json.loads(tools.search_materials_basic("LoRA"))["results"] == [hit]


@pytest.mark.parametrize("original,variant", [
    (Tool(tools.browse_tree), agent_module.TREE_TOPICS_TOOL),
    (Tool(tools.search_materials), agent_module.BASIC_RAG_TOOL),
])
def test_comparison_tools_keep_names_descriptions_and_argument_schemas(original, variant):
    definitions = []
    for tool in [original, variant]:
        model = TestModel(call_tools=[])
        with agent_module.agent.override(model=model, tools=[tool]):
            agent_module.agent.run_sync("ツール確認")
        definitions.append(model.last_model_request_parameters.function_tools[0])
    before, after = definitions
    assert before.name == after.name
    assert before.description == after.description
    assert before.parameters_json_schema == after.parameters_json_schema


@pytest.mark.parametrize("flag,name", [("--basic-rag", "search_materials"), ("--tree-topics", "browse_tree")])
def test_comparison_cli_lists_only_the_selected_tool(monkeypatch, capsys, flag, name):
    monkeypatch.setattr("sys.argv", ["app.agent", flag, "--list-tools"])
    agent_module.main()
    output = capsys.readouterr().out
    assert "モデルに見えているツール: 1 個" in output
    assert f"[{name}]" in output
    assert "[search_glossary]" not in output
    assert "[save_learning_log]" not in output
