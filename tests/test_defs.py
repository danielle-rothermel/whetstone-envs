from __future__ import annotations

import importlib
import tomllib
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

DEFS_DIR = Path(__file__).parents[1] / ".defs"
RELATIONSHIP_FIELDS = ("is_a", "part_of")


class _DefsIndexParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.sources: set[tuple[str, str]] = set()
        self.module_scripts: set[str] = set()

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attributes = dict(attrs)
        source = attributes.get("data-defs-file")
        kind = attributes.get("data-defs-kind")
        if source is not None and kind is not None:
            self.sources.add((source, kind))
        if tag == "script" and attributes.get("type") == "module":
            source = attributes.get("src")
            if source is not None:
                self.module_scripts.add(source)


def _terms() -> list[dict[str, Any]]:
    with (DEFS_DIR / "terms.toml").open("rb") as file:
        return tomllib.load(file)["terms"]


def _find_cycle(graph: dict[str, list[str]]) -> list[str] | None:
    visited: set[str] = set()
    active: dict[str, int] = {}
    path: list[str] = []

    def visit(name: str) -> list[str] | None:
        visited.add(name)
        active[name] = len(path)
        path.append(name)
        for target in graph[name]:
            if target in active:
                return [*path[active[target] :], target]
            if target not in visited and (cycle := visit(target)) is not None:
                return cycle
        path.pop()
        del active[name]
        return None

    for name in graph:
        if name not in visited and (cycle := visit(name)) is not None:
            return cycle
    return None


def test_term_names_and_relationships_form_a_valid_graph() -> None:
    terms = _terms()
    names = [term["name"] for term in terms]
    assert len(names) == len(set(names))

    graph = {name: [] for name in names}
    for term in terms:
        source = term["name"]
        for field in RELATIONSHIP_FIELDS:
            for target in term.get(field, []):
                assert target in graph
                assert target != source
                graph[source].append(target)

    cycle = _find_cycle(graph)
    assert cycle is None, " -> ".join(cycle or [])


def test_exported_symbols_are_unique_and_resolvable() -> None:
    symbol_terms: dict[str, list[str]] = defaultdict(list)
    for term in _terms():
        for symbol in term.get("exported_symbols", []):
            symbol_terms[symbol].append(term["name"])

    assert all(len(term_names) == 1 for term_names in symbol_terms.values())
    for symbol in symbol_terms:
        module_name, _, attribute = symbol.rpartition(".")
        assert getattr(importlib.import_module(module_name), attribute)


def test_index_loads_both_authoritative_toml_files() -> None:
    parser = _DefsIndexParser()
    parser.feed((DEFS_DIR / "index.html").read_text(encoding="utf-8"))

    assert parser.sources == {
        ("contracts.toml", "contracts"),
        ("terms.toml", "terms"),
    }
    assert parser.module_scripts == {"defs-render.js"}
