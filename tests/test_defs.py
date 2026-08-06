from __future__ import annotations

import importlib
import tomllib
from collections import defaultdict
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

DEFS_DIR = Path(__file__).parents[1] / ".defs"
PACKAGE_DIR = Path(__file__).parents[1] / "src" / "whetstone_envs"
RELATIONSHIP_FIELDS = ("is_a", "part_of")
REQUIRED_CONTRACT_FIELDS = {"date", "rationale", "statement", "title"}
OPTIONAL_CONTRACT_FIELDS = {"check"}


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


def _contracts() -> list[dict[str, Any]]:
    with (DEFS_DIR / "contracts.toml").open("rb") as file:
        return tomllib.load(file)["contracts"]


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


def test_exported_symbols_cover_owning_package_apis() -> None:
    expected_symbols: set[str] = set()
    package_names = sorted(
        path.name
        for path in PACKAGE_DIR.iterdir()
        if path.is_dir() and (path / "__init__.py").is_file()
    )
    for package_name in package_names:
        module_name = f"whetstone_envs.{package_name}"
        module_exports = importlib.import_module(module_name).__all__
        assert isinstance(module_exports, list)
        assert all(isinstance(name, str) for name in module_exports)
        expected_symbols.update(
            f"{module_name}.{name}" for name in module_exports
        )

    documented_symbols = {
        symbol
        for term in _terms()
        for symbol in term.get("exported_symbols", [])
    }
    assert documented_symbols == expected_symbols


def test_contracts_have_complete_unique_records() -> None:
    contracts = _contracts()
    titles = [contract["title"] for contract in contracts]
    assert len(titles) == len(set(titles))

    for contract in contracts:
        assert contract.keys() >= REQUIRED_CONTRACT_FIELDS
        assert contract.keys() <= (
            REQUIRED_CONTRACT_FIELDS | OPTIONAL_CONTRACT_FIELDS
        )
        for field in contract:
            value = contract[field]
            assert isinstance(value, str)
            assert value.strip()
        date.fromisoformat(contract["date"])


def test_index_loads_both_authoritative_toml_files() -> None:
    parser = _DefsIndexParser()
    parser.feed((DEFS_DIR / "index.html").read_text(encoding="utf-8"))

    assert parser.sources == {
        ("contracts.toml", "contracts"),
        ("terms.toml", "terms"),
    }
    assert parser.module_scripts == {"defs-render.js"}
