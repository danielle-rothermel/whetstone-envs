from __future__ import annotations

import importlib
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any

import whetstone_envs.instances
import whetstone_envs.manifests
import whetstone_envs.pools
import whetstone_envs.probes
import whetstone_envs.scoring

DEFS_DIR = Path(__file__).parents[1] / ".defs"
RELATIONSHIP_FIELDS = ("is_a", "part_of")
PUBLIC_MODULES = (
    whetstone_envs.instances,
    whetstone_envs.manifests,
    whetstone_envs.pools,
    whetstone_envs.probes,
    whetstone_envs.scoring,
)


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


def test_exported_symbols_exactly_cover_the_public_subpackages() -> None:
    symbol_terms: dict[str, list[str]] = defaultdict(list)
    for term in _terms():
        for symbol in term.get("exported_symbols", []):
            symbol_terms[symbol].append(term["name"])

    assert all(len(term_names) == 1 for term_names in symbol_terms.values())
    for symbol in symbol_terms:
        module_name, _, attribute = symbol.rpartition(".")
        assert getattr(importlib.import_module(module_name), attribute)

    expected = {
        f"{module.__name__}.{name}"
        for module in PUBLIC_MODULES
        for name in module.__all__
    }
    assert set(symbol_terms) == expected


def test_index_loads_both_authoritative_toml_files() -> None:
    index = (DEFS_DIR / "index.html").read_text(encoding="utf-8")
    assert 'data-defs-file="terms.toml" data-defs-kind="terms"' in index
    assert (
        'data-defs-file="contracts.toml" data-defs-kind="contracts"' in index
    )
    assert '<script type="module" src="defs-render.js"></script>' in index
