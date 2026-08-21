from __future__ import annotations

import ast
from pathlib import Path

OPTIM_ROOT = Path(__file__).parents[2] / "src" / "whetstone_envs" / "optim"
ALLOWED_PRIVATE_IMPORTS = frozenset(
    {
        "whetstone.optim.proposal.proposer._durable_proposal_executor",
    }
)


def _private_whetstone_names(tree: ast.AST) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if not node.module.startswith("whetstone"):
                continue
            parts = node.module.split(".")
            if any(part.startswith("_") for part in parts):
                names.append(node.module)
            names.extend(
                f"{node.module}.{alias.name}"
                for alias in node.names
                if alias.name.startswith("_")
            )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if not alias.name.startswith("whetstone"):
                    continue
                parts = alias.name.split(".")
                if any(part.startswith("_") for part in parts):
                    names.append(alias.name)
    return names


def test_optim_package_imports_no_private_whetstone_members() -> None:
    offenders: list[str] = []
    for path in sorted(OPTIM_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        offenders.extend(
            f"{path.relative_to(OPTIM_ROOT)}: {name}"
            for name in _private_whetstone_names(tree)
            if name not in ALLOWED_PRIVATE_IMPORTS
        )
    assert offenders == []
