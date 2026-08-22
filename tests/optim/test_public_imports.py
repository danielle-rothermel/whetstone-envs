from __future__ import annotations

import ast
from pathlib import Path

OPTIM_ROOT = Path(__file__).parents[2] / "src" / "whetstone_envs" / "optim"
ALLOWED_PRIVATE_IMPORTS: frozenset[str] = frozenset()
ALLOWED_PRIVATE_ATTRIBUTES: frozenset[str] = frozenset()

#: The package whose privates are off limits: whetstone-ai, the upstream
#: dependency. This repo's own ``whetstone_envs`` privates are ours to use --
#: ``optim/audit/_evidence.py`` and ``_mutate.py`` are deliberately private
#: module names within our package. Matching on the bare ``whetstone``
#: prefix would also catch ``whetstone_envs``, so the check is anchored to
#: the exact top-level package name.
UPSTREAM_PACKAGE = "whetstone"


def _is_upstream(module: str) -> bool:
    """True when ``module`` names whetstone-ai, not our own package."""
    return module == UPSTREAM_PACKAGE or module.startswith(
        f"{UPSTREAM_PACKAGE}."
    )


def _annotation_name(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.rsplit(".", 1)[-1]
    return None


def _name_types(tree: ast.AST) -> dict[str, str]:
    types: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for arg in (*node.args.args, *node.args.kwonlyargs):
                name = _annotation_name(arg.annotation)
                if name is not None:
                    types[arg.arg] = name
        elif isinstance(node, ast.AnnAssign) and isinstance(
            node.target, ast.Name
        ):
            name = _annotation_name(node.annotation)
            if name is not None:
                types[node.target.id] = name
    return types


def _is_cast_any(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name):
        call_name = func.id
    elif isinstance(func, ast.Attribute):
        call_name = func.attr
    else:
        return False
    if call_name != "cast" or not node.args:
        return False
    first = node.args[0]
    if isinstance(first, ast.Constant) and first.value == "Any":
        return True
    return isinstance(first, ast.Name) and first.id == "Any"


def _cast_subject_name(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    if len(node.args) < 2:
        return None
    subject = node.args[1]
    if isinstance(subject, ast.Name):
        return subject.id
    return None


def _private_whetstone_names(tree: ast.AST) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if not _is_upstream(node.module):
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
                if not _is_upstream(alias.name):
                    continue
                parts = alias.name.split(".")
                if any(part.startswith("_") for part in parts):
                    names.append(alias.name)
    return names


def _getattr_private_access(
    node: ast.Call,
    types: dict[str, str],
    cast_names: dict[str, str],
) -> str | None:
    """Name the owner reached by ``getattr(obj, "_private")``, if any."""
    func = node.func
    if not isinstance(func, ast.Name) or func.id != "getattr":
        return None
    if len(node.args) < 2 or not isinstance(node.args[1], ast.Constant):
        return None
    name = node.args[1].value
    if not isinstance(name, str) or not name.startswith("_"):
        return None
    subject = node.args[0]
    owner: str | None = None
    if _is_cast_any(subject):
        cast_subject = _cast_subject_name(subject)
        owner = types.get(cast_subject) if cast_subject else None
    elif isinstance(subject, ast.Name):
        owner = cast_names.get(subject.id) or types.get(subject.id)
    return f"{owner or 'Unknown'}.{name}"


def _private_cast_attributes(tree: ast.AST) -> list[str]:
    types = _name_types(tree)
    cast_names: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if value is None or not _is_cast_any(value):
            continue
        subject = _cast_subject_name(value)
        owner = types.get(subject, "Unknown") if subject else "Unknown"
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                cast_names[target.id] = owner
        elif isinstance(node, ast.AnnAssign) and isinstance(
            node.target, ast.Name
        ):
            cast_names[node.target.id] = owner

    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            reached = _getattr_private_access(node, types, cast_names)
            if reached is not None:
                found.append(reached)
            continue
        if not isinstance(node, ast.Attribute) or not node.attr.startswith(
            "_"
        ):
            continue
        if _is_cast_any(node.value):
            subject = _cast_subject_name(node.value)
            owner = types.get(subject, "Unknown") if subject else "Unknown"
            found.append(f"{owner}.{node.attr}")
        elif isinstance(node.value, ast.Name) and node.value.id in cast_names:
            found.append(f"{cast_names[node.value.id]}.{node.attr}")
    return found


def test_optim_package_imports_no_private_whetstone_members() -> None:
    offenders: list[str] = []
    for path in sorted(OPTIM_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        offenders.extend(
            f"{path.relative_to(OPTIM_ROOT)}: {name}"
            for name in _private_whetstone_names(tree)
            if name not in ALLOWED_PRIVATE_IMPORTS
        )
        offenders.extend(
            f"{path.relative_to(OPTIM_ROOT)}: {name}"
            for name in _private_cast_attributes(tree)
            if name not in ALLOWED_PRIVATE_ATTRIBUTES
        )
    assert offenders == []


def test_cast_any_private_attribute_access_is_detected() -> None:
    tree = ast.parse(
        "from typing import Any, cast\n"
        "cast('Any', obj)._hidden\n"
        "bound = cast(Any, obj)\n"
        "bound._also\n"
    )
    names = _private_cast_attributes(tree)
    assert "Unknown._hidden" in names
    assert "Unknown._also" in names


def test_getattr_private_access_is_detected() -> None:
    tree = ast.parse(
        "from typing import Any, cast\n"
        "def use(authority: CanonicalGepaEvalAuthority) -> None:\n"
        "    getattr(authority, '_completed_result')\n"
        "    getattr(cast('Any', authority), '_store')\n"
        "    getattr(authority, 'public')\n"
    )
    names = _private_cast_attributes(tree)
    assert "CanonicalGepaEvalAuthority._completed_result" in names
    assert "CanonicalGepaEvalAuthority._store" in names
    assert not any(name.endswith(".public") for name in names)


def test_our_own_private_modules_are_not_flagged() -> None:
    """``whetstone_envs`` privates are ours; only whetstone-ai's are barred.

    The guard exists to stop this repo reaching into its upstream
    dependency's internals. A bare ``whetstone`` prefix match would also
    catch our own ``whetstone_envs`` package, making a deliberately private
    module such as ``optim/audit/_evidence.py`` unimportable by its own
    package.
    """
    tree = ast.parse(
        "from whetstone_envs.optim.audit._evidence import RunEvidence\n"
        "import whetstone_envs.optim.audit._mutate\n"
    )
    assert _private_whetstone_names(tree) == []


def test_upstream_private_imports_are_still_flagged() -> None:
    tree = ast.parse(
        "from whetstone.optim.gepa._internal import thing\n"
        "from whetstone.optim.contracts import _private\n"
        "import whetstone._secret\n"
    )
    assert sorted(_private_whetstone_names(tree)) == [
        "whetstone._secret",
        "whetstone.optim.contracts._private",
        "whetstone.optim.gepa._internal",
    ]
