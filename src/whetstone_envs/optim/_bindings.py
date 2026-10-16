from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from whetstone.optim.codex.adapter import CODEX_ADAPTER_KEY, CodexAdapter
from whetstone.optim.codex.control import CodexControl
from whetstone.optim.copro.adapter import COPRO_ADAPTER_KEY, CoproAdapter
from whetstone.optim.copro.control import CoproControl
from whetstone.optim.gepa.control import GepaControl
from whetstone.optim.gepa.harness_adapter import (
    GEPA_ADAPTER_KEY,
    GepaHarnessAdapter,
)
from whetstone.optim.miprov2.adapter import MIPROV2_ADAPTER_KEY, Miprov2Adapter
from whetstone.optim.miprov2.control import Miprov2Control


@dataclass(frozen=True, slots=True)
class CoproBinding:
    adapter: CoproAdapter
    control: CoproControl
    adapter_key: ClassVar[str] = COPRO_ADAPTER_KEY


@dataclass(frozen=True, slots=True)
class GepaBinding:
    adapter: GepaHarnessAdapter
    control: GepaControl
    adapter_key: ClassVar[str] = GEPA_ADAPTER_KEY


@dataclass(frozen=True, slots=True)
class Miprov2Binding:
    adapter: Miprov2Adapter
    control: Miprov2Control
    adapter_key: ClassVar[str] = MIPROV2_ADAPTER_KEY


@dataclass(frozen=True, slots=True)
class CodexBinding:
    adapter: CodexAdapter
    control: CodexControl
    adapter_key: ClassVar[str] = CODEX_ADAPTER_KEY


OptimizerBinding = CoproBinding | GepaBinding | Miprov2Binding | CodexBinding
