"""Generate the committed Codex run fixtures. See ``README.md``.

Usage::

    PYTHONPATH=<whetstone-ai>/src python generate.py <whetstone-ai> <out>

``<whetstone-ai>`` must carry both the Codex-direct surface and
``OptimStepResult.proposer_usage``; neither the 0.1.6 tip nor the Codex
branch has both alone. The output is a run directory in exactly the shape
the envs audit reads: ``result.json`` beside ``runtime.sqlite``.

The agent is an in-process ``CodexRunner`` stand-in rather than the
subprocess fake CLI: the admission authority, the evaluating executor,
the effect leases, the ledger, and the adapter's own reconciliation are
all the production path either way, and the subprocess only changes how
the agent's scripted decisions arrive.
"""

from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path(sys.argv[1])
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import whetstone.eval.drivers  # noqa: F401  (breaks a circular import)
from dr_store.sync import open_sqlite
from tests.codex_support import (
    toy_capacity_binding,
    toy_codex_control,
    toy_codex_run,
    toy_codex_step_request,
    toy_tool_args,
)
from whetstone.core.identity import ImmutableJsonObject
from whetstone.core.leasing import (
    EffectLeaseAuthority,
    ReplayPolicy,
)
from whetstone.eval.reference_runtime import (
    ReferenceEvalRuntimeConfig,
)
from whetstone.optim.adapters import MappingAdapterRegistry
from whetstone.optim.codex.adapter import (
    CODEX_ADAPTER_KEY,
    CodexAdapter,
    CodexOutputArtifact,
    CodexRunResult,
    codex_lease_token_hash,
)
from whetstone.optim.contracts import (
    OptimStepResultRef,
    optimization_run_reference,
)
from whetstone.optim.harness import OptimHarness
from whetstone.optim.tools.contracts import (
    ToolCall,
)
from whetstone.optim.tools.evaluator import EngineToolEvaluator
from whetstone.optim.tools.execution import (
    EvaluatingToolExecutor,
)
from whetstone.optim.tools.facade import (
    ToolAdmissionAuthority,
    ToolCallStore,
)

TEMPLATE_A = "Answer {prompt} in one short sentence."
TEMPLATE_B = "Answer {prompt} with a single friendly word."
LEASE_TOKEN = "f" * 64
MAX_TOOL_CALLS = 8


class ScriptedRunner:
    """An agent that issues its calls through the real Runtime Tool Handle.

    It drives the same ``RuntimeToolHandle`` the MCP server drives, so
    every evaluation is admitted, leased, executed, and ledgered exactly
    as a real agent's would be.
    """

    def __init__(
        self,
        *,
        candidate,
        engine,
        steps,
        selected,
        agent_handle,
        extra_reported=(),
    ):
        self._extra_reported = tuple(extra_reported)
        self._candidate = candidate
        self._engine = engine
        self._steps = steps
        self._selected = selected
        # The agent drives the *evaluation server's* own handle, exactly
        # as a real out-of-process Codex CLI does over MCP. The handle the
        # adapter passes in is the Step's guarded one; the adapter
        # re-issues each reported call through that itself.
        self._agent_handle = agent_handle

    def run(self, request, handle, *, lease_token):
        evaluated = list(self._extra_reported)
        for call_id, template in self._steps:
            args = toy_tool_args(
                candidate=self._candidate,
                engine=self._engine,
                template=template,
            )
            call = ToolCall(
                call_id=call_id,
                tool_config=self._agent_handle.tool_config_ref,
                capacity_binding=self._agent_handle.binding,
                args=ImmutableJsonObject(args),
            )
            self._agent_handle(call)
            evaluated.append(call_id)
        return CodexRunResult(
            artifact=CodexOutputArtifact(
                run_id=request.run_id,
                evaluated_call_ids=tuple(evaluated),
                selected_call_id=self._selected,
                lease_token_hash=codex_lease_token_hash(lease_token),
            )
        )


def build(destination: Path, *, steps, selected, extra_reported=()) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    sqlite_path = str((destination / "runtime.sqlite").resolve())
    with open_sqlite(sqlite_path) as store:
        engine = ReferenceEvalRuntimeConfig().build_engine(store)
        control = toy_codex_control(
            engine=engine, max_tool_calls=MAX_TOOL_CALLS
        )
        run, config, candidate = toy_codex_run(control=control, engine=engine)
        effect_authority = EffectLeaseAuthority.sqlite(sqlite_path)
        tool_store = ToolCallStore(
            store,
            ToolAdmissionAuthority.sqlite(sqlite_path),
            effect_authority,
        )
        tool_executor = EvaluatingToolExecutor(
            EngineToolEvaluator(engine),
            engine.reward_policy,
            effect_authority,
            owner_id="codex-audit-fixture",
            replay_policy=ReplayPolicy.IDEMPOTENT,
            lease_duration=timedelta(minutes=5),
        )
        agent_handle = tool_executor.runtime_handle(
            config, tool_store, toy_capacity_binding(run)
        )
        adapter = CodexAdapter(
            ScriptedRunner(
                candidate=candidate,
                engine=engine,
                steps=steps,
                selected=selected,
                agent_handle=agent_handle,
                extra_reported=extra_reported,
            ),
            store=store,
            lease_token_factory=lambda: LEASE_TOKEN,
        )
        adapter.bind_tool_store(tool_store)
        harness = OptimHarness(
            store=store,
            adapter_registry=MappingAdapterRegistry(
                {CODEX_ADAPTER_KEY: adapter}
            ),
            tool_store=tool_store,
            effect_authority=effect_authority,
            owner_id="codex-audit-fixture",
            adapter_replay_policy=ReplayPolicy.NO_REDRIVE,
            lease_duration=timedelta(minutes=5),
            tool_executor=tool_executor,
        )
        harness.bind_run(run)
        request = toy_codex_step_request(
            control=control, run=run, candidate=candidate
        )
        result, ref = harness.run_step(request)
        print("   status", result.status, result.terminal_failure)
        print("   tool_evidence", len(result.tool_evidence))
        # Terminalize through the harness so the fixture is exactly the
        # OptimResult whetstone persists, not one assembled by hand.
        optim_result, _result_ref = harness.terminalize(
            run=optimization_run_reference(run),
            step_results=(OptimStepResultRef(record=result, record_ref=ref),),
        )
        (destination / "result.json").write_text(
            json.dumps(json.loads(optim_result.model_dump_json()), indent=2),
            encoding="utf-8",
        )
        effect_authority.close()
        tool_store.close()


if __name__ == "__main__":
    out = Path(sys.argv[2])
    print("building completed run...")
    build(
        out / "codex-completed",
        steps=[("c1", TEMPLATE_A), ("c2", TEMPLATE_B)],
        selected="c2",
    )
    print("building failed run (unevaluated selection)...")
    build(
        out / "codex-failed",
        steps=[("c1", TEMPLATE_A)],
        selected="never-issued",
        extra_reported=("never-issued",),
    )
    print("done")
