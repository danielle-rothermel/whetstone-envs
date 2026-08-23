#!/usr/bin/env bash
#
# Run the envs Codex arm's real-CLI ladder against a live Codex session.
#
# This is a MANUAL check. It drives the real Codex CLI and spends real
# Codex agent turns on a logged-in subscription session, so it is never
# part of `scripts/pre-check.sh` and never runs automatically in CI.
#
# The task model stays fake throughout: every rung runs `--transport fake`,
# so the ladder spends Codex turns and no eval-provider credit. No
# OPENAI_API_KEY is set or required, and nothing here reads credential
# material -- the runner's own staging copies ~/.codex/auth.json into each
# run's scratch CODEX_HOME.
#
# Usage:
#   scripts/check-real-codex.sh                  # whole ladder, stop at first break
#   scripts/check-real-codex.sh -k rung3         # one rung
#   WHETSTONE_ENVS_REAL_CODEX_BINARY=/path/to/codex scripts/check-real-codex.sh
#
# To pin the model the §6 run will use (rung 6 reads these):
#   WHETSTONE_ENVS_REAL_CODEX_MODEL=gpt-5.4 \
#   WHETSTONE_ENVS_REAL_CODEX_EFFORT=medium scripts/check-real-codex.sh
#
# Prerequisites: macOS (the sandbox is sandbox-exec only), the Codex CLI on
# PATH or at /opt/homebrew/bin/codex, and a logged-in session
# (`codex login`).

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Run outputs live outside the repository, one directory per invocation.
timestamp="$(date +%Y%m%d-%H%M%S)"
output_root="${WHETSTONE_ENVS_REAL_CODEX_OUTPUT_DIR:-$HOME/drotherm/data/whetstone-envs/real-codex}"
output_dir="$output_root/$timestamp"
mkdir -p "$output_dir"

log="$output_dir/pytest.log"
report="$output_dir/rungs.txt"
table="$output_dir/rung-table.txt"
codex_binary="${WHETSTONE_ENVS_REAL_CODEX_BINARY:-/opt/homebrew/bin/codex}"
ladder_file="tests/real_codex/test_real_codex_ladder.py"

# How many rungs this ladder *is*, asked of pytest rather than pinned here,
# so adding a rung cannot leave the completeness check silently checking
# the old number. Collection is free: it spawns no Codex and spends
# nothing.
expected_rungs="$(
    WHETSTONE_ENVS_REAL_CODEX=1 \
        .venv/bin/python -m pytest "$ladder_file" \
        -m real_codex --collect-only -q -p no:cacheprovider 2>/dev/null \
        | grep -c "::test_rung" || true
)"
if [ -z "$expected_rungs" ] || [ "$expected_rungs" -eq 0 ]; then
    echo "could not determine the ladder's rung count; refusing to run" >&2
    exit 1
fi

echo "envs real-Codex ladder"
echo "  repo:    $repo_root"
echo "  output:  $output_dir"
echo "  codex:   $codex_binary"
echo "  model:   ${WHETSTONE_ENVS_REAL_CODEX_MODEL:-<arm default>}"
echo

# -x: the ladder is ordered by cost and by what each rung presupposes, so a
# broken lower rung makes every higher one uninterpretable.
#
# Both opt-ins are set here and nowhere else: the package's own
# (WHETSTONE_ENVS_REAL_CODEX) and the production spend gate
# (WHETSTONE_ENVS_ALLOW_REAL_CODEX) that run_optimizer refuses without.
set +e
WHETSTONE_ENVS_REAL_CODEX=1 \
WHETSTONE_ENVS_ALLOW_REAL_CODEX=1 \
    .venv/bin/python -m pytest \
    tests/real_codex/test_real_codex_ladder.py \
    -m real_codex \
    -x -v -s -p no:cacheprovider \
    "$@" 2>&1 | tee "$log"
status="${PIPESTATUS[0]}"
set -e

# One line per rung, in ladder order, from pytest's own verbose output.
{
    echo "envs real-Codex ladder — $timestamp"
    echo "codex version: $("$codex_binary" --version 2>/dev/null || echo unknown)"
    echo "commit:        $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "model:         ${WHETSTONE_ENVS_REAL_CODEX_MODEL:-<arm default>}"
    echo "effort:        ${WHETSTONE_ENVS_REAL_CODEX_EFFORT:-<arm default>}"
    echo
    printf '%-8s %-6s %s\n' RUNG RESULT TEST
    # A rung's verdict is not always on the same line as its name. The
    # ladder runs with -s, so anything a rung prints (rung 7's scale
    # line, rung 9's stage and leakage output) lands between the two --
    # and a regex that required them adjacent silently dropped exactly
    # those rungs from the table while still reporting "all rungs
    # passed". So this remembers the rung awaiting a verdict and pairs it
    # with the next verdict token, wherever that lands.
    #
    # The name is anchored at the start of the line and required to carry
    # the ladder file's own prefix. pytest emits a collected item as
    # "tests/real_codex/test_real_codex_ladder.py::test_rungN ...", and
    # nothing else legitimately starts that way -- while an unanchored
    # match accepted "::test_rung9" anywhere on a line, including inside
    # a temp path or a traceback frame that -s had printed.
    #
    # And a name is taken only while no rung is pending. Last-seen-wins
    # let a later mention overwrite the rung actually awaiting a verdict,
    # so the next PASSED/SKIPPED token was filed against the wrong rung.
    awk -v ladder="$ladder_file" '
        index($0, ladder "::test_rung") == 1 {
            if (pending_name == "") {
                match($0, /::test_rung[0-9a-c]+[a-z0-9_]*/)
                name = substr($0, RSTART + 2, RLENGTH - 2)
                rung = name
                sub(/^test_/, "", rung)
                sub(/_.*$/, "", rung)
                pending_name = name
                pending_rung = rung
            }
        }
        /(PASSED|FAILED|ERROR|SKIPPED)/ {
            if (pending_name != "") {
                match($0, /PASSED|FAILED|ERROR|SKIPPED/)
                printf "%-8s %-6s %s\n", pending_rung, \
                    substr($0, RSTART, RLENGTH), pending_name
                pending_name = ""
            }
        }
    ' "$log" > "$table" || true
    if [ ! -s "$table" ]; then
        echo "(no rung results parsed; see pytest.log)"
    fi
    cat "$table"
    echo
    # Rung 7 prints the full-size artifact sizes; carry them into the table
    # so a reader does not have to open the transcript for the one number
    # the §6 run has to budget disk against.
    grep -E '^rung7 scale:' "$log" || true
    echo
    # The verdict comes from the parsed table, never from pytest's exit
    # status alone. pytest exits 0 on a session where every rung SKIPPED,
    # and live-skips on rungs 2/6/7/8 make a partly-skipped ladder the
    # *expected* path -- so "all rungs passed" from $status was a claim
    # the run had not earned and could not support.
    observed="$(wc -l < "$table" | tr -d ' ')"
    passed="$(awk '$2 == "PASSED"' "$table" | wc -l | tr -d ' ')"
    not_passed="$(awk '$2 != "PASSED"' "$table" | wc -l | tr -d ' ')"
    echo "rungs: $passed passed, $not_passed not passed, of $expected_rungs"
    if [ "$status" -eq 0 ] \
        && [ "$not_passed" -eq 0 ] \
        && [ "$observed" -eq "$expected_rungs" ]; then
        echo "RESULT: all rungs passed"
    elif [ "$status" -ne 0 ]; then
        echo "RESULT: ladder stopped (exit $status) — see pytest.log"
    else
        # Exit 0 but the ladder was not fully observed: skipped rungs, or
        # rungs that never reached the table at all.
        awk '$2 != "PASSED" { printf "  %s: %s\n", $1, $2 }' "$table"
        if [ "$observed" -ne "$expected_rungs" ]; then
            echo "  $((expected_rungs - observed)) rung(s) produced no verdict"
        fi
        echo "RESULT: ladder not fully observed"
    fi
} | tee "$report"

echo
echo "transcript + rung table: $output_dir"
# A ladder that was not fully observed fails the check even when pytest
# exited 0, so a fully-skipped session can never be read as a green run.
if [ "$status" -ne 0 ]; then
    exit "$status"
fi
if grep -q '^RESULT: all rungs passed$' "$report"; then
    exit 0
fi
exit 1
