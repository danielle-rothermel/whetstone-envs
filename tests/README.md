# Test conventions

## Two-tier suite (fast / slow)

The suite is split into two tiers by the `slow` pytest marker (declared in
`pyproject.toml` under `[tool.pytest.ini_options].markers`).

- **Fast tier (default).** `uv run pytest` runs everything EXCEPT
  `@pytest.mark.slow`. `addopts = "-m 'not slow'"` deselects the slow tier
  automatically, so the everyday run stays under ~60s wall time.
- **Slow tier.** `uv run pytest -m slow` runs ONLY the slow tests. These are
  irreducibly slow (>5s): the byte-for-byte manifest regeneration diffs for
  the c18 default pool (~12s) and the c18 hard preset (~16s), which reseed
  the vendored PrOntoQA generator through a subprocess per depth. Relevant-
  distractor generation is heavy rejection sampling, so a full-N pool is
  slow; each regeneration diff is the determinism guarantee for its committed
  manifest.

Deselected-by-default is NOT never-run: the slow tier runs as its own CI job
(`slow-tests` in `.github/workflows/ci.yml`, `uv run pytest -m slow`) on every
push/PR.

### When to mark a test `slow`

Mark a test `slow` only when it is irreducibly >5s AND its cost cannot be
removed by restructuring. Before reaching for the marker:

- **Prefer tiny-N and the committed manifest.** Composition and split
  properties (stratum labels, per-stratum balance, seed freshness) are
  N-INDEPENDENT: assert them against the committed `manifest.json` /
  `manifest_hard.json`, or against a tiny-N pool. Keep exactly ONE test per
  pool that regenerates the FULL committed pool and diffs it against the
  frozen manifest -- that is the determinism guarantee -- and mark THAT one
  `slow`.

### Running

```sh
uv run pytest              # fast tier (default; slow deselected)
uv run pytest -m slow      # slow tier only
uv run pytest -m ''        # both tiers (override the default deselect)
```
