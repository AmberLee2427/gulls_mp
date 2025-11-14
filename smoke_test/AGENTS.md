# Agent Coding Contract (Strict Mode)

This document defines how agents write Smoke Test code in this repository. The goal is to catch errors in simulations early, never mask missing physics, and make failures obvious and actionable. The goal is to try to get the fucking thing to fail. Any excuse to fail is a good one. Use petty, rediculous, neurotic, OCD levels of validation. If you can catch a legitimate fail, you save the day. The goal is NEVER a "robust" test that runs to completion. Fuck that. Make the bitch so brittle a gentle breaze has her screaming TracebackError. This Smoke Test is the only source of regression testing currently implemented--it's failures are sacred. 

## Core principles

- Fail fast
  - If a required value is missing, non-finite, or out of range, raise immediately.
  - Do not continue in a degraded or "best-effort" mode.

- No defaults for physical parameters
  - Never substitute defaults for physics/astrometry inputs (examples: `pi_n`, `pi_e` (parallax), `theta_e`, `rho`, `tE_ref`, `u0`, `alpha_event`, `t0`, `event_ra`, `event_dec`, proper motions).
  - If the value is expected in a specific source (e.g., the summary metrics), require it there; if absent, raise.

- Single source of truth
  - Do not search multiple locations or fallback across sources (e.g., do not read from event header if summary is missing). If it’s not where expected, something is broken—raise.

- Branches only from explicit settings
  - Only add branches that originate from:
    - The parameter file values (e.g., `params` fields in `*.prm`), or
    - Command-line flags and tags handled by `run_smoke_test.py` (e.g., `--ci`, `--validate-bagle`).
  - Do not introduce branches that invent behavior not controlled by those settings.

- Warnings allowed; FP errors fatal
  - Let warnings through (do not escalate to errors) so receipts (e.g., BAGLE FSBL→PSBL fallback) are visible.
  - Keep `numpy` floating-point errors fatal with `np.seterr(all='raise')`.

- No permissive exception handling
  - Do not catch exceptions to continue silently. Only catch to add context, then re-raise.

- Required parameters are required in signatures
  - Do not use `None` defaults for required values. Make them positional/required and validate at boundaries.

- Validate data rigorously
  - Check shapes, units, and frames at handoff points; perform finite checks (`np.isfinite`).
  - For rotation/transform estimation, assert structure (e.g., 2×2 rotation `[[c,s],[-s,c]]`) and raise on violations.
  - Validation helpers must not silently skip on missing/non-finite inputs; they must raise immediately (e.g., `validate_summary_vs_event`).

## Conversation and context requirements

- This document is the agreed contract for agent work in this repo.
  - Every conversation summary must explicitly reference AGENTS.md as the operative contract.
  - Any deviation from this contract must be negotiated and recorded in the conversation before changes proceed.

- No code edits without contract in context
  - The agent must ensure the contents of AGENTS.md are in its active context before making any code edits.
  - If AGENTS.md is not available in context, the agent must load it (or request it) before proceeding.
  - If the document cannot be loaded or validated, the agent must abort edits and report that the contract is missing.
  - Make this clear in conversation summaries.

## Prohibited patterns (with examples)

- Fallbacks across sources
  - Bad: `alpha = summary.get('alpha_event') or float(event_vals[1])`  # hides broken summary
  - Required: `alpha = summary['alpha_event']` and raise if missing/non-finite.

- Defaults for physical params
  - Bad: `pi_n = summary.get('pi_n', 0.0)` or `if pi_e is None: pi_e = 0.0`
  - Required: `pi_n = require_finite(summary.get('pi_n'), 'pi_n')`.

- Permissive optionals for required values
  - Bad: `def f(alpha: float | None = None): ...`
  - Required: `def f(alpha: float, /, ...):` and validate.

- Swallowing exceptions
  - Bad: `try: ... except Exception: pass`
  - Required: catch only to add context and re-raise `raise SmokeTestError("context") from e`.
  
  - Silent validation skips
    - Bad: `if value is None or not np.isfinite(value): return  # skip`
    - Required: `require_finite(value, 'label')` and raise; validation must fail-fast when inputs are not valid.

- Invented branching
  - Bad: adding modes that are not tied to params or CLI flags.

## Allowed patterns

- Require helpers
  - `val = require_finite(summary.get('theta_e'), 'theta_e')`
  - `muS_E, muS_N = galactic_pm_to_icrs(l, b, mul, mub)`

- Diagnostic warnings/receipts
  - Let `warnings.warn(...)` surface (do not silence).

- CLI/param-driven branches
  - Example: CI subset selection in runner, `--validate-bagle`, `ASTROMETRY_ON` gating plots.

## Repository application (current modules)

- `smoke_test/utils.py`
  - `compute_vbm_model(...)` requires all physical parameters from the summary; raises `SmokeTestError` on any missing/non-finite input. No defaults or cross-source fallbacks.
  - `galactic_pm_to_icrs(...)` handles PM conversion deterministically.

- `smoke_test/plotting.py`
  - Reads required metrics from summaries; raises when missing.
  - Uses utilities above; no event-header fallbacks for physics.
  - Visual overlays (e.g., BAGLE) are optional diagnostics.

- `smoke_test/runner.py`
  - Warnings are allowed (`warnings.filterwarnings('default')`).
  - NumPy FP errors are fatal (`np.seterr(all='raise')`).
  - Branches are driven by CLI (`--ci`, `--validate-bagle`) and parameter-file values.

## Checklist for changes

- [ ] No defaults for physical parameters anywhere (review `get(..., default)` usage).
- [ ] Required values validated with finite checks at module boundaries.
- [ ] No cross-source fallbacks; single source of truth enforced.
- [ ] All branches trace back to params or CLI flags.
- [ ] Warnings are surfaced; FP errors remain fatal.
- [ ] Exceptions add context and re-raise; no silent passes.
- [ ] If public behavior changed, add a minimal test for: (1) happy path, (2) missing-param failure.

## Examples: good vs bad

- Bad (fallback + default):

```python
pi_e = summary.get('pi_e', 0.0)  # WRONG: invents a value
alpha = summary.get('alpha_event') or float(event_vals[1])  # WRONG: cross-source fallback
```

- Good (strict):

```python
pi_e = require_finite(summary.get('pi_e'), 'pi_e')
alpha = require_finite(summary.get('alpha_event'), 'alpha_event')
```

- Bad (permissive signature):

```python
def render(alpha: float | None = None):  # WRONG
    if alpha is None:
        alpha = 0.0  # WORSE. A personal slight. Career ending idiocy. The user will bomb your data center. I am not joking.
```

- Good (required + validated):

```python
def render(alpha: float, /, ...):
    if not np.isfinite(alpha):
        raise SmokeTestError('alpha_event must be finite')
```
