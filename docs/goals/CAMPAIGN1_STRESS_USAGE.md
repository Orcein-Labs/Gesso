# Ornith correctness and recovery stress

Run `scripts/stress_ornith_lava.jl` in an environment that loads this Gesso
checkout and its vendored Lava. It uses ordinary `reference_generate` and
`GenerateState` APIs. The CPU reference is an internal isolation oracle;
it is not independent model-family validation or a PyStack performance gate.

Required environment:

- `GESSO_REPO`: this checkout; the script verifies the loaded Gesso path.
- `GESSO_ORNITH35_GGUF`: the exact Ornith GSQ-RCO checkpoint.
- `GESSO_GGML_DIR`: the directory containing the ggml shared libraries.
- `GESSO_L1_JSONL`: a new result file; existing files are never overwritten.
- `GESSO_L1_ROLE=oracle`: CPU mode, with a new `GESSO_L1_ORACLE_OUT` file.
- `GESSO_L1_ROLE=validate`: Lava mode, with the CPU `GESSO_L1_ORACLE` file.

Set `GESSO_MOE_HOST_EXPERTS=1`, the declared GPU budget, and bounded Julia,
OpenBLAS and OMP thread counts. Serialize GPU jobs with the operator's GPU
lock. Do not run competing model loads.

```sh
julia --startup-file=no --project=YOUR_ENV -t16 scripts/stress_ornith_lava.jl
```

The default cases are `p64,hello`, with two repetitions and both implicit and
explicit state modes. Set `GESSO_L1_CASES` to a comma-separated list drawn from
`p1,p31,p32,p33,p63,p64,p65,p127,p128,p129,p256,hello,varied`.
The prefix corpus is `docs/Gesso_Stack.md`; oracle prompt IDs must match exactly.
`varied` uses `GESSO_L1_SEED` (default 20261014) and
`GESSO_L1_VARIANT_LEN` (default 64). Reverse the case list in a separate fresh
process to test ordering. `GESSO_PREFILL_CHUNK` selects the implementation's
prefill chunk limit; use the same setting when collecting and validating an
oracle unless explicitly testing chunk equivalence.

Other controls:

- `GESSO_L1_MAX_NEW_TOKENS` (default 1), `GESSO_L1_REPEATS` (default 2).
- `GESSO_L1_STATE_MODES=implicit|state|both` (default both).
- `GESSO_L1_GC_EVERY` (default 4): full collection after completed requests;
  zero disables this stress operation.
- `GESSO_L1_CANCEL=0|1` (default 1), `GESSO_L1_CANCEL_CASE` (default p64),
  `GESSO_L1_CANCEL_AFTER` and `GESSO_L1_RECOVERY_TOKENS` (both default 1).

Every run also injects a callback exception on its first selected case, checks
that state was invalidated and the callback token matches the CPU oracle,
retries that state, then closes it and tests a new state. Cancellation requires
the selected cancellation case and oracle lengths for the partial and resumed
continuations. Collect the oracle with the same cancellation controls so those
records exist.

Exit 0 requires all requests and recovery checks to pass. Unexpected exceptions,
wrong IDs, unstable repeated IDs, missing oracle records, and incompatible
prompt IDs fail the run. Logs report every request and the final summary;
inspect named skipped scenarios and require all requested scenarios to execute.
Each successful request also checks public `info.reused`: implicit states report
zero, and an explicit state holding an exact prompt prefix must reuse all of
that prefix. The log records reuse counts and cached-ID lengths. Explicit
cached IDs must equal the returned sequence excluding its final sampled token;
that token has not been consumed yet. These checks prevent a full re-prefill
from silently substituting for the requested reuse scenario.
For the harness's positive and negative ID gates, use
`GESSO_L1_ROLE=selfcheck` with a new JSONL path; this does not load a model.

Invoke the script directly as above. A custom wrapper that uses `include`
must explicitly call `main()` and propagate its return code; inclusion alone
does not execute the driver. Diagnostic controls that disable overlap or
multi-column prefill are separate runs, not default-environment acceptance.
