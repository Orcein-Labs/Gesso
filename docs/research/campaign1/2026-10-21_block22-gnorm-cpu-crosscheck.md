# Campaign 1: isolate block22 normalization error on archived operands

## Objective
Separate gated-normalization arithmetic error from upstream core differences using the already qualified CPU/Lava block22 replay.

## Permitted files
New block22_gnorm_cpu_* helper, optional independent verifier and bounded receipts under campaign scripts/results; this item and current status/frontier. No maintained/frozen source, consumed artifacts, oracle or model edits. CPU only; no Lava import, backend construction, GPU lock or model load.

## Interfaces
Actual Gesso.Inference._gdn_gnorm_merge!(::AbstractVector{Float32},nv,dv,core,z,scale,epsf) CPU fast-path entry and qualified replay2 raw core/z/merged/parameters. Convert recorded head-major GPU core to the actual CPU interleaved layout with an exact inverse. Use exact model-native epsilon from GGUF header metadata (header-only mmap permitted).

## Invariants
All916 tracked source files and candidate2108748 unchanged; exact captured Float32 operands, source-validated CPU entry and header/fixture identity. Preserve all failed results. This is an operand crosscheck, with no source/model admission, CPU tensor injection or throughput claim. Bound6GiB RAM/1GiBswap,300sec,1MiB new raw data.

## Tests
Bind replay2 qualified receipt/events and all raw descriptors. Execute actual CPU function on CPU and GPU core values, three repeats each, using respective recorded z and same captured scale/epsilon. CPU-core result must reproduce archived CPU merged exactly. Compare CPU recomputation from GPU core against archived GPU merged independently of upstream core differences; report raw bit mismatches/maxabs and save outputs for independent inspection. Input arrays must remain unchanged. Explicitly retain CPU/GPU layout inverse and finite/type/geometry checks. Report separate CPU-fast oracle versus general RMS formula semantics from source, without claiming general CPU and fast CPU bit equivalence.

## Performance target
N/A; CPU-only causal crosscheck. No performance claim.

## Expected artifact
Bounded real-operand evidence deciding whether whole gated normalization merits factor-level observation. This crosscheck does not select a specific component change or authorize a model intervention. Full Campaign1 unchanged.

## Result — 2026-10-22
Accepted after independent raw manager QA. The v4 runner invoked the maintained serial-F32 CPU fast `_gdn_gnorm_merge!` on qualified replay2 block22 CPU and Lava operands (`nv=32`, `dv=128`) with captured `z`, scale, and native epsilon. On archived CPU-core inputs it reproduces the archived CPU merge bit-for-bit in all three repeats. On the matching Lava core, `z`, and scale, the CPU-fast result differs from archived Lava merge in 2,441/4,096 scalars (`max_abs=2.9802322387695312e-8`) in each of three repeats; upstream CPU/Lava core tensors already differ in 3,809 scalars (`max_abs=1.862645149230957e-8`). All six outputs are finite, all inputs are preserved, and the capture contains 98,304 new raw bytes.

The maintained candidate is `2108748f29b44bbfcba7c992e0f326f0abe2b4ae` / 916 tracked files; the archived replay source is `ad47cb3241bb56f9289e3d6a10323e38b8566dbc` / 912 files. Their `cpu_decode.jl` digests match exactly (`61810b84cb665ff6f2a746f686adc0fb2f4545b528a4711e01c91f409ce48d2e`). Independent QA revalidated all 200 replay descriptors (37,352,704 bytes), event/qualification bindings, and all six outputs. Evidence: `block22_gnorm_cpu_v4/audit.json`, `block22_gnorm_cpu_manager_qa_v4.json`, and `block22_gnorm_cpu_manager_qa_v4.md`.

This matched-operand result isolates the complete gated-normalization stage from upstream core differences. The CPU crosscheck replaces the entire GN arithmetic, so it cannot attribute the delta among RMS accumulation/reduction, normalization quotient, scale, SiLU, or final multiplication. It does not authorize a quotient-only intervention. This is CPU-only diagnostic evidence: no backend or model load, source/model admission, or performance claim.

Retained unadmitted attempts: v1 failed parsing at the reserved `module=` named-tuple field and produced no raw events; v2 had a replay-event digest binding mismatch after string conversion plus failure-insensitive soft-scope exit handling, with its audit failures preserved; v3 failed lowering on a duplicate `replay_events_sha256` named-tuple field and produced no raw events. See `block22_gnorm_cpu_v1/`, `block22_gnorm_cpu_v2/`, `block22_gnorm_cpu_v3/`, and the manager QA v4 receipt. The next bounded step is the prepared `2026-10-22_block22-gnorm-factors.md` observational factor work item; no GPU job has launched.

## Receipt
ACCEPTED for the bounded CPU-only crosscheck after independent raw manager QA. Existing model controls remain 9/13; this receipt admits no source or model behavior. The separate general-route discriminator is terminal and documented in the current status.
