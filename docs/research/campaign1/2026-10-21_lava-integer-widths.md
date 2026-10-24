# Campaign 1: preserve LLVM logical integer widths in SPIR-V

## Objective
Repair the demonstrated loss of logical integer width during Lava SPIR-V lowering so the unchanged integer rounding program preserves all declared CPU/reference bits on the GPU.

## Permitted files
In a new isolated Lava worktree based on289d0693e84a3ad6ae6869cbf81b47dd4bc20ba7: src/compiler/spirv/emit.jl, src/compiler/spirv/types.jl, test/test_integer_width_semantics.jl, test/runtests.jl. External versioned reproduction/validation helpers and results under the existing campaign scripts/results roots; this work item and current planning/frontier records. Do not edit libs/, frozen Gesso ad47/vendor, or consumed helpers/results. Additional runtime/source files require a separate bounded item.

## Interfaces
LLVM integer types carry their own logical width. SPIR-V uses native8/16/32/64 buckets (i1 maps Bool). Current map_type!/spirv_int_width, emit_icmp!, emit_trunc! and neighboring integer casts must preserve logical low bits and signed interpretation rather than silently treating the bucket width as the source width. Follow Gesso charter first-principle/development-law and receipts boundaries; Lava remains a separately qualified supplier.

## Invariants
Native-width arithmetic/capabilities/ABI must stay intact. No source-level rounding-expression workaround, NaN removal, oracle retarget, tolerance relaxation, CPU tensor fallback or preferred token. Undefined LLVM inputs such as shifts by at least logical width and signed-min/-1 division are explicitly excluded from semantic positive cases, not silently assigned reference values. Audit adjacent integer arithmetic/casts/shifts/div/rem/intrinsics for the same widening hazard; either preserve a supported case correctly or explicitly decline unsupported cases rather than silently emit wrong arithmetic. Do not claim full general odd-width legalization without evidence for that scope.

## Tests
Retain both original GPU negatives and CPU/rational84,066case positive. Actual failed LLVM uses trunc i64 to i11 then signed i11 comparison; actual structurally valid SPIR-V directly UConverts to16 and compares it signed without logical sign extension. Value2047 is logical signed11=-1, bucket signed16=2047. Both fresh GPU processes fail exactly all37NaNs and match all84,029other inputs; actual shader has no float types. Receipt float32-rounding-gpu-negative-1-validation.json.

Add targeted scalar width tests covering trunc (including low-bit i1 behavior), zero/sign extension, signed/unsigned comparisons, and boundary arithmetic/shift/div/rem behavior for representative nonnative widths2,3,7,9,11,15,17,23,31,33,53,63 plus native width controls. Exercise actual optimized LLVM/SPIR-V and GPU entry points; inspect actual IR to establish that the intended width path was tested. Re-run the unchanged rounding primitive on all84,066cases and both UInt32 and bitcast-Float32storage paths, three repetitions, preserving NaN payload policy and853subnormal targets. Archive actual shaders/LLVM and independent output audits/mutations. Run existing float-controls and NoContraction regressions plus supplier GEMM/capability/registration regressions appropriate to the touched lowering; do not transfer old pass counts. Final integrated model/stress admission is separate and remains open.

## Performance target
N/A to throughput optimization. Record compile/runtime/allocation changes for the rounding workload and verify no new normalization instructions in unaffected native-width paths. Any cost gate for the eventual maintained Gesso candidate remains required.

## Expected artifact
Minimal reversible Lava supplier diff and tests, exact baseline-red/candidate-green receipts with source/env/helper identity, actual shader semantics, effective mutation rejection, and a clear ledger of any unimplemented logical-width cases. Only qualified supplier bytes may later enter a new Gesso candidate; ad47 remains frozen.

## Receipt
OPEN. Compiler defect is demonstrated, not repaired. CPU fixture validates against independent rational-nearest reference; both GPU negatives and actual wrong-width shader remain retained. No source/provider/model admission and no live rounding job at this checkpoint.

## Isolated repair progress
V1 scalar-mask/sign-extension/cast/arithmetic repair produced two exact integer rounding output paths across84,066inputs and3repeats; actual optimized i11 and actual16-bit mask+logical-sign-extension chain qualify. Repaired-digest NaN output and structurally valid logical-shift shader mutations are rejected. Receipts float32-rounding-supplier-1-validation-v2.json and float32-rounding-supplier-1-mutation-validation.json. Native Float64-to-Float32 control also matches this corpus; earlier provider-level subnormal loss is not proven to be caused by cast alone.

First16-width unit run is terminal56pass/3errors, zero numeric assertion failures: widths33/53/63 hit preexisting explicit rejection in map_constant_int!. V2 repairs constant padding/storage for33..63, explicitly declines unsupported scalar memory/intrinsic/atomic boundaries and odd-width vectors, and registers the new maintained test. Source278pinned by integer-widths-source-v2.json. Rounding fresh-process replay, maintained unit suite with actual shader capture, and separate strictFloatControls/NoContraction/direct-store-GEMM/subgroup regressions are active/queued. None is yet a qualified v2 result. No supplier integration or original model admission.

## V2 qualification checkpoint
Fresh v2 rounding capture qualifies all84,066cases on both positive paths across3repeats; v2 effective output and valid-shader sign mutations reject. float32-rounding-supplier-2-validation.json and float32-rounding-supplier-2-mutation-validation.json. All three shaders match v1 after ID compaction; native-cast and Float32-storage shader raw bytes are identical (not a pre-repair native baseline).

Maintained targeted suite now passes65/65 with BigInt references and3GPU repeats, covering6,116input pairs across16widths and15operations. Actual16optimizedLLVM/shader pairs archived; all intended widths survive optimization and all shaders validate. integer-widths-targeted-2/receipt.json and integer-widths-targeted-2-shaders.json. Timed include96.02seconds,5.3GiB cumulative allocations,1.93secondsGC; not an end-to-end or baseline-relative performance claim.

Regressionlauncher v1 failed before executing any suites because Main.include(Module,path) is not Base.include; this is a retained harness failure. Version2 creates a normal Julia module and uses Base.include. Its additional external boundary/same-bucket suite passes45, strictFloatControls116 and NoContraction141, zero failures/skips in those groups. Direct-storeGEMM, subgroup, and real registered supported/unsupported suites remain active; same maintained integer test against frozen pre-repair supplier is queued. Boundary and same-bucket cases still need moving into maintained tests; integer/FP casts, final baseline-relative costs, final source identity and any Gesso integration remain open.

## Qualified local commit
Local supplier commit388e6f95b8db99e16bb238ef6827a3768addbf11, clean278tracked files, exactly integer-widths-source-v3.json. Compiler bytes are identical to the v2 rounding/regression captures; only test/test_integer_width_semantics.jl changes between v2 and v3 by maintaining the boundary/cast cases and adding integer/float coverage. V3 passes238maintained assertions and58actual shader validations. Integer-to-float covers signed/unsigned logical widths and signedi1=-1 versus unsignedi1=1; valid float-to-integer cases preserve signed/unsigned results. Rounding2x3, effective output/shader mutations, strictFloatControls116, NoContraction141, directGEMM460, subgroup12, and actual supported registration460 plus deliberate unsupportedskip1 all qualify. Retained original-supplier same-test negative has27numeric failures,3explicit constant errors and1newguard failure; native8/16/32/64controls green. All jobs terminal.

Exact receipt index integer-widths-commit-1.json binds local commit,278files, qualified source/test snapshots and all six principal receipt digests; frozen Gesso912 unchanged. Full v3 include122.09seconds,6.53GiB cumulative allocations,2.26secondsGC (not peak RAM or a baseline-relative speed claim). Baseline-relative costs/portability and actual model/stress admission remain open. In particular, isolated native cast preserves all853subnormal targets, so fixture19 provider loss still needs actual intermediate localization; do not assume an integer-cast replacement alone fixes it. The broad Campaign1 objective remains active. Nothing pushed or integrated into frozen Gesso.
