# GESSO REPRESENTATION PROGRAM
## Gauge-Compiled / Caged Physical Realization

    Internal names:     Caged Representation, Gauge-Compiled Quantization
    Document class:     research program / phase-7–10 specification seed
    Status:             accepted direction, pre-data
                    (no public/performance claim until a receipt exists)
    Index:              docs/research/README.md
    Sibling:            docs/research/KV_MEMORY_PROGRAM.md  (Magenta —
                        same philosophy, working state, not weights)
    Canon:              docs/Gesso_Stack.md remains canon; this document
                        extends §XIV, §XV, §XIII and feeds Phases 7 / 10.
                        Phase 5 (native engine) is NOT this program.

---

## 0. THE SEQUENCING LAW

Exotic representation research is why Gesso exists.
It is not what Gesso is allowed to become *before* it is a stack.

    FIRST:  a boringly reliable functional ML stack
            (Phases 0–6: meaning, oracle, import, backends, engine, measure)
    THEN:   one semantic win (Phase 7) and a representation planner (Phase 10)
    THEN:   Magenta topology / cages on working state
    NEVER:  contaminate Phase 5 with a quantizer, a cage kernel, or a
            "lossless 2-bit" story

This file is the parking lot with teeth. Tackle each problem
individually when its phase owns it. Do not build the atlas because
the notebook is exciting.

---

## 1. THE SENTENCE

    DO NOT QUANTIZE THE ACCIDENTAL REPRESENTATION.

Training left a coordinate system. That coordinate system is not
semantics. Search the exact functional equivalence class for a
minimum-description *executable* form, and run through that form.
Never reconstruct a dense FP matrix you do not need.

Lateral origin (AMD tetrahedral cages, GPUOpen / HPG 2026, Holger
Gruen): rest-pose geometry stays cheap and static; a cheap cage
deforms; rays are transformed into the coordinate system where the
expensive object already lives. The steal is not "tetrahedra for
matrices." The steal is:

    Separate the coordinate system of the query from the physical
    representation of the expensive object.

For a linear operator that means \(Wx = A(C(Bx))\) is allowed to be
the implementation, with no dense \(W\) in HBM.

Two expensive objects, same philosophy:

    WEIGHTS     static learned operators     this program
    KV / attn   dynamic working state        Magenta Memory

KV is the closer *animation* analogue (the cage is what changes).
Weights are a legitimate descendant of the *coordinate* analogue
(the query moves; the object stays cheap). Do not demote either.
Engineer KV first (Phase 5 paged manager). Probe weights on the
CPU oracle in parallel when a snapshot exists. Weaponize GPU
kernels only after Measurement 1 is interesting.

---

## 2. THREE LADDERS (FIREWALLED)

A result may not climb a stronger contract than it proved.

    ExactBits
        Reconstruct the original checkpoint bit patterns
        (BF16/FP16 as stored). Entropy codecs. Unweight / DFloat11
        class. Oracle: bitwise equality of weights.

    ExactOperator
        \(f_{\theta'} = f_{\theta}\) for every valid input under a
        *declared arithmetic contract* (F64 CPU, F32 CUDA, …).
        Gauge orbit. Machine-exact transforms. Oracle: Gesso
        reference_prefill / generate vs immutable FP reference.

    ReachabilityExact
        Equality only on the activation manifold actually visited.
        Null-space / Q2N-class. Different receipt. Never advertised
        as ExactOperator.

Below those: BoundedError (declared ε) and EmpiricalEquivalent
(calibration / perplexity). Calibration agreement is not
ExactOperator.

These names belong in receipts and, when Phase 10 lands, in the
realization type. They do not belong in `src/` until a work item
says so.

---

## 3. LADDER I — EXACT PHYSICAL COMPRESSION

Boring foundation. Original bits remain sacred.

    BF16 tensor → entropy / structural codec
                → compressed HBM representation
                → fused decode + GEMM  (reconstruction in SRAM/SMEM
                   is allowed; writing dense W back to HBM is the
                   thing Unweight already refused)

Baseline we compete with, not the moonshot:

    DFloat11    ~30% size, Huffman on BF16 exponents (~2.6 bits
                entropy in an 8-bit field), bit-identical outputs,
                decompress-and-discard. (Zhang et al., 2025)

    Unweight    Cloudflare 2026. Same exponent fact. Reconstructive
                matrix multiplication: reconstruct BF16 tiles in
                shared memory, feed tensor cores, never round-trip
                dense W through HBM. ~15–22% whole-model / ~30% MLP
                on Llama 3.1 8B, bit-exact. Pays ~30–40% tok/s on
                H100. Autotunes pipelines. MLP-scoped in the report.

Measurement 0: \(B_0\) = encoded bits of the original BF16 tensors
**including all metadata**, on a named checkpoint. This is the
entropy baseline. Sign/mantissa are near-incompressible in the
published accounting; that wall is why Ladder II is interesting.

Receipts for Ladder I record size *and* \(L_{\mathrm{execution}}\).
Size-only receipts are lies.

---

## 4. LADDER II — GAUGE-COMPILED FUNCTION COMPRESSION

The science. Objective is not "which rotation quantizes better."

    θ* = argmin_{θ' ~_f θ}  [ L_bits(θ') + λ L_execution(θ') ]

    θ' ~_f θ  means ExactOperator under the declared contract.

Wang & Wang 2026 characterize the maximal gauge of canonical MHA:

    G_max = ((GL(d_k))^h × (GL(d_v))^h) ⋊ S_h

RoPE shrinks the Q/K sector to the rotary commutant. They report
>1.1M redundant symmetry dimensions in a 110M Transformer, with
valid transforms preserving outputs to machine precision in their
experiments. Gauge-fixing eliminates *null directions*. That is
exact compression of redundancy. It is **not** "the whole FP16
tensor is 2-bit." Do not mix those claims.

QuaRot / SpinQuant / FPTQuant / GaugeQuant choose equivalent
coordinates so *quantization* hurts less. We ask which equivalent
representative has the **shortest executable description**.

After Measurement 1 (below) is interesting, a layer may lower to
something conceptually like:

    CagedLinear{E8Core, ButterflyGauge, SparseExactResidual}

Dispatch is the kernel. `ProjectionWeight` stays the logical
object. This is §XIV (logical ≠ physical) applied to operators.
Do not add these types in `src/` until Phase 10 (or a Phase 7
work item that names *one* win).

Low-bit cores (ternary, INT2, E8, additive codebooks) are
*primitives in a search space*. QuIP# is one point. It is
approximate PTQ. Our meter after a core is introduced:

    B_total = B_Q + B_R + B_A + B_B + B_metadata
    graph:  exact residual bits per original parameter

BitNet shows neural *functions* need not be 16-bit. It does not
show a pretrained FP function can be *transported* into that
compact form. Transport is the open problem. Training is still
out of scope (§LVIII).

---

## 5. LADDER III — REACHABILITY COMPRESSION

ΔW X = 0 on a calibration batch is not ExactOperator.

Q2N (2025) projects quantization error into an estimated
activation nullspace. Useful. Weaker. Encode it as
ReachabilityExact. Combining gauge orbit with a data-dependent
quotient is allowed as research; it does not get to climb.

Magenta's attention-damage oracle is the working-state cousin of
this ladder. Keep the objects distinct in receipts.

---

## 6. THE RED SEAM — MACHINE-EXACT GAUGES

Theoretical \(TT^{-1}=I\) is over ℝ. GPUs are floating-point
machines. `fl(T fl(T^{-1} x))` need not equal `x`.

First search family is **machine-exact under binary arithmetic
in range**, not free GL(n):

    signed permutations
    channel permutations
    sign flips
    powers-of-two / dyadic diagonal scalings
    unimodular / integer transforms where applicable
    compositions whose inverse is exact on the declared dtype

Hadamard (`1/√n`) is not in this family. Wang & Wang's full
GL(d_k) is not in this family until an arithmetic contract says
otherwise.

    g* = argmin_{g ∈ G_machine-exact}  bits(g · θ) + bits(g)

As of the writing of this file, a full min-description search
over machine-exact Transformer gauges has not been found in the
consulted literature. That is the seam. If Measurement 1 is
negative, the seam closes cleanly.

---

## 7. THE KILLER EXPERIMENT (BEFORE ANY Q2)

Model: HuggingFaceTB/SmolLM2-135M (the Phase 3 chosen net),
immutable reference via Gesso's CPU oracle. toy2 is too small
to see gauge dimension. llama_micro is a dispatch unit test,
not the science. Skip until `GESSO_SMOLLM2_DIR` exists.

    Measurement 0
        Losslessly compress original BF16 (Ladder I codec or
        equivalent entropy accounting).  →  B0
        Count every bit of metadata.

    Measurement 1
        Search only G_machine-exact. Do not quantize.
        Entropy-compress g(θ) plus the bits needed to specify g.
        →  B1 = B(g(θ)) + B(g)
        ExactOperator gate: reference_prefill / generate vs the
        untransformed model under the stamped arithmetic.

        Ask:  B1 < B0  ?

If no: kill the "gauge exposes latent compressibility" branch.
If modestly yes: interesting, proceed to cores + exact residual.
If dramatically yes: the information content of a trained net
depends on which member of its exact orbit we store. That is a
paper. It requires zero 2-bit hype.

Arithmetic hygiene: ExactBits is on **safetensors bit patterns**.
ExactOperator is logits under a declared contract. Gauging in
F64 then casting to BF16 is an ExactBits violation unless the
receipt says that was the contract.

Then, only then, add a low-bit core + exact residual stream and
plot residual bits/weight.

This experiment does **not** require Phase 5, a paged KV
manager, or a reconstructive GEMM. It requires the snapshot,
an enumerator for G_machine-exact, an entropy coder, and the
CPU oracle. It does **not** become a Buffy sprint while Phase 5
is open.

---

## 8. PHASE GATES

    Phase 5    native engine (prefill/decode/KV/sample/schedule)
               Magenta step 1: paged KV manager
               NOT this file's cages
    Phase 6    measure — so L_execution is a real number
    Phase 7    ONE semantic win; gauge-compiled compression is a
               *candidate*, not a default
    Phase 8    Lava (same logical model, other silicon)
    Phase 9    autotune (kernel search; speculative *plans* are
               not a client — Magenta §7.1)
    Phase 10   representation planner hosts Ladders I–II as
               realizations of ProjectionWeight
    Phase 11   memory planner / Magenta residency
    Phase 20   generalized speculation — different program

Public claims: Gesso_Stack §XLII / North Star §42. No paper
before data. Agent confidence is not evidence.

---

## 9. WHAT THIS FILE IS NOT

    a work item
    permission to add CagedLinear to src/ now
    permission to add a quantizer, E8 kernel, or Huffman codec
        to core Gesso
    a replacement for Magenta
    a claim that 2-bit lossless Llama exists
    training (BitNet-style native ternary is out of scope;
        loading such a checkpoint later is import, not training)

---

## 10. VERIFY-BEFORE-CITING

Citation in this repo = claim of having been read enough to
use. Abstracts, vendor blogs, and search-consulted PDFs are
marked. Promotion into canon requires a full read.

Consulted for this program (2026-09-30):

    AMD tetrahedral cages — GPUOpen / HPG 2026, Holger Gruen;
        80GB→1.7GB BVH, 300ms→3.3ms on RX 9070 XT demo
        (press + GPUOpen reports; paper not fully transcribed here)
    DFloat11 — Zhang et al., arXiv:2504.11651
    Unweight — Nikulin, Cloudflare Cf-TR-2026.04;
        blog.cloudflare.com/unweight-tensor-compression
        github.com/cloudflareresearch/unweight-kernels
    Wang & Wang 2026 — Complete Characterization of Gauge
        Symmetries in Transformer Architectures, PMLR 282
    FPTQuant — van Breugel et al., arXiv:2506.04985
    GaugeQuant — Bento & Seabra, arXiv:2607.20757
    Q2N — Zhao et al., arXiv:2506.11044
    QuaRot, SpinQuant, QuIP#, AQLM, BitNet — used as class
        markers; full re-read before any numerical comparison
        in a receipt

---

## 11. WHITEBOARD

                     logical operator (ProjectionWeight, …)
                               │
            ┌──────────────────┼──────────────────┐
            │                  │                  │
            ▼                  ▼                  ▼
      ExactBits          ExactOperator       ReachabilityExact
      entropy codec      machine-exact       activation quotient
      DFloat/Unweight    gauge + min desc    nullspace / Magenta
            │                  │                  │
            └──────────────────┼──────────────────┘
                               ▼
                     representation planner   (Phase 10)
                               │
                               ▼
                    specialized native kernel

    KV / Magenta (side): canonical pages, shared prefixes, CoW,
    heat, residency. Same sentence. Different object.

    DO NOT QUANTIZE THE ACCIDENTAL REPRESENTATION.
    DO NOT BUILD THE ATLAS BEFORE THE ENGINE.
