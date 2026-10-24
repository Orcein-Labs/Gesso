# Per-kernel NoContraction contract

`Lava.with_no_contraction(kernel)` opts one KernelAbstractions compute kernel
into a compiler contract that prevents contraction of the core SPIR-V
`OpFAdd`, `OpFSub`, `OpFMul` and `OpFDiv` results emitted for that kernel.
The default kernel remains unwrapped and keeps the default compiler behavior.
This is a per-kernel setting; it does not change a device-global precision
mode.

```julia
ordinary = my_kernel!(backend, 256)
separate = Lava.with_no_contraction(ordinary)
```

The policy is part of the callable type, compiler configuration and cache
identity. It therefore follows a kernel through ordinary public dispatch and
the direct compiler entry points. A required shader is checked by its bytes
when it is compiled, stored or read from the disk and frozen caches. A cache
key alone does not establish conformance: if any required arithmetic result
lacks its `NoContraction` decoration, the write is rejected or the cache read
misses and compilation can proceed normally. Positive cache reads still load
and link the valid shader through the usual path.

Conformance also requires a structurally valid SPIR-V module, checked with the
bundled SPIRV-Tools validator. Invalid required disk entries miss, and invalid
frozen memory entries are evicted. Invalid stores and links fail explicitly.
A validator process failure or timeout also fails explicitly. These checks add
work at compilation and shader-cache boundaries; ordinary warm launch-plan
hits reuse the validated linked kernel.

The contract composes canonically with
`Lava.with_float_controls(kernel, widths)` in either order:

```julia
both_a = Lava.with_float_controls(Lava.with_no_contraction(ordinary), (32,))
both_b = Lava.with_no_contraction(Lava.with_float_controls(ordinary, (32,)))
```

Both forms carry the same Float32 signed-zero/Inf/NaN-preservation mask and
NoContraction flag. FloatControls support is checked against the selected
device at dispatch time. NoContraction itself is an opt-in arithmetic
decoration, not a request for unsupported DenormPreserve modes.

An explicitly written fused operation such as `muladd(a, b, c)` or
`fma(a, b, c)` remains fused. The contract leaves that source intent intact;
it prevents the compiler from introducing contraction in the covered core
operations. A test should distinguish the two with independently rounded
Float32 references rather than assume LLVM IR syntax proves the device
operation.

This policy does not promise CPU-equivalent division rounding or exact transcendental results, a particular
implementation of GLSL extended instructions, denormal preservation, or exact
behavior for arithmetic outside the four covered core operations. It does not
claim a speedup or a uniform performance cost. Those properties require
separate contracts and measurements.

The maintained test exercises wrapper composition and adaptation,
compiler/cache identity, actual pre- and post-optimization SPIR-V coverage,
valid-decoration removal and cache rejection, positive cache restoration, and
poisoned public-dispatch repeats against a BigFloat oracle. The GPU numeric
check is a kernel correctness test, not model admission or a performance
benchmark.
