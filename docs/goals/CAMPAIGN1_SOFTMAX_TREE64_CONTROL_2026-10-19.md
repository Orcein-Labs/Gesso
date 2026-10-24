# Campaign 1: fixed-tree checked-softmax operator control

## Objective
Qualify a shape-independent64-laneFloat32row sum for checkedsoftmax,retaining existing exponentials,normalization,guards and source.

## Permitted files
New external versioned helper/privatecandidate/controlclones/rawcaptures/auditors/mutations and planning records only. Source6d3683c,910trackedfiles,allconsumed helpers/environments/captures,CPUoracle/targets/gates,qualifiedLava unchanged. No libs,source,vendor,dependency,globalprecision,compiler/link changes.

## Interfaces
Existing checked _lava_softmax! body under two new private names. Replace only denominator sum with explicitone-workgroup-per-row64-laneFloat32tree,using publicLavaNoContraction/FloatControls32;observedvariant adds readbacks forE/den only. Ordinaryoriginal remains available. No existing method replacement. Existing scalarGesso reduction already specifies64-laneAKblock_size;this applies one fixed realization to every rowcount.

## Invariants
Do not change mask,max,nativeFloat32exp,Float32denominator/output storage or existing division/finite checks. No output-ID specialcase or policy cap. Candidate sum consumes GPU exponentarrays and leaves inputs intact. Sharedlocal64tree with lane-strided ascendingloads,32/16/8/4/2/1steps,explicitbarriers,onegroupperrow;no atomics or subgroup-size assumption. No CPUactivation fallback. Source/all910hashes before/after.64MiBartifacts. Operatorcontrolonly;source/model/fullstress/production admission false.

## Tests
Qualifiedbaseline replay6/6reproducesordinary/modeloutputs. Mask/max/Eidentical;firstdifference denominator3ULP:whole0x40790754,partial0x40790751;independent64-laneF32tree equals0x40790751. AcceleratedKernels0.4.4nativeNDcode selects by-thread whenoutputcount>=reducedextent,otherwiseby-block;publicNDAPI has no forcingkeyword. Newcontrol testswhole127,prefix64,tail63,three repetitions;completecausalprobabilityassembly must be bitwiseidentical. Observed/privateunobserved candidate controls exact,inputpreserved,denominatorsbitwiseindependentCPU64-tree oncapturedE,actualNoContr/mode32shaderproof andeffective numericmutation. Retain numericalCPUcomparison as supplemental,not a new CPUbit-exactsoftmax claim. Onlyafterqualification scope process-localmodelcontrol then new bounded source candidate/fulloriginalstress;do not alter oracle or tolerances.

## Performance target
No inference-throughput claim. SameFloat32sum precision with a fixed parallel realization;cost/compile/alloc/memory/capability limits measured before eventualsourceadmission. Operatorrun1200seconds,20GiBRAM/4GiBswap,4800MiBGPU,16threads,BLAS/OMP1,sharedlocks.

## Expected artifact
Qualifiedpartition-invariantoperatorcandidate or retainednegative,exactkernelsource/shader/control evidence. No production/source patch inthisitem.

## Receipt
OPEN.Baselinestages independentlyqualify;candidateunwritten/unrun.

### Qualified operator result
Runner terminal0. Independent softmax-tree64-operator-validation-1.json passes9/9, all3 full probability partitions bitwise exact, all9 CPU tree checks exact,910source files unchanged,1,374,562artifact bytes. Query116 denominator is0x40790751 in both layouts. SPIR-V shader1b59c1c9 passes FloatControls32/NoContraction and Vulkan1.3 validation. softmax-tree64-operator-mutation-validation-1.json rejects an altered query116 denominator with repaired hash and retained green flags. Source/model/production admission remain false. Next: separately scoped model control.
